"""Coordinator over an in-memory Valkey: leases, documents being enrolled and the
journal, drain, the revision job, the reconcile lead."""

from __future__ import annotations

import asyncio
import signal

import pytest

from virag.coordination import RECONCILE_ON, REVISION_ON, EnrollInProgress
from virag.errors import ApiError

from conftest import FakeStore, FakeValkey, lease_key, make_coordinator


def _pair():
    """Two processes sharing one Valkey and one Qdrant."""
    store = FakeStore()
    a = make_coordinator(store)
    b = make_coordinator(store, valkey=a.valkey)

    return store, a, b


# ---------------------------------------------------------------------------
# lease
# ---------------------------------------------------------------------------

def test_register_takes_a_lease_with_the_ttl():
    valkey = FakeValkey()
    coordinator = make_coordinator(valkey=valkey)
    valkey.values.clear()

    asyncio.run(coordinator.register())

    value, expires_at = valkey.values[lease_key(coordinator)]
    assert value["token"] == coordinator.token and expires_at is not None


def test_heartbeat_renews_the_lease():
    coordinator = make_coordinator(ttl=0.03)

    async def go():
        await coordinator.register()
        task = asyncio.create_task(coordinator.heartbeat())
        await asyncio.sleep(0.1)                                        # past the ttl: renewed
        alive = await coordinator.valkey.exists(lease_key(coordinator))
        task.cancel()

        return alive

    assert asyncio.run(go()) is True


def test_heartbeat_on_an_expired_lease_stops_the_process(monkeypatch):
    coordinator = make_coordinator(ttl=0.03)
    killed = []
    monkeypatch.setattr("virag.coordination.os.kill", lambda pid, sig: killed.append(sig))

    async def go():
        coordinator.valkey.expire(lease_key(coordinator))               # the loop stalled past the ttl
        await coordinator.heartbeat()                                   # returns once it gave up

    asyncio.run(go())

    assert killed == [signal.SIGTERM]
    assert lease_key(coordinator) not in coordinator.valkey.values      # never revived


def test_heartbeat_retries_while_valkey_is_unreachable(monkeypatch):
    coordinator = make_coordinator(ttl=0.03)
    killed = []
    monkeypatch.setattr("virag.coordination.os.kill", lambda pid, sig: killed.append(sig))

    async def go():
        coordinator.valkey.down = True
        task = asyncio.create_task(coordinator.heartbeat())
        await asyncio.sleep(0.05)
        coordinator.valkey.down = False
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(go())

    assert killed == []


def test_close_hands_back_the_lease_and_the_reconcile_lead():
    _, a, _ = _pair()

    async def go():
        assert await a.lead_reconcile()
        await a.close()

    asyncio.run(go())

    assert lease_key(a) not in a.valkey.values
    assert RECONCILE_ON not in a.valkey.values


# ---------------------------------------------------------------------------
# documents being enrolled, journal, drain
# ---------------------------------------------------------------------------

def test_enrolling_holds_the_document_pinned_to_the_serving_collection():
    _, a, _ = _pair()

    async def go():
        async with a.enrolling("doc") as physical:
            held = dict(a.valkey.values)

        return physical, held

    physical, held = asyncio.run(go())

    assert physical == "c__v1"
    assert held["enroll:doc"][0] == a.token                             # the name: what, the value: who
    assert "enroll:doc" not in a.valkey.values                          # released


def test_a_document_is_enrolled_by_one_request_at_a_time():
    _, a, b = _pair()

    async def go():
        refused = []

        async with a.enrolling("doc"):
            for coordinator in (a, b):                                  # the same process, and another
                with pytest.raises(EnrollInProgress) as exc:
                    async with coordinator.enrolling("doc"):
                        pass

                refused.append((exc.value.status_code, exc.value.code))

            still = await a.holds_enroll("doc")                         # a refused enroll releases nothing

            async with b.enrolling("other"):                            # another document is free
                pass

        async with b.enrolling("doc"):                                  # and this one, once released
            pass

        return refused, still

    refused, still = asyncio.run(go())

    assert refused == [(409, "ENROLL_IN_PROGRESS")] * 2
    assert still is True


def test_a_document_of_a_dead_process_is_taken_over():
    _, stalled, b = _pair()

    async def go():
        stale = stalled.enrolling("doc")
        await stale.__aenter__()
        stalled.valkey.expire(lease_key(stalled))                       # stalled past its lease

        async with b.enrolling("doc"):
            holds = (await stalled.holds_enroll("doc"), await b.holds_enroll("doc"))
            await stale.__aexit__(None, None, None)                     # the stale enroll ends meanwhile

            return holds, await b.holds_enroll("doc")

    holds, kept = asyncio.run(go())

    assert holds == (False, True)
    assert kept is True                                                 # it released nothing of b's


def test_enrolling_now_spans_processes_and_drops_dead_ones():
    _, a, b = _pair()

    async def go():
        async with a.enrolling("from-a"):
            async with b.enrolling("from-b"):
                both = await a.enrolling_now()
                a.valkey.expire(lease_key(b))                           # b's process died mid-enroll
                alive = await a.enrolling_now()
                left = sorted(name for name in a.valkey.values if name.startswith("enroll:"))

        return both, alive, left

    both, alive, left = asyncio.run(go())

    assert both == {"from-a": a.token, "from-b": b.token}
    assert alive == {"from-a": a.token}
    assert left == ["enroll:from-a"]                                    # the dead one's key is gone


def test_journal_holds_enrolls_in_flight_and_the_ones_starting_after():
    _, a, b = _pair()

    async def go():
        async with b.enrolling("early"):                                # before the job, elsewhere
            assert await a.acquire_revision() is None
            await a.start_journal()

            async with b.enrolling("during"):
                pass

        return await a.journal()

    assert asyncio.run(go()) == {"early", "during"}


def test_enrolls_without_a_job_are_not_journaled():
    _, a, _ = _pair()

    async def go():
        async with a.enrolling("no-job"):
            pass

        return await a.journal()

    assert asyncio.run(go()) == set()


def test_start_journal_drops_the_last_jobs():
    _, a, _ = _pair()

    async def go():
        await a.acquire_revision()
        async with a.enrolling("last-job"):
            pass
        await a.start_journal()

        return await a.journal()

    assert asyncio.run(go()) == set()


def test_drain_waits_for_the_enrolls_in_flight_and_no_later_one():
    _, a, b = _pair()
    a.DRAIN_POLL_SEC = 0.005

    async def go():
        release = asyncio.Event()

        async def slow_enroll():
            async with b.enrolling("long"):                             # may write the old collection
                await release.wait()

        task = asyncio.create_task(slow_enroll())
        await asyncio.sleep(0)

        assert await a.drain(timeout=0.02) is False                     # still enrolling

        draining = asyncio.create_task(a.drain(timeout=1))
        await asyncio.sleep(0)                                          # it has seen who is in flight

        async with a.enrolling("late"):                                 # taken after the swap
            release.set()
            assert await draining is True                               # not waited for

        await task

    asyncio.run(go())


def test_drain_ignores_documents_of_dead_processes():
    _, a, b = _pair()

    async def go():
        async with b.enrolling("orphan"):
            a.valkey.expire(lease_key(b))

            return await a.drain(timeout=0.05)

    assert asyncio.run(go()) is True


def test_enrolling_fails_as_not_ready_while_valkey_is_unreachable():
    _, a, _ = _pair()
    a.valkey.down = True

    async def go():
        async with a.enrolling("doc"):
            pass

    with pytest.raises(ApiError) as exc:
        asyncio.run(go())

    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# revision job
# ---------------------------------------------------------------------------

def test_acquire_refuses_a_live_holder_with_its_status():
    _, a, b = _pair()

    async def go():
        assert await a.acquire_revision() is None
        await a.set_status({"kind": "revision", "state": "running", "owner": a.identity})

        return await b.acquire_revision()

    holder = asyncio.run(go())

    assert holder["owner"]["token"] == a.token
    assert a.valkey.values[REVISION_ON][0] == a.token


def test_acquire_takes_over_from_a_dead_holder_once():
    store, dead, b = _pair()
    c = make_coordinator(store, valkey=dead.valkey)

    async def go():
        await dead.acquire_revision()
        dead.valkey.expire(lease_key(dead))

        return await b.acquire_revision(), await c.acquire_revision()

    taken, refused = asyncio.run(go())

    assert taken is None                                                # b took it over
    assert refused is not None                                          # c sees b, alive
    assert dead.valkey.values[REVISION_ON][0] == b.token


def test_a_process_that_lost_the_job_neither_owns_nor_publishes_it():
    _, a, b = _pair()

    async def go():
        await a.acquire_revision()
        a.valkey.expire(lease_key(a))
        await b.acquire_revision()                                      # taken over
        await b.set_status({"state": "running", "owner": b.identity})

        owns = await a.owns_revision()
        await a.set_status({"state": "succeeded", "owner": a.identity})
        await a.release_revision()

        return owns, await b.get_status()

    owns, status = asyncio.run(go())

    assert owns is False
    assert status["owner"]["token"] == b.token and status["state"] == "running"
    assert a.valkey.values[REVISION_ON][0] == b.token


def test_status_of_a_job_whose_owner_died_is_failed():
    _, a, b = _pair()

    async def go():
        await a.acquire_revision()
        await a.set_status({"kind": "revision", "state": "running", "owner": a.identity})
        a.valkey.expire(lease_key(a))

        return await b.get_status()

    status = asyncio.run(go())

    assert status["state"] == "failed" and "owner lost" in status["error"]


def test_status_without_any_job_is_idle():
    _, a, _ = _pair()

    assert asyncio.run(a.get_status()) == {"state": "idle"}


# ---------------------------------------------------------------------------
# reconcile lead
# ---------------------------------------------------------------------------

def test_reconcile_lead_is_held_and_taken_over_from_a_dead_leader():
    _, a, b = _pair()

    async def go():
        first = (await a.lead_reconcile(), await b.lead_reconcile())
        again = await a.lead_reconcile()
        a.valkey.expire(lease_key(a))
        after = (await b.lead_reconcile(), await a.lead_reconcile())

        return first, again, after

    first, again, after = asyncio.run(go())

    assert first == (True, False)
    assert again is True
    assert after == (True, False)
