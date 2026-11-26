"""Revision job: background build (the flat re-embed), journal sync after the swap, abort
without swap, rollback, and the job's lock and status shared across processes."""

from __future__ import annotations

import asyncio

import pytest

from virag.api import revision as revision_api
from virag.coordination import REVISION_ON, REVISION_STATUS
from virag.enroll import Enroller
from virag.errors import ApiError
from virag.revision import RevisionManager
from virag.settings import EnrollSettings, RevisionSettings

from fastapi.testclient import TestClient

from conftest import FakeRetriever, FakeStore, FakeTiler, tiler_doc as _doc, build_app, lease_key, make_coordinator


def _manager(store, tiler, coordinator=None, batch_size=4):
    coordinator = coordinator if coordinator is not None else make_coordinator(store)
    retriever = FakeRetriever()
    enroll_settings = EnrollSettings(batch_size=batch_size)

    return RevisionManager(
        store,
        coordinator,
        tiler,
        retriever,
        Enroller(store, coordinator, tiler, retriever, enroll_settings),
        enroll_settings,
        RevisionSettings(revision_drain_timeout_sec=5),
    )


def _run_job(manager, start) -> dict:
    async def go():
        await start()
        await manager._task

        return manager.status

    return asyncio.run(go())


def test_revision_builds_swaps_and_syncs_journal():
    store, tiler = FakeStore(), FakeTiler()
    store.metadata = {"tiler_volume_id": "vol-test", "render_version": "rv-old"}
    tiler.stored = [_doc("a", 2), _doc("b", 1, complete=False), _doc("q", 1, quarantined=True), _doc("fresh", 1)]
    manager = _manager(store, tiler)

    listing = tiler.docs

    async def docs(*, filters=None):
        if not getattr(tiler, "enrolled_fresh", False):
            tiler.enrolled_fresh = True
            async with manager.coordinator.enrolling("fresh"):        # enrolled while building
                pass
        return await listing(filters=filters)

    tiler.docs = docs

    status = _run_job(manager, manager.start)

    assert status["state"] == "succeeded"
    assert status["total"] == 4 and status["skipped"] == 2 and status["reembedded"] == 2
    assert status["synced"] == 1
    assert store.swapped_to == "c__v2"
    assert store.previous_name == "c__v1"                               # retained for rollback
    assert store.metadata == {"tiler_volume_id": "vol-test", "render_version": "rv-test"}   # bound to the tiler's
    assert all(u["collection"] == "c__v2" for u in store.upserts)
    assert ("fresh", "c__v2") in zip(store.deletes, store.delete_collections)   # re-synced after swap
    assert {doc_id for doc_id, _ in tiler.metadata_patches} == {"fresh"}   # enrolled again, by the Enroller
    assert tiler.doc_stream_calls == ["fresh"]                          # from its stored tiles


def test_revision_releases_the_job_and_publishes_its_outcome():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)
    valkey = manager.coordinator.valkey

    _run_job(manager, manager.start)

    assert REVISION_ON not in valkey.values
    assert valkey.values[REVISION_STATUS][0]["state"] == "succeeded"


def test_revision_is_refused_while_the_tiler_is_not_serving():
    tiler = FakeTiler(info={"volume_id": "vol-test", "render_version": "rv-test",
                            "revision": {"state": "running", "serving": False}})
    manager = _manager(FakeStore(), tiler)

    with pytest.raises(ApiError) as exc:
        asyncio.run(manager.start())

    assert exc.value.status_code == 409
    assert manager._task is None
    assert REVISION_ON not in manager.coordinator.valkey.values             # the job handed back


def test_startup_mismatch_is_only_logged_by_default(caplog):
    store = FakeStore()
    store.metadata = {"tiler_volume_id": "vol-test", "render_version": "rv-old"}
    manager = _manager(store, FakeTiler())

    asyncio.run(manager.revise_if_stale(auto=False))

    assert manager._task is None
    assert "render_version mismatch" in caplog.text


def test_startup_mismatch_starts_the_revision_when_enabled():
    store, tiler = FakeStore(), FakeTiler()
    store.metadata = {"tiler_volume_id": "vol-test", "render_version": "rv-old"}
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)

    status = _run_job(manager, lambda: manager.revise_if_stale(auto=True))

    assert status["state"] == "succeeded"
    assert store.metadata["render_version"] == "rv-test"


def test_startup_mismatch_leaves_a_revision_running_elsewhere_alone():
    store, tiler = FakeStore(), FakeTiler()
    store.metadata = {"tiler_volume_id": "vol-test", "render_version": "rv-old"}
    other = make_coordinator(store)
    manager = _manager(store, tiler, make_coordinator(store, valkey=other.valkey))

    async def go():
        await other.acquire_revision()
        await manager.revise_if_stale(auto=True)                        # no raise: it runs there

    asyncio.run(go())

    assert manager._task is None


def test_startup_without_mismatch_starts_nothing():
    store = FakeStore()
    store.metadata = {"tiler_volume_id": "vol-test", "render_version": "rv-test"}
    manager = _manager(store, FakeTiler())

    asyncio.run(manager.revise_if_stale(auto=True))

    assert manager._task is None


def test_revision_aborts_when_the_tiler_render_version_changes_mid_build():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)

    async def go():
        await manager.start()
        tiler._info = {**tiler._info, "render_version": "rv-newer"}    # tiler redeployed meanwhile
        await manager._task

        return manager.status

    status = asyncio.run(go())

    assert status["state"] == "failed" and "changed during the build" in status["error"]
    assert not hasattr(store, "swapped_to")
    assert store.dropped == ["c__v2"]


def test_revision_with_failed_docs_does_not_swap():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1), _doc("bad", 1)]
    tiler.fail_tiles = {("bad", 0)}
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start)

    assert status["state"] == "failed"
    assert status["failed_doc_ids"] == ["bad"]
    assert not hasattr(store, "swapped_to")
    assert store.dropped == ["c__v2"]                                   # aborted build removed


def test_revision_that_lost_its_lease_does_not_swap():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)

    async def lease_runs_out():
        manager.coordinator.valkey.expire(lease_key(manager.coordinator))

    store.on_build = lease_runs_out

    status = _run_job(manager, manager.start)

    assert status["state"] == "failed" and "lost" in status["error"]
    assert not hasattr(store, "swapped_to")
    assert store.dropped == ["c__v2"]


def test_second_job_is_rejected_while_running():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)

    async def go():
        await manager.start()

        with pytest.raises(ApiError) as exc:
            await manager.start()

        await manager._task

        return exc.value.status_code

    assert asyncio.run(go()) == 409


def test_job_of_another_live_process_is_reported_not_repeated():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    first = _manager(store, tiler)
    second = _manager(store, tiler, make_coordinator(store, valkey=first.coordinator.valkey))

    async def go():
        await first.start()

        with pytest.raises(ApiError) as exc:
            await second.start()

        status = await second.get_status()                              # any process answers
        await first._task

        return exc.value, status

    error, status = asyncio.run(go())

    assert error.status_code == 409
    assert error.details["status"]["owner"]["token"] == first.coordinator.token
    assert status["kind"] == "revision" and status["owner"]["token"] == first.coordinator.token


def test_job_of_a_dead_process_is_reported_lost_and_taken_over():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    dead = make_coordinator(store)
    manager = _manager(store, tiler, make_coordinator(store, valkey=dead.valkey))

    async def go():
        await dead.acquire_revision()
        await dead.set_status({"kind": "revision", "state": "running", "owner": dead.identity})
        dead.valkey.expire(lease_key(dead))                             # its process was killed

        lost = await manager.get_status()

        await manager.start()
        await manager._task

        return lost, manager.status

    lost, status = asyncio.run(go())

    assert lost["state"] == "failed" and "owner lost" in lost["error"]
    assert status["state"] == "succeeded"
    assert status["owner"]["token"] == manager.coordinator.token


def test_rollback_requires_a_previous_collection():
    manager = _manager(FakeStore(), FakeTiler())

    with pytest.raises(ApiError) as exc:
        asyncio.run(manager.start_rollback())

    assert exc.value.status_code == 409
    assert REVISION_ON not in manager.coordinator.valkey.values


def test_rollback_is_refused_across_render_versions():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name, store.previous_name = "c__v2", "c__v1"
    store.collection_metadata = {"c__v1": {"render_version": "rv-before-the-tiler-upgrade"}}
    manager = _manager(store, tiler)

    with pytest.raises(ApiError) as exc:
        asyncio.run(manager.start_rollback())

    assert exc.value.status_code == 409 and "render_version" in exc.value.message
    assert manager._task is None


def test_rollback_resyncs_docs_enrolled_since_retirement():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name, store.previous_name = "c__v2", "c__v1"
    store.collection_metadata = {"c__v1": {"render_version": "rv-test", "retired_at": 100}}
    tiler.stored = [_doc("old", 1, enrolled_at=50), _doc("new", 1, enrolled_at=150)]
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start_rollback)

    assert status["state"] == "succeeded"
    assert status["collection"] == "c__v1" and status["abandoned"] == "c__v2"
    assert store.dropped == ["c__v2"]                                   # once the enrolls in flight ended
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["new"]
    assert {u["collection"] for u in store.upserts} == {"c__v1"}


def test_revision_api_starts_in_background_and_reports_status():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]

    app = build_app(store=store, tiler=tiler, retriever=FakeRetriever())
    app.include_router(revision_api.router)
    app.state.revision = _manager(store, tiler, app.state.coordinator)

    with TestClient(app) as client:
        started = client.post("/revision")
        assert started.status_code == 202
        assert started.json()["state"] == "running"

        status = client.get("/revision").json()
        assert status["kind"] == "revision"


# ---------------------------------------------------------------------------
# a job that stopped after moving the alias
# ---------------------------------------------------------------------------

def _left_running(store, tiler, *, journal=(), **status):
    """A job its process left `running` and died; returns the manager of another, live process."""
    dead = make_coordinator(store)
    valkey = dead.valkey
    valkey.values[REVISION_ON] = (dead.token, None)
    valkey.values[REVISION_STATUS] = ({"state": "running", "owner": dead.identity, **status}, None)
    valkey.sets["revision:journal"] = set(journal)
    valkey.expire(lease_key(dead))

    return _manager(store, tiler, make_coordinator(store, valkey=valkey))


def test_a_revision_left_after_its_swap_is_finished_by_another_process():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name, store.previous_name = "c__v2", "c__v1"        # the swap happened
    tiler.stored = [_doc("built", 1), _doc("fresh", 1)]                   # fresh: enrolled during the build
    manager = _left_running(
        store, tiler, journal={"fresh"}, kind="revision", collection="c__v2", phase="building", reembedded=1,
    )

    status = _run_job(manager, manager.resume_orphaned)

    assert status["state"] == "succeeded" and status["synced"] == 1
    assert status["owner"]["token"] == manager.coordinator.token and "resumed_at" in status
    assert status["reembedded"] == 1                                    # what the build published stays
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["fresh"]   # the tail only, no rebuild
    assert {u["collection"] for u in store.upserts} == {"c__v2"}
    assert REVISION_ON not in manager.coordinator.valkey.values


def test_a_revision_left_before_its_swap_is_not_resumed():
    store, tiler = FakeStore(), FakeTiler()                             # still serving c__v1
    manager = _left_running(store, tiler, journal={"fresh"}, kind="revision", collection="c__v2", phase="building")

    async def go():
        await manager.resume_orphaned()

        return await manager.get_status()

    status = asyncio.run(go())

    assert manager._task is None and store.upserts == []
    assert status["state"] == "failed" and "owner lost" in status["error"]   # POST /revision starts anew


def test_a_rollback_left_after_its_repoint_is_finished_by_another_process():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name = "c__v1"                                       # repointed back already
    tiler.stored = [_doc("old", 1, enrolled_at=50), _doc("new", 1, enrolled_at=150)]
    manager = _left_running(
        store, tiler, kind="rollback", collection="c__v1", abandoned="c__v2", retired_at=100, phase="repointing",
    )

    status = _run_job(manager, manager.resume_orphaned)

    assert status["state"] == "succeeded" and status["synced"] == 1
    assert store.dropped == ["c__v2"]
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["new"]


def test_a_job_another_live_process_runs_is_not_touched():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name = "c__v2"
    other = make_coordinator(store)
    manager = _manager(store, tiler, make_coordinator(store, valkey=other.valkey))

    async def go():
        await other.acquire_revision()
        await other.set_status({"kind": "revision", "state": "running", "owner": other.identity,
                                "collection": "c__v2", "phase": "syncing"})
        await manager.resume_orphaned()

        return await manager.get_status()

    status = asyncio.run(go())

    assert manager._task is None
    assert status["state"] == "running" and status["owner"]["token"] == other.token


def test_of_two_processes_one_finishes_the_job():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name = "c__v2"
    tiler.stored = [_doc("fresh", 1)]
    first = _left_running(store, tiler, journal={"fresh"}, kind="revision", collection="c__v2", phase="syncing")
    second = _manager(store, tiler, make_coordinator(store, valkey=first.coordinator.valkey))

    async def go():
        await first.resume_orphaned()
        await second.resume_orphaned()                                  # sees a live process on it
        await first._task
        await second.resume_orphaned()                                  # and afterwards a concluded job

        return first.status

    status = asyncio.run(go())

    assert status["state"] == "succeeded"
    assert second._task is None
    assert tiler.doc_stream_calls == ["fresh"]                          # synced once


def test_a_job_failed_by_a_backend_after_its_swap_stays_running_and_is_finished_later():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)
    listing = tiler.docs
    swap = store.swap_alias

    async def swap_then_the_tiler_goes_away(*args):
        await swap(*args)

        async def unavailable(*, filters=None):
            raise ApiError(status_code=503, message="tiler is unavailable")

        tiler.docs = unavailable

    store.swap_alias = swap_then_the_tiler_goes_away

    interrupted = dict(_run_job(manager, manager.start))

    assert interrupted["state"] == "running" and "tiler is unavailable" in interrupted["interrupted"]
    assert store.swapped_to == "c__v2" and store.dropped == []          # the swap stands
    assert REVISION_ON not in manager.coordinator.valkey.values         # released, to be taken again

    tiler.docs = listing                                           # the tiler is back
    status = _run_job(manager, manager.resume_orphaned)

    assert status["state"] == "succeeded" and "interrupted" not in status


def test_a_job_shut_down_after_its_swap_is_finished_by_the_next_process():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)
    manager.coordinator.DRAIN_POLL_SEC = 0.005
    valkey = manager.coordinator.valkey
    enrolling_elsewhere = make_coordinator(store, valkey=valkey)
    successor = _manager(store, tiler, make_coordinator(store, valkey=valkey))

    async def go():
        async with enrolling_elsewhere.enrolling("long"):               # the job waits for it after its swap
            await manager.start()

            while manager.status.get("phase") != "syncing":
                await asyncio.sleep(0.001)

            await manager.close()                                       # shutdown
            await manager.coordinator.close()

        left = dict(valkey.values[REVISION_STATUS][0])

        await successor.resume_orphaned()
        await successor._task

        return left, successor.status

    left, status = asyncio.run(go())

    assert left["state"] == "running" and "error" not in left          # not failed: to be finished
    assert status["state"] == "succeeded" and status["owner"]["token"] == successor.coordinator.token


def test_a_job_shut_down_while_building_is_failed():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("a", 1)]
    manager = _manager(store, tiler)
    reached = asyncio.Event()

    async def stall():
        reached.set()
        await asyncio.sleep(60)

    store.on_build = stall

    async def go():
        await manager.start()
        await reached.wait()
        await manager.close()

        return manager.status

    status = asyncio.run(go())

    assert status["state"] == "failed" and "cancelled" in status["error"]
    assert not hasattr(store, "swapped_to")


# ---------------------------------------------------------------------------
# the sync after a swap
# ---------------------------------------------------------------------------

def _sync(manager, doc_ids):
    manager.status = {"kind": "revision"}

    return manager._sync(set(doc_ids))


def test_sync_mirrors_the_tiler_for_the_given_docs():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [
        _doc("live", 1),
        _doc("pending", 1, enrolled_at=0),                              # rolled-back enroll
        _doc("partial", 1, complete=False),
        _doc("untouched", 1),
    ]
    manager = _manager(store, tiler)

    failed = asyncio.run(_sync(manager, {"live", "pending", "partial", "gone"}))

    assert failed == set()
    assert sorted(store.deletes) == ["gone", "live", "partial", "pending"]  # all dropped first
    assert set(store.delete_collections) == {"c__v1"}                   # the serving collection
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["live"]
    assert tiler.doc_stream_calls == ["live"]


def test_sync_leaves_a_document_being_enrolled_to_its_enroll():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("live", 1), _doc("enrolling", 1)]
    manager = _manager(store, tiler)

    async def go():
        async with manager.coordinator.enrolling("enrolling"):
            return await _sync(manager, {"live", "enrolling"})

    assert asyncio.run(go()) == set()                                   # busy is not failed
    assert store.deletes == ["live"]
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["live"]


def test_a_doc_that_fails_to_sync_is_reported_and_the_rest_go_on():
    store, tiler = FakeStore(), FakeTiler()
    tiler.stored = [_doc("bad", 2), _doc("good", 1)]
    tiler.fail_tiles = {("bad", 1)}
    manager = _manager(store, tiler)

    failed = asyncio.run(_sync(manager, {"bad", "good"}))

    assert failed == {"bad"}
    assert {p["doc_id"] for u in store.upserts for p in u["payloads"]} >= {"good"}
    assert tiler.metadata_patches[-1] == ("good", tiler.metadata_patches[-1][1])
    assert all(patch["virag_enrolled_at"] == 0 for doc_id, patch in tiler.metadata_patches if doc_id == "bad")


# ---------------------------------------------------------------------------
# the flat re-embed of a build
# ---------------------------------------------------------------------------

def test_batches_cross_document_boundaries():
    store, tiler = FakeStore(), FakeTiler()
    docs = [_doc("a", 1), _doc("b", 2), _doc("c", 1)]

    failed = asyncio.run(_manager(store, tiler, batch_size=2).reembed(docs, "c__v2"))

    assert failed == set()
    batches = sorted([p["doc_id"], p["tile_num"]] for u in store.upserts for p in u["payloads"])
    assert batches == [["a", 0], ["b", 0], ["b", 1], ["c", 0]]
    assert len(store.upserts) == 2                                      # two full batches, not three
    assert all(u["collection"] == "c__v2" for u in store.upserts)
    assert tiler.metadata_patches == []                                 # a build touches no markers


def test_failed_batch_is_retried_and_only_the_bad_doc_fails():
    store, tiler = FakeStore(), FakeTiler()
    docs = [_doc("a", 1), _doc("b", 2), _doc("c", 1)]
    tiler.fail_tiles = {("b", 1)}                                       # shares a batch with c

    failed = asyncio.run(_manager(store, tiler, batch_size=2).reembed(docs, "c__v2"))

    assert failed == {"b"}
    stored = {p["doc_id"] for u in store.upserts for p in u["payloads"]}
    assert {"a", "c"} <= stored                                         # c recovered on retry
