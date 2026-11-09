"""Revision job: background build, journal sync after the swap, abort without swap,
rollback, and the job's lock and status shared across processes."""

from __future__ import annotations

import asyncio

import pytest

from virag.api import revision as revision_api
from virag.coordination import REVISION_ON, REVISION_STATUS
from virag.errors import ApiError
from virag.revision import RevisionManager
from virag.settings import EnrollSettings, ReconcileSettings

from fastapi.testclient import TestClient

from conftest import FakeRetriever, FakeStore, FakeTiler, build_app, lease_key, make_coordinator


def _doc(doc_id, tile_count, *, complete=True, enrolled_at=1):
    return {
        "doc_id": doc_id, "doc_type": "PDF", "tile_count": tile_count, "complete": complete,
        "render_version": "rv", "source_url": None,
        "metadata": {"name": doc_id, "virag_started_at": 1, "virag_enrolled_at": enrolled_at},
    }


def _manager(store, tiler, coordinator=None):
    return RevisionManager(
        store,
        coordinator if coordinator is not None else make_coordinator(store),
        tiler,
        FakeRetriever(),
        EnrollSettings(batch_size=4),
        ReconcileSettings(self_heal=True, interval_sec=0, pending_ttl_sec=5),
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
    tiler.docs = [_doc("a", 2), _doc("b", 1, complete=False), _doc("q", 1) | {"quarantined": True}, _doc("fresh", 1)]
    manager = _manager(store, tiler)

    listing = tiler.iter_docs

    async def iter_docs(*, filters=None):
        if not getattr(tiler, "enrolled_fresh", False):
            tiler.enrolled_fresh = True
            async with manager.coordinator.writer(["fresh"]):         # enrolled while building
                pass
        return await listing(filters=filters)

    tiler.iter_docs = iter_docs

    status = _run_job(manager, manager.start)

    assert status["state"] == "succeeded"
    assert status["total"] == 4 and status["skipped"] == 2 and status["reembedded"] == 2
    assert status["synced"] == 1
    assert store.swapped_to == "c__v2"
    assert store.previous_name == "c__v1"                               # retained for rollback
    assert store.metadata == {"tiler_volume_id": "vol-test", "render_version": "rv-test"}   # bound to the tiler's
    assert all(u["collection"] == "c__v2" for u in store.upserts)
    assert ("fresh", "c__v2") in zip(store.deletes, store.delete_collections)   # re-synced after swap
    assert tiler.metadata_patches == []                                 # revision never touches markers


def test_revision_releases_the_job_and_publishes_its_outcome():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1), _doc("bad", 1)]
    tiler.fail_tiles = {("bad", 0)}
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start)

    assert status["state"] == "failed"
    assert status["failed_doc_ids"] == ["bad"]
    assert not hasattr(store, "swapped_to")
    assert store.dropped == ["c__v2"]                                   # aborted build removed


def test_revision_that_lost_its_lease_does_not_swap():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1)]
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
    tiler.docs = [_doc("a", 1)]
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
    store.collection_metadata = {"c__v1": {"render_version": "rv-test"}}
    store.rollback_retired_at = 100
    tiler.docs = [_doc("old", 1, enrolled_at=50), _doc("new", 1, enrolled_at=150)]
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start_rollback)

    assert status["state"] == "succeeded"
    assert status["collection"] == "c__v1" and status["abandoned"] == "c__v2"
    assert store.dropped == ["c__v2"]                                   # once its writers drained
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["new"]
    assert {u["collection"] for u in store.upserts} == {"c__v1"}


def test_revision_api_starts_in_background_and_reports_status():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 1)]

    app = build_app(store=store, tiler=tiler, retriever=FakeRetriever())
    app.include_router(revision_api.router)
    app.state.revision = _manager(store, tiler, app.state.coordinator)

    with TestClient(app) as client:
        started = client.post("/revision")
        assert started.status_code == 202
        assert started.json()["state"] == "running"

        status = client.get("/revision").json()
        assert status["kind"] == "revision"
