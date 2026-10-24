"""Revision job: background build, journal sync after the swap, abort without swap,
rollback; plus the QdrantStore writers journal it relies on."""

from __future__ import annotations

import asyncio

import pytest

from virag.api import revision as revision_api
from virag.clients.qdrant_store import QdrantStore
from virag.errors import ApiError
from virag.revision import RevisionManager
from virag.settings import EnrollSettings, ReconcileSettings

from fastapi.testclient import TestClient

from conftest import FakeRetriever, FakeStore, FakeTiler, build_app, make_store


def _doc(doc_id, tile_count, *, complete=True, enrolled_at=1):
    return {
        "doc_id": doc_id, "doc_type": "PDF", "tile_count": tile_count, "complete": complete,
        "render_version": "rv", "source_url": None,
        "metadata": {"name": doc_id, "virag_started_at": 1, "virag_enrolled_at": enrolled_at},
    }


def _manager(store, tiler):
    return RevisionManager(
        store,
        tiler,
        FakeRetriever(),
        EnrollSettings(batch_size=4),
        ReconcileSettings(self_heal=True, interval_sec=0, pending_ttl_sec=5),
    )


def _run_job(manager, start) -> dict:
    async def go():
        start()
        await manager._task

        return manager.status

    return asyncio.run(go())


def test_revision_builds_swaps_and_syncs_journal():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 2), _doc("b", 1, complete=False), _doc("fresh", 1)]
    store.journal_seen = {"fresh"}                                      # enrolled while building
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start)

    assert status["state"] == "succeeded"
    assert status["total"] == 3 and status["skipped"] == 1 and status["reembedded"] == 2
    assert status["synced"] == 1
    assert store.swapped_to == "c__v2"
    assert store.previous_name == "c__v1"                               # retained for rollback
    assert all(u["collection"] == "c__v2" for u in store.upserts)
    assert ("fresh", "c__v2") in zip(store.deletes, store.delete_collections)   # re-synced after swap
    assert tiler.metadata_patches == []                                 # revision never touches markers


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
    assert store._journal is None                                       # journal closed


def test_second_job_is_rejected_while_running():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 1)]
    manager = _manager(store, tiler)

    async def go():
        manager.start()

        with pytest.raises(ApiError) as exc:
            manager.start()

        await manager._task

        return exc.value.status_code

    assert asyncio.run(go()) == 409


def test_rollback_requires_a_previous_collection():
    manager = _manager(FakeStore(), FakeTiler())

    with pytest.raises(ApiError) as exc:
        manager.start_rollback()

    assert exc.value.status_code == 409


def test_rollback_resyncs_docs_enrolled_since_retirement():
    store, tiler = FakeStore(), FakeTiler()
    store.physical_name, store.previous_name = "c__v2", "c__v1"
    store.rollback_retired_at = 100
    tiler.docs = [_doc("old", 1, enrolled_at=50), _doc("new", 1, enrolled_at=150)]
    manager = _manager(store, tiler)

    status = _run_job(manager, manager.start_rollback)

    assert status["state"] == "succeeded"
    assert status["collection"] == "c__v1" and status["abandoned"] == "c__v2"
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["new"]


def test_revision_api_starts_in_background_and_reports_status():
    store, tiler = FakeStore(), FakeTiler()
    tiler.docs = [_doc("a", 1)]

    app = build_app(store=store, tiler=tiler, retriever=FakeRetriever())
    app.include_router(revision_api.router)
    app.state.revision = _manager(store, tiler)

    with TestClient(app) as client:
        started = client.post("/revision")
        assert started.status_code == 202
        assert started.json()["state"] == "running"

        status = client.get("/revision").json()
        assert status["kind"] == "revision"


# ---------------------------------------------------------------------------
# QdrantStore writers journal (no network: the client is never called)
# ---------------------------------------------------------------------------

def _store() -> QdrantStore:
    store = make_store()
    store.physical_name = "c__v1"

    return store


def test_journal_includes_writers_in_flight_and_new_ones():
    store = _store()

    async def go():
        async with store.writer("early"):
            store.start_journal()

            async with store.writer("during"):
                pass

        async with store.writer("during-2"):
            pass

        return store.stop_journal()

    assert asyncio.run(go()) == {"early", "during", "during-2"}


def test_writers_after_stop_are_not_journaled():
    store = _store()

    async def go():
        store.start_journal()
        journal = store.stop_journal()

        async with store.writer("late"):
            pass

        return journal

    assert asyncio.run(go()) == set()


def test_drain_waits_for_writers_of_the_old_collection():
    store = _store()

    async def go():
        release = asyncio.Event()

        async def slow_writer():
            async with store.writer("long"):
                await release.wait()

        task = asyncio.create_task(slow_writer())
        await asyncio.sleep(0)

        store.physical_name = "c__v2"                                   # swapped meanwhile

        async with store.writer("new-collection"):
            assert await store.drain("c__v1", timeout=0.01) is False   # still writing

            release.set()
            assert await store.drain("c__v1", timeout=1) is True       # ignores c__v2 writers

        await task

    asyncio.run(go())
