"""reconcile: Qdrant ⊆ Tiler enforcement, and abandoned enrolls finished through the Enroller."""

from __future__ import annotations

import asyncio

from virag.enroll import Enroller
from virag.errors import ApiError
from virag.reconcile import reconcile
from virag.settings import EnrollSettings, ReconcileSettings

from conftest import FakeRetriever, FakeStore, FakeTiler, point_id, tiler_doc as _doc, lease_key, make_coordinator


def _reconcile(store, tiler, coordinator=None, *, self_heal=True, retriever=None):
    """One reconcile pass, as the coroutine to run."""
    coordinator = coordinator if coordinator is not None else make_coordinator(store)
    enroller = Enroller(store, coordinator, tiler, retriever or FakeRetriever(), EnrollSettings(batch_size=16))

    return reconcile(
        store, coordinator, tiler, enroller, ReconcileSettings(self_heal=self_heal, interval_sec=0),
    )


def _enrolled_docs(store) -> list[str]:
    return sorted({p["doc_id"] for u in store.upserts for p in u["payloads"]})


# ---------------------------------------------------------------------------
# Qdrant ⊆ Tiler
# ---------------------------------------------------------------------------

def test_reconcile_deletes_dangling_points():
    store, tiler = FakeStore(), FakeTiler()
    store._doc_ids = {"kept", "dangling"}                       # dangling has no tiler doc
    tiler.stored = [_doc("kept")]

    asyncio.run(_reconcile(store, tiler))

    assert "dangling" in store.deletes
    assert "kept" not in store.deletes


def test_reconcile_deletes_points_of_quarantined_docs():
    store, tiler = FakeStore(), FakeTiler()
    store._doc_ids = {"kept", "quarantined", "reenrolling"}
    tiler.stored = [
        _doc("kept"),
        _doc("quarantined", quarantined=True),
        _doc("reenrolling", quarantined=True),
    ]
    coordinator = make_coordinator(store)

    async def go():
        async with coordinator.enrolling("reenrolling"):       # a re-enroll is lifting it
            await _reconcile(store, tiler, coordinator)

    asyncio.run(go())

    assert store.deletes == ["quarantined"]


def test_reconcile_runs_only_in_the_leading_process():
    store, tiler = FakeStore(), FakeTiler()
    store._doc_ids = {"dangling"}
    leader = make_coordinator(store)
    follower = make_coordinator(store, valkey=leader.valkey)

    asyncio.run(_reconcile(store, tiler, leader))
    store.deletes.clear()
    store._doc_ids = {"dangling"}

    asyncio.run(_reconcile(store, tiler, follower))

    assert store.deletes == []                                          # the leader's to do


# ---------------------------------------------------------------------------
# abandoned enrolls
# ---------------------------------------------------------------------------

def test_an_abandoned_enroll_is_finished_from_the_stored_tiles():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("doc-a", 2, enrolled_at=0)]

    asyncio.run(_reconcile(store, tiler))

    assert tiler.doc_stream_calls == ["doc-a"]                          # the tiler's stream of the stored doc
    assert tiler.get_tile_calls == []                                   # not tile by tile

    assert len(store.upserts) == 1
    up = store.upserts[0]
    assert up["ids"] == [point_id("doc-a", 0), point_id("doc-a", 1)]    # an enroll: the same ids
    assert set(up["payloads"][0]) == {"doc_id", "tile_num", "image_hash", "render_version"}
    assert up["collection"] == "c__v1"                                  # pinned to the serving one

    marks = [patch["virag_enrolled_at"] for doc_id, patch in tiler.metadata_patches if doc_id == "doc-a"]
    assert marks[0] == 0 and marks[-1] > 0                              # pending, then promoted
    assert store.deletes == ["doc-a"]                                   # its partial points cleared first


def test_a_pending_doc_is_abandoned_whatever_its_age():
    """No age to wait out: pending and held by nobody means its enroll is over."""
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("just-failed", 1, enrolled_at=0)]

    asyncio.run(_reconcile(store, tiler))

    assert _enrolled_docs(store) == ["just-failed"]


def test_reconcile_leaves_a_document_being_enrolled_to_its_enroll():
    """Neither its points (dangling as the listing saw them) nor its pending mark are
    reconcile's to touch while an enroll holds the document."""
    store, tiler = FakeStore(), FakeTiler()
    store._doc_ids = {"dangling", "enrolling"}                  # neither is in the tiler's listing
    tiler.pending = [_doc("enrolling", 1, enrolled_at=0), _doc("abandoned", 1, enrolled_at=0)]
    coordinator = make_coordinator(store)
    other = make_coordinator(store, valkey=coordinator.valkey)

    async def go():
        async with other.enrolling("enrolling"):
            await _reconcile(store, tiler, coordinator)

    asyncio.run(go())

    assert sorted(store.deletes) == ["abandoned", "dangling"]
    assert _enrolled_docs(store) == ["abandoned"]
    assert tiler.doc_stream_calls == ["abandoned"]


def test_a_document_taken_after_reconcile_looked_is_left_alone(monkeypatch):
    """An enroll that takes the document between reconcile's look at the keys and its own
    attempt: taking it is refused, and nothing of the document is touched."""
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("raced", 1, enrolled_at=0)]
    coordinator = make_coordinator(store)
    other = make_coordinator(store, valkey=coordinator.valkey)
    listing = tiler.docs

    async def docs(*, filters=None):
        if filters:
            coordinator.valkey.values["enroll:raced"] = (other.token, None)   # taken meanwhile
        return await listing(filters=filters)

    tiler.docs = docs

    asyncio.run(_reconcile(store, tiler, coordinator))

    assert store.deletes == [] and store.upserts == [] and tiler.metadata_patches == []


def test_a_document_of_a_dead_process_is_abandoned():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("orphan", 1, enrolled_at=0)]
    coordinator = make_coordinator(store)
    dead = make_coordinator(store, valkey=coordinator.valkey)
    coordinator.valkey.values["enroll:orphan"] = (dead.token, None)
    coordinator.valkey.expire(lease_key(dead))                          # it died mid-enroll

    asyncio.run(_reconcile(store, tiler, coordinator))

    assert _enrolled_docs(store) == ["orphan"]


def test_self_heal_is_journaled_by_a_running_revision():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("doc-a", 1, enrolled_at=0)]
    coordinator = make_coordinator(store)

    async def go():
        await coordinator.acquire_revision()
        await coordinator.start_journal()
        await _reconcile(store, tiler, coordinator)

        return await coordinator.journal()

    assert asyncio.run(go()) == {"doc-a"}


def test_a_doc_the_tiler_does_not_serve_only_loses_its_partial_points():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [
        _doc("incomplete", 0, complete=False, enrolled_at=0),
        _doc("quarantined", 1, enrolled_at=0, quarantined=True),
    ]

    asyncio.run(_reconcile(store, tiler))

    assert sorted(store.deletes) == ["incomplete", "quarantined"]       # partial cleared
    assert store.upserts == [] and tiler.doc_stream_calls == []         # left for a re-enroll
    assert tiler.metadata_patches == []


def test_self_heal_off_only_clears_partial_points():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("doc-a", 2, enrolled_at=0)]

    asyncio.run(_reconcile(store, tiler, self_heal=False))

    assert store.deletes == ["doc-a"]
    assert store.upserts == [] and tiler.metadata_patches == []


def test_a_failed_self_heal_leaves_the_doc_pending_and_goes_on():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("bad", 2, enrolled_at=0), _doc("good", 1, enrolled_at=0)]
    tiler.fail_tiles = {("bad", 1)}                                     # its stream breaks mid-way
    coordinator = make_coordinator(store)

    asyncio.run(_reconcile(store, tiler, coordinator))

    assert store.deletes == ["bad", "bad", "good"]                      # bad: cleared, then rolled back
    assert all(patch["virag_enrolled_at"] == 0 for doc_id, patch in tiler.metadata_patches if doc_id == "bad")
    assert _enrolled_docs(store) == ["good"]                            # the pass went on
    assert not any(name.startswith("enroll:") for name in coordinator.valkey.values)   # both released


def test_a_refused_stream_fails_only_its_doc():
    store, tiler = FakeStore(), FakeTiler()
    tiler.pending = [_doc("gone", 1, enrolled_at=0), _doc("good", 1, enrolled_at=0)]
    stream = tiler.doc_stream

    def doc_stream(doc_id):
        if doc_id == "gone":
            return _refused()

        return stream(doc_id)

    async def _refused():
        raise ApiError(status_code=503, message="Upstream failed", details={"upstream_status": 404})
        yield  # pragma: no cover - makes this an async generator

    tiler.doc_stream = doc_stream

    asyncio.run(_reconcile(store, tiler))

    assert _enrolled_docs(store) == ["good"]
    assert store.deletes == ["good"]                                    # nothing of `gone` was begun
