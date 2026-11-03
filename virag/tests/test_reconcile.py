"""reconcile: Qdrant ⊆ Tiler enforcement, self-heal of stale pending enrolls, flat re-embed."""

from __future__ import annotations

import asyncio

from virag.api.enroll import point_id
from virag.reconcile import reconcile, reembed_docs, sync_docs
from virag.settings import EnrollSettings, ReconcileSettings

from conftest import FakeRetriever, FakeStore, FakeTiler, make_coordinator


_ENROLL = EnrollSettings(batch_size=16)


def _settings(self_heal=True):
    return ReconcileSettings(self_heal=self_heal, interval_sec=0, pending_ttl_sec=3600)


def _doc(doc_id, tile_count, *, complete=True, enrolled_at=1):
    return {
        "doc_id": doc_id, "doc_type": "PDF", "tile_count": tile_count, "complete": complete,
        "render_version": "rv", "source_url": None,
        "metadata": {"name": doc_id, "virag_started_at": 1, "virag_enrolled_at": enrolled_at},
    }


def test_reconcile_deletes_dangling_points():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    store._doc_ids = {"kept", "dangling"}                       # dangling has no tiler doc
    tiler.docs = [{"doc_id": "kept"}]

    asyncio.run(reconcile(store, make_coordinator(store), tiler, retriever, _settings(), _ENROLL))

    assert "dangling" in store.deletes
    assert "kept" not in store.deletes


def test_reconcile_deletes_points_of_quarantined_docs():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    store._doc_ids = {"kept", "quarantined", "reenrolling"}
    tiler.docs = [
        {"doc_id": "kept"},
        {"doc_id": "quarantined", "quarantined": True},
        {"doc_id": "reenrolling", "quarantined": True},
    ]
    coordinator = make_coordinator(store)

    async def go():
        async with coordinator.writer(["reenrolling"]):        # a re-enroll is lifting it
            await reconcile(store, coordinator, tiler, retriever, _settings(), _ENROLL)

    asyncio.run(go())

    assert store.deletes == ["quarantined"]


def test_reconcile_self_heals_complete_pending():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    tiler.pending_docs = [{
        "doc_id": "doc-a", "doc_type": "PDF", "tile_count": 2, "complete": True,
        "render_version": "rv-test", "source_url": None,
        "metadata": {"name": "A", "virag_started_at": 1, "virag_enrolled_at": 0},
    }]

    asyncio.run(reconcile(store, make_coordinator(store), tiler, retriever, _settings(), _ENROLL))

    assert len(store.upserts) == 1
    up = store.upserts[0]
    assert up["ids"] == [point_id("doc-a", 0), point_id("doc-a", 1)]     # matches enroll's ids
    assert up["payloads"][0]["metadata"] == {"name": "A"}               # virag_* markers stripped
    assert tiler.metadata_patches[-1][0] == "doc-a"                     # pending promoted
    assert tiler.metadata_patches[-1][1]["virag_enrolled_at"] > 0
    assert up["collection"] == "c__v1"                                  # pinned to the serving one


def test_self_heal_is_journaled_by_a_running_revision():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    tiler.pending_docs = [_doc("doc-a", 1, enrolled_at=0)]
    coordinator = make_coordinator(store)

    async def go():
        await coordinator.acquire_revision()
        await coordinator.start_journal()
        await reconcile(store, coordinator, tiler, retriever, _settings(), _ENROLL)

        return await coordinator.journal()

    assert asyncio.run(go()) == {"doc-a"}


def test_reconcile_runs_only_in_the_leading_process():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    store._doc_ids = {"dangling"}
    leader = make_coordinator(store)
    follower = make_coordinator(store, valkey=leader.valkey)

    asyncio.run(reconcile(store, leader, tiler, retriever, _settings(), _ENROLL))
    store.deletes.clear()
    store._doc_ids = {"dangling"}

    asyncio.run(reconcile(store, follower, tiler, retriever, _settings(), _ENROLL))

    assert store.deletes == []                                          # the leader's to do


def test_reconcile_leaves_incomplete_for_reenroll():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    tiler.pending_docs = [{
        "doc_id": "doc-b", "doc_type": "PDF", "tile_count": 0, "complete": False,
        "render_version": "rv-test", "source_url": None, "metadata": {"name": "B"},
    }]

    asyncio.run(reconcile(store, make_coordinator(store), tiler, retriever, _settings(), _ENROLL))

    assert store.deletes == ["doc-b"]      # partial cleared
    assert store.upserts == []             # not re-embedded (tiler incomplete)


def test_failed_self_heal_clears_partial_and_stays_pending():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    tiler.pending_docs = [_doc("doc-a", 2, enrolled_at=0)]
    tiler.fail_tiles = {("doc-a", 1)}

    asyncio.run(reconcile(store, make_coordinator(store), tiler, retriever, _settings(), _ENROLL))

    assert store.deletes == ["doc-a", "doc-a"]                          # before, and after the failure
    assert tiler.metadata_patches == []                                 # never promoted


# ---------------------------------------------------------------------------
# flat tile stream
# ---------------------------------------------------------------------------

def test_batches_cross_document_boundaries():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    docs = [_doc("a", 1), _doc("b", 2), _doc("c", 1)]

    failed = asyncio.run(reembed_docs(store, tiler, retriever, docs, batch_size=2, collection="c__v2"))

    assert failed == set()
    batches = sorted([p["doc_id"], p["tile_num"]] for u in store.upserts for p in u["payloads"])
    assert batches == [["a", 0], ["b", 0], ["b", 1], ["c", 0]]
    assert len(store.upserts) == 2                                      # two full batches, not three
    assert all(u["collection"] == "c__v2" for u in store.upserts)


def test_failed_batch_is_retried_and_only_the_bad_doc_fails():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    docs = [_doc("a", 1), _doc("b", 2), _doc("c", 1)]
    tiler.fail_tiles = {("b", 1)}                                       # shares a batch with c

    failed = asyncio.run(reembed_docs(store, tiler, retriever, docs, batch_size=2))

    assert failed == {"b"}
    stored = {p["doc_id"] for u in store.upserts for p in u["payloads"]}
    assert {"a", "c"} <= stored                                         # c recovered on retry


def test_mark_enrolled_promotes_only_finished_docs():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    docs = [_doc("a", 2), _doc("b", 1), _doc("empty", 0)]
    tiler.fail_tiles = {("b", 0)}

    asyncio.run(reembed_docs(store, tiler, retriever, docs, batch_size=1, mark_enrolled=True))

    assert sorted(doc_id for doc_id, _ in tiler.metadata_patches) == ["a", "empty"]


def test_sync_mirrors_tiler_state_for_given_docs():
    store, tiler, retriever = FakeStore(), FakeTiler(), FakeRetriever()
    tiler.docs = [
        _doc("live", 1),
        _doc("pending", 1, enrolled_at=0),                              # rolled-back enroll
        _doc("partial", 1, complete=False),
    ]

    failed = asyncio.run(sync_docs(
        store, tiler, retriever, {"live", "pending", "partial", "gone"},
        batch_size=4, collection="c__v2",
    ))

    assert failed == set()
    assert sorted(store.deletes) == ["gone", "live", "partial", "pending"]  # all dropped first
    assert set(store.delete_collections) == {"c__v2"}
    assert [p["doc_id"] for u in store.upserts for p in u["payloads"]] == ["live"]
