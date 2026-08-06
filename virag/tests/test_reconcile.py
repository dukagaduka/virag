"""reconcile: Qdrant ⊆ Tiler enforcement and self-heal of stale pending enrolls."""

from __future__ import annotations

import asyncio

from virag.api.enroll import point_id
from virag.reconcile import reconcile
from virag.settings import ReconcileSettings

from conftest import FakeEmbeddings, FakeStore, FakeTiler


def _settings(self_heal=True):
    return ReconcileSettings(self_heal=self_heal, interval_sec=0, pending_ttl_sec=3600)


def test_reconcile_deletes_dangling_points():
    store, tiler, emb = FakeStore(), FakeTiler(), FakeEmbeddings()
    store._doc_ids = {"kept", "dangling"}                       # dangling has no tiler doc
    tiler.docs = [{"doc_id": "kept"}]

    asyncio.run(reconcile(store, tiler, emb, _settings()))

    assert "dangling" in store.deletes
    assert "kept" not in store.deletes


def test_reconcile_self_heals_complete_pending():
    store, tiler, emb = FakeStore(), FakeTiler(), FakeEmbeddings()
    tiler.pending_docs = [{
        "doc_id": "doc-a", "doc_type": "PDF", "tile_count": 2, "complete": True,
        "render_version": "rv-test", "source_url": None,
        "metadata": {"name": "A", "virag_started_at": 1, "virag_enrolled_at": 0},
    }]

    asyncio.run(reconcile(store, tiler, emb, _settings()))

    assert len(store.upserts) == 1
    up = store.upserts[0]
    assert up["ids"] == [point_id("doc-a", 0), point_id("doc-a", 1)]     # matches enroll's ids
    assert up["payloads"][0]["metadata"] == {"name": "A"}               # virag_* markers stripped
    # pending is promoted after re-embed
    assert ("doc-a", {"virag_enrolled_at": tiler.metadata_patches[-1][1]["virag_enrolled_at"]}) \
        == tiler.metadata_patches[-1]
    assert tiler.metadata_patches[-1][1]["virag_enrolled_at"] > 0


def test_reconcile_leaves_incomplete_for_reenroll():
    store, tiler, emb = FakeStore(), FakeTiler(), FakeEmbeddings()
    tiler.pending_docs = [{
        "doc_id": "doc-b", "doc_type": "PDF", "tile_count": 0, "complete": False,
        "render_version": "rv-test", "source_url": None, "metadata": {"name": "B"},
    }]

    asyncio.run(reconcile(store, tiler, emb, _settings()))

    assert store.deletes == ["doc-b"]      # partial cleared
    assert store.upserts == []             # not re-embedded (tiler incomplete)
