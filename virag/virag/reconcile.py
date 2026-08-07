"""Enroll self-repair: keep Qdrant a consistent subset of the tiler.

Runs at startup and (optionally) on an interval. Two passes:

  1. Enforce the `Qdrant ⊆ Tiler` invariant — delete any Qdrant points whose doc_id
     the tiler no longer has (a dangling hit would 404 on tile fetch).
  2. Clear stale pending enrolls — a doc marked pending (`virag_enrolled_at == 0`)
     older than the TTL is a crashed/partial enroll; drop its partial points.

The TTL gate (`virag_started_at < now - pending_ttl_sec`) skips in-flight enrolls,
whose `virag_started_at` is ~now. Because the tiler is bound to this virag by secret,
every tiler doc carries virag metadata, so there are no foreign docs to misjudge.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time

from virag.api.enroll import point_id
from virag.clients.embeddings_client import EmbeddingsClient
from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService
from virag.settings import ReconcileSettings


log = logging.getLogger("virag.reconcile")


async def reconcile(
    store: QdrantStore,
    tiler: TilerService,
    embeddings: EmbeddingsClient,
    settings: ReconcileSettings,
) -> None:
    """Run one reconcile pass: enforce the invariant, then clear stale pending enrolls."""
    await _enforce_subset(store, tiler)
    await _clear_stale_pending(store, tiler, embeddings, settings)


async def _enforce_subset(store: QdrantStore, tiler: TilerService) -> None:
    """Delete Qdrant points whose doc_id is absent from the tiler (invariant repair)."""
    qdrant_ids = await store.iter_doc_ids()
    tiler_ids = {doc["doc_id"] for doc in await tiler.iter_docs()}

    dangling = qdrant_ids - tiler_ids
    if not dangling:
        return

    log.warning("reconcile: %d dangling doc_id(s) in qdrant not in tiler; deleting", len(dangling))

    for doc_id in dangling:
        await store.delete_by_doc(doc_id)


async def _clear_stale_pending(
    store: QdrantStore,
    tiler: TilerService,
    embeddings: EmbeddingsClient,
    settings: ReconcileSettings,
) -> None:
    """Drop partial points for enrolls left pending past the TTL, then self-heal."""
    cutoff = int(time.time()) - settings.pending_ttl_sec

    stale = await tiler.iter_docs(
        filters=[
            {"key": "virag_enrolled_at", "op": "eq", "value": 0},
            {"key": "virag_started_at", "op": "lt", "value": cutoff},
        ]
    )
    if not stale:
        return

    log.warning("reconcile: %d stale pending enroll(s); clearing partial points", len(stale))

    for doc in stale:
        doc_id = doc["doc_id"]
        await store.delete_by_doc(doc_id)                       # always clear the partial first

        if not (settings.self_heal and doc.get("complete")):
            # An incomplete tiler doc can't be re-embedded (no tiles yet); leave it for
            # tiler_revision / re-enroll to restore.
            log.info("reconcile: doc_id=%s left for re-enroll (self-heal off or tiler incomplete)", doc_id)
            continue

        try:
            await _reembed_doc(store, tiler, embeddings, doc)
            log.info("reconcile: self-healed doc_id=%s", doc_id)

        except Exception as e:  # noqa: BLE001 - one bad doc must not stop the sweep
            log.error("reconcile: self-heal failed for doc_id=%s: %r", doc_id, e)


async def _reembed_doc(
    store: QdrantStore,
    tiler: TilerService,
    embeddings: EmbeddingsClient,
    summary: dict,
    *,
    collection: str | None = None,
    mark_enrolled: bool = True,
) -> None:
    """Re-embed a complete tiler doc into Qdrant (self-heal, or a revision target).

    Tiles are pulled via get_tile and embedded in batches; the point ids match enroll's
    (`point_id(doc_id, tile_num)`), so this restores exactly what a fresh enroll would.
    `collection` targets a specific physical collection (a revision build); `mark_enrolled`
    clears the pending marker (self-heal) — a revision leaves it untouched.
    """
    doc_id = summary["doc_id"]
    tile_count = summary["tile_count"]
    render_version = summary.get("render_version") or ""
    # Strip virag's own lifecycle markers — the payload carries user metadata only.
    payload_meta = {k: v for k, v in (summary.get("metadata") or {}).items() if not k.startswith("virag_")}

    batch: list[int] = []

    async def flush() -> None:
        if not batch:
            return

        jpegs = [await tiler.get_tile(doc_id, n) for n in batch]
        b64s = [base64.b64encode(j).decode("ascii") for j in jpegs]
        vectors = await embeddings.embed_images(b64s)
        payloads = [
            {
                "doc_id": doc_id,
                "tile_num": n,
                "image_hash": hashlib.sha256(j).hexdigest(),
                "doc_type": summary.get("doc_type"),
                "source_url": summary.get("source_url"),
                "render_version": render_version,
                "metadata": payload_meta,
            }
            for n, j in zip(batch, jpegs)
        ]
        ids = [point_id(doc_id, n) for n in batch]

        await store.upsert(vectors, payloads, ids=ids, collection=collection)
        batch.clear()

    for tile_num in range(tile_count):
        batch.append(tile_num)
        if len(batch) >= embeddings.chunk_size:
            await flush()

    await flush()

    if mark_enrolled:
        await tiler.patch_metadata(doc_id, {"virag_enrolled_at": int(time.time())})


async def revise(store: QdrantStore, tiler: TilerService, embeddings: EmbeddingsClient) -> dict:
    """Re-embed the whole corpus into a fresh collection, then blue-green swap the alias.

    The live collection keeps serving until the new one is fully built and the alias flips
    atomically — no empty window. Use after an embedding-model change (or a tiler /revision).
    """
    new_collection = await store.create_revision_collection()
    log.info("revision: building %s", new_collection)

    reembedded = 0
    skipped = 0

    for doc in await tiler.iter_docs():
        if not doc.get("complete"):
            skipped += 1  # incomplete docs aren't servable; leave them for tiler /revision
            continue

        await _reembed_doc(store, tiler, embeddings, doc, collection=new_collection, mark_enrolled=False)
        reembedded += 1

    await store.swap_alias(new_collection)

    log.info("revision: re-embedded=%d skipped=%d, alias now → %s", reembedded, skipped, new_collection)
    return {"reembedded": reembedded, "skipped": skipped, "collection": new_collection}


async def run_periodic(
    store: QdrantStore,
    tiler: TilerService,
    embeddings: EmbeddingsClient,
    settings: ReconcileSettings,
) -> None:
    """Loop reconcile on the configured interval until cancelled (interval 0 = disabled)."""
    if settings.interval_sec <= 0:
        return

    while True:
        await asyncio.sleep(settings.interval_sec)

        try:
            await reconcile(store, tiler, embeddings, settings)

        except asyncio.CancelledError:
            raise

        except Exception as e:  # noqa: BLE001 - a periodic sweep must never crash the loop
            log.error("periodic reconcile failed: %r", e)
