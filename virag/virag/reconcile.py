"""Enroll self-repair: keep Qdrant a consistent subset of the tiler.

Runs at startup and (optionally) on an interval. Two passes:

  1. Enforce the `Qdrant ⊆ Tiler` invariant — delete any Qdrant points whose doc_id
     the tiler no longer has (a dangling hit would 404 on tile fetch).
  2. Clear stale pending enrolls — a doc marked pending (`virag_enrolled_at == 0`)
     older than the TTL is a crashed/partial enroll; drop its partial points.

The TTL gate (`virag_started_at < now - pending_ttl_sec`) skips in-flight enrolls,
whose `virag_started_at` is ~now. Because the tiler is bound to this virag by secret,
every tiler doc carries virag metadata, so there are no foreign docs to misjudge.

Also home of the flat tile-stream re-embed (`reembed_docs`, `sync_docs`) shared by
self-heal, revision and rollback.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import AsyncExitStack
from itertools import islice

from virag.api.enroll import point_id
from virag.clients.retriever_client import RetrieverClient
from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService
from virag.settings import EnrollSettings, ReconcileSettings
from virag.utils import gather


log = logging.getLogger("virag.reconcile")

# Tiles assumed in flight when the tiler has no bulk concurrency cap; sizes the window.
_UNCAPPED_TILES = 128


async def reconcile(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    settings: ReconcileSettings,
    enroll_settings: EnrollSettings,
) -> None:
    """Run one reconcile pass: enforce the invariant, then clear stale pending enrolls."""
    await enforce_subset(store, tiler)
    await _clear_stale_pending(store, tiler, retriever, settings, enroll_settings)


def embeddable(doc: dict) -> bool:
    """A tiler doc whose tiles can be fetched: complete and not quarantined by a tiler
    revision (its tiles answer 410 until it is re-enrolled)."""
    return bool(doc.get("complete")) and not doc.get("quarantined")


async def enforce_subset(
    store: QdrantStore,
    tiler: TilerService,
    *,
    collection: str | None = None,
) -> int:
    """Delete points whose doc_id is absent from the tiler (invariant repair) or
    quarantined by it, in the serving collection or in `collection` (a revision build).
    Returns how many docs."""
    qdrant_ids = await store.iter_doc_ids(collection=collection)
    docs = await tiler.iter_docs()

    dangling = qdrant_ids - {doc["doc_id"] for doc in docs}

    # A quarantined doc's tiles answer 410, so its points only crowd live hits out of
    # top_n. One being written right now is a re-enroll lifting the quarantine.
    quarantined = {doc["doc_id"] for doc in docs if doc.get("quarantined")} & qdrant_ids
    quarantined -= store.writing()

    if dangling:
        log.warning("reconcile: %d dangling doc_id(s) in qdrant not in tiler; deleting", len(dangling))

    if quarantined:
        log.warning("reconcile: %d doc_id(s) quarantined by the tiler; deleting their points", len(quarantined))

    stale = dangling | quarantined
    if not stale:
        return 0

    await store.delete_by_docs(sorted(stale), collection=collection)

    return len(stale)


async def _clear_stale_pending(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    settings: ReconcileSettings,
    enroll_settings: EnrollSettings,
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

    # Registered as writers so a concurrent revision journals and later syncs these docs.
    async with AsyncExitStack() as writing:
        for doc in stale:
            await writing.enter_async_context(store.writer(doc["doc_id"]))

        await store.delete_by_docs([doc["doc_id"] for doc in stale])      # always clear the partial first

        # An incomplete or quarantined tiler doc can't be re-embedded (no servable tiles);
        # leave it for a re-enroll to restore.
        healable = [doc for doc in stale if settings.self_heal and embeddable(doc)]

        for doc in stale:
            if doc not in healable:
                log.info("reconcile: doc_id=%s left for re-enroll (self-heal off, or tiler doc incomplete or quarantined)", doc["doc_id"])

        if not healable:
            return

        failed = await reembed_docs(
            store,
            tiler,
            retriever,
            healable,
            batch_size=enroll_settings.batch_size,
            mark_enrolled=True,
        )

        if failed:
            # Leave no partial points behind; the docs stay pending for the next sweep.
            await store.delete_by_docs(sorted(failed))
            log.error("reconcile: self-heal failed for %d doc(s): %s", len(failed), sorted(failed))

        log.info("reconcile: self-healed %d doc(s)", len(healable) - len(failed))


async def sync_docs(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    doc_ids: Iterable[str],
    *,
    batch_size: int,
    collection: str | None = None,
) -> set[str]:
    """Make `collection` mirror the tiler for these docs: drop their points, then
    re-embed the ones the tiler holds complete and enrolled. Docs gone, incomplete or
    still pending stay dropped — exactly as the serving collection treats them.
    Returns the doc_ids that failed to re-embed."""
    wanted = set(doc_ids)
    if not wanted:
        return set()

    await store.delete_by_docs(sorted(wanted), collection=collection)

    current = [
        doc
        for doc in await tiler.iter_docs()
        if doc["doc_id"] in wanted
        and embeddable(doc)
        and (doc.get("metadata") or {}).get("virag_enrolled_at", 0) > 0
    ]

    return await reembed_docs(
        store,
        tiler,
        retriever,
        current,
        batch_size=batch_size,
        collection=collection,
    )


async def reembed_docs(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    docs: list[dict],
    *,
    batch_size: int,
    collection: str | None = None,
    mark_enrolled: bool = False,
) -> set[str]:
    """Re-embed complete tiler docs into Qdrant (self-heal, revision, sync); docs that
    fail are retried once in per-document batches, so a doc that merely shared a batch
    with a broken one recovers. Returns the doc_ids that still failed."""
    failed = await _reembed_pass(
        store, tiler, retriever, docs,
        batch_size=batch_size, collection=collection, mark_enrolled=mark_enrolled,
    )

    if not failed:
        return failed

    log.warning("reembed: retrying %d failed doc(s)", len(failed))

    return await _reembed_pass(
        store, tiler, retriever, [doc for doc in docs if doc["doc_id"] in failed],
        batch_size=batch_size, collection=collection, mark_enrolled=mark_enrolled,
        per_doc=True,
    )


async def _reembed_pass(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    docs: list[dict],
    *,
    batch_size: int,
    collection: str | None,
    mark_enrolled: bool,
    per_doc: bool = False,
) -> set[str]:
    """One pass of the flat tile stream.

    Every (doc_id, tile_num) of every doc is cut into batches of `batch_size` across
    document boundaries, so small docs share full batches. Each batch fetches its
    tiles concurrently (bulk, capped by the tiler's `max_concurrency`), embeds them in
    one bulk retriever call and upserts once. A window of batches stays in flight, so
    one batch fetches while another embeds; it is derived from the tiler cap — more
    would only queue at the guard. Point ids match enroll's (`point_id`), so this
    restores exactly what a fresh enroll would.

    A doc is done when its last tile lands; a failed batch fails every doc it touched
    (their later tiles are skipped). `per_doc` keeps batches within one document (the
    retry pass). `collection` targets a revision build; `mark_enrolled` promotes
    self-healed docs out of pending.
    """
    by_id = {doc["doc_id"]: doc for doc in docs}
    remaining = {doc_id: doc["tile_count"] for doc_id, doc in by_id.items()}
    failed: set[str] = set()

    # Strip virag's own lifecycle markers — the payload carries user metadata only.
    payload_meta = {
        doc_id: {k: v for k, v in (doc.get("metadata") or {}).items() if not k.startswith("virag_")}
        for doc_id, doc in by_id.items()
    }

    async def finish(doc_ids: list[str]) -> None:
        if not mark_enrolled:
            return

        for doc_id in doc_ids:
            try:
                await tiler.patch_metadata(doc_id, {"virag_enrolled_at": int(time.time())})

            except Exception as e:  # noqa: BLE001 - one doc must not stop the stream
                log.error("reembed: cannot promote doc_id=%s: %r", doc_id, e)
                failed.add(doc_id)

    async def process(batch: list[tuple[str, int]]) -> None:
        batch = [(doc_id, n) for doc_id, n in batch if doc_id not in failed]
        if not batch:
            return

        try:
            jpegs = await gather(tiler.get_tile(doc_id, n, bulk=True) for doc_id, n in batch)
            b64s = [base64.b64encode(j).decode("ascii") for j in jpegs]
            vectors = await retriever.embed_images(b64s, bulk=True)

            payloads = [
                {
                    "doc_id": doc_id,
                    "tile_num": n,
                    "image_hash": hashlib.sha256(j).hexdigest(),
                    "doc_type": by_id[doc_id].get("doc_type"),
                    "source_url": by_id[doc_id].get("source_url"),
                    "render_version": by_id[doc_id].get("render_version") or "",
                    "metadata": payload_meta[doc_id],
                }
                for (doc_id, n), j in zip(batch, jpegs)
            ]
            ids = [point_id(doc_id, n) for doc_id, n in batch]

            await store.upsert(vectors, payloads, ids=ids, collection=collection)

        except Exception as e:  # noqa: BLE001 - a failed batch fails its docs, not the stream
            touched = {doc_id for doc_id, _ in batch}
            failed.update(touched)
            log.error("reembed: batch failed for doc(s) %s: %r", sorted(touched), e)

            return

        finished = []
        for doc_id, _ in batch:
            remaining[doc_id] -= 1
            if remaining[doc_id] == 0:
                finished.append(doc_id)

        await finish([doc_id for doc_id in finished if doc_id not in failed])

    # A doc without tiles has nothing to stream — it is done as it stands.
    await finish([doc_id for doc_id, count in remaining.items() if count == 0])

    if per_doc:
        batches = (
            batch
            for doc_id, doc in by_id.items()
            for batch in _batches(((doc_id, n) for n in range(doc["tile_count"])), batch_size)
        )
    else:
        tiles = ((doc_id, n) for doc_id, doc in by_id.items() for n in range(doc["tile_count"]))
        batches = _batches(tiles, batch_size)

    window = max(1, (tiler.max_concurrency or _UNCAPPED_TILES) // batch_size) + 1
    slots = asyncio.Semaphore(window)

    async def run(batch: list[tuple[str, int]]) -> None:
        try:
            await process(batch)

        finally:
            slots.release()

    async with asyncio.TaskGroup() as group:
        for batch in batches:
            await slots.acquire()
            group.create_task(run(batch))

    return failed


def _batches(items: Iterator, size: int) -> Iterator[list]:
    """Lazily cut an iterator into lists of `size` (the last may be shorter)."""
    while batch := list(islice(items, size)):
        yield batch


async def run_periodic(
    store: QdrantStore,
    tiler: TilerService,
    retriever: RetrieverClient,
    settings: ReconcileSettings,
    enroll_settings: EnrollSettings,
) -> None:
    """Loop reconcile on the configured interval until cancelled (interval 0 = disabled)."""
    if settings.interval_sec <= 0:
        return

    while True:
        await asyncio.sleep(settings.interval_sec)

        try:
            await reconcile(store, tiler, retriever, settings, enroll_settings)

        except asyncio.CancelledError:
            raise

        except Exception as e:  # noqa: BLE001 - a periodic sweep must never crash the loop
            log.error("periodic reconcile failed: %r", e)
