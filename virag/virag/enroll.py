"""Enroll: how a document's points are written into the serving collection.

The Enroller consumes a tile stream of the tiler — `Begin`, a `Tile` per tile, then
`Committed` — and turns it into points:

  1. `Begin` names the document. The enroll takes it (`Coordinator.enrolling`): a
     second enroll of the same document is refused with 409 until this one ends, and
     reconcile and a revision leave the document alone meanwhile.
  2. The document is marked pending in the tiler (`TilerClient.mark_pending`) and its
     earlier points are dropped. A crash from here leaves a clean, detectable
     state that reconcile finishes or rolls back.
  3. Tiles are embedded and upserted batch by batch, into the collection the serving
     alias resolved to when the document was taken.
  4. `Committed` promotes the document (`TilerClient.mark_enrolled`).

A document whose ingest fails right after `Begin` may be gone from the tiler before it
is marked pending; the enroll then ends with what the stream failed with, not with the
mark that found no document.

A stream that fails or ends without `Committed` rolls the document back out of Qdrant —
only while the document is still this process's: a process that stalled past its lease
may find another enroll writing it already. A doc the tiler held before stays there
pending, and reconcile enrolls it again from its stored tiles; a new one the tiler
drops itself when its ingest ends without a commit.

The Enroller is the one writer of a document's points in the serving collection:
`/enroll` feeds it the tiler's ingest stream, reconcile's self-heal and a revision's
sync the stream of a stored doc (`enroll_stored`), and both drop a document's points
through it (`drop`).

It also writes a revision's build (`reembed`) — a collection nobody else writes and
nobody reads yet, so no document is taken or marked: every tile of every stored doc is
fetched from the tiler and embedded in one flat stream across documents, into the same
points an enroll writes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from collections.abc import Callable
from contextlib import aclosing
from typing import AsyncIterator

from virag.clients.qdrant_store import QdrantStore, TilePoint
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import Begin, Committed, Event, Tile, TilerClient, TilerDoc
from virag.coordination import Coordinator
from virag.errors import ApiError
from virag.settings import EnrollSettings
from virag.utils import chunked, gather


log = logging.getLogger("virag.enroll")


class Enroller:
    """Writes points from the tiler's tiles: one document's from its tile stream, or a
    revision's whole build."""

    # Tiles assumed in flight when the tiler has no bulk concurrency cap; sizes the
    # window of a re-embed.
    UNCAPPED_TILES = 128


    def __init__(
        self,
        store: QdrantStore,
        coordinator: Coordinator,
        tiler: TilerClient,
        retriever: RetrieverClient,
        enroll_settings: EnrollSettings,
    ):

        self.store       = store
        self.coordinator = coordinator
        self.tiler       = tiler
        self.retriever   = retriever

        self.batch_size = enroll_settings.batch_size


    async def enroll(
        self,
        events: AsyncIterator[Event],
        *,
        metadata: Callable[[], dict | None] | None = None,
    ) -> dict:
        """Enroll the document `events` streams and return its summary.

        `metadata` gives the document's metadata; it is asked at `Begin` — an upload
        relayed to the tiler has only then gone by in full. The metadata goes to the
        tiler along with the pending mark: the tiler keeps it for everyone (points carry
        none) and leaves the metadata of bytes it already holds as it is, so a re-enroll
        updates it here.

        Raises what the tiler refused the document with (before `Begin`) or failed its
        stream with, EnrollInProgress when another enroll holds it, and whatever broke
        the write — after rolling it back.
        """
        # Closed with the enroll: a stream it gives up on ends in the tiler at once.
        async with aclosing(events):
            begin = await self._begin(events)
            doc_id = begin.doc_id

            async with self.coordinator.enrolling(doc_id) as collection:
                try:
                    await self.tiler.mark_pending(doc_id, metadata() if metadata is not None else None)

                except ApiError as e:
                    # The tiler has dropped the doc already: its ingest failed right after
                    # `Begin`. The stream ends with the reason — that is what to answer with.
                    if self.tiler.unknown_doc(e):
                        await self._end(events)

                    raise

                try:
                    await self.store.delete_by_doc(doc_id, collection=collection)

                    enrolled, committed = await self._write(begin, events, collection)

                    # Promote: the enroll is durable end-to-end — clear pending.
                    await self.tiler.mark_enrolled(doc_id)

                except Exception:
                    await self._rollback(doc_id, collection)
                    raise

        log.info("enrolled: doc_id=%s points=%d", doc_id, enrolled)

        return {
            "doc_id": doc_id,
            "doc_type": begin.doc_type,
            "render_version": begin.render_version,
            "tile_count": committed.tile_count,
            "enrolled": enrolled,
        }


    async def enroll_stored(self, doc_id: str) -> dict:
        """Enroll a document the tiler holds again, from its stored tiles."""
        return await self.enroll(self.tiler.doc_stream(doc_id))


    async def drop(self, doc_id: str) -> None:
        """Drop a document's points — held like an enroll, so never under one
        (EnrollInProgress)."""
        async with self.coordinator.enrolling(doc_id) as collection:
            await self.store.delete_by_doc(doc_id, collection=collection)


    async def reembed(self, docs: list[TilerDoc], collection: str) -> set[str]:
        """Re-embed complete tiler docs into `collection`, a build nobody else writes;
        docs that fail are retried once in per-document batches, so a doc that merely
        shared a batch with a broken one recovers. Returns the doc_ids that still failed."""
        failed = await self._reembed_pass(docs, collection)

        if not failed:
            return failed

        log.warning("reembed: retrying %d failed doc(s)", len(failed))

        return await self._reembed_pass([doc for doc in docs if doc.doc_id in failed], collection, per_doc=True)


    # ------------------------------------------------------------------ #
    # one document's stream
    # ------------------------------------------------------------------ #
    async def _begin(self, events: AsyncIterator[Event]) -> Begin:
        """The stream's first event, which names the document."""
        try:
            event = await anext(events)

        except StopAsyncIteration:
            raise ApiError(status_code=502, message="tiler stream ended before it began") from None

        if not isinstance(event, Begin):
            raise ApiError(status_code=502, message=f"tiler stream began with {type(event).__name__}")

        return event


    @staticmethod
    async def _end(events: AsyncIterator[Event]) -> None:
        """Read a stream the tiler has given up on to its end, raising what it ends with."""
        async for _ in events:
            pass


    async def _write(self, begin: Begin, events: AsyncIterator[Event], collection: str) -> tuple[int, Committed]:
        """Embed and upsert the stream's tiles batch by batch, up to its `Committed`
        event. Returns how many points were written, and that event."""
        batch: list[Tile] = []
        enrolled = 0

        async for event in events:
            if isinstance(event, Tile):
                batch.append(event)

                if len(batch) >= self.batch_size:
                    enrolled += await self._flush(begin, batch, collection)
                    batch = []

            elif isinstance(event, Committed):
                enrolled += await self._flush(begin, batch, collection)

                return enrolled, event

        raise ApiError(status_code=502, message="tiler stream ended without a committed event")


    async def _flush(self, begin: Begin, batch: list[Tile], collection: str) -> int:
        """Embed one batch of tiles and upsert it. The tiler knows the doc since `Begin`
        (Qdrant ⊆ Tiler) and serves its tiles once the ingest commits."""
        if not batch:
            return 0

        vectors = await self.retriever.embed_images([tile.data for tile in batch], bulk=True)
        points = [TilePoint(begin.doc_id, tile.tile_num, tile.image_hash, begin.render_version) for tile in batch]

        await self.store.upsert(vectors, points, collection=collection)

        return len(batch)


    async def _rollback(self, doc_id: str, collection: str) -> None:
        """Drop the document's points after a failed write — unless the document is no
        longer this process's. Never masks the failure being handled."""
        try:
            if not await self.coordinator.holds_enroll(doc_id):
                log.error("rollback skipped: doc_id=%s is no longer this process's to write", doc_id)

                return

            await self.store.delete_by_doc(doc_id, collection=collection)

        except Exception as e:  # noqa: BLE001
            log.error("rollback failed for doc_id=%s: %r", doc_id, e)


    # ------------------------------------------------------------------ #
    # the flat re-embed
    # ------------------------------------------------------------------ #
    async def _reembed_pass(self, docs: list[TilerDoc], collection: str, per_doc: bool = False) -> set[str]:
        """One pass of the flat tile stream.

        Every (doc_id, tile_num) of every doc is cut into batches of `batch_size` across
        document boundaries, so small docs share full batches. Each batch fetches its
        tiles concurrently (bulk, capped by the tiler's `max_concurrency`), embeds them in
        one bulk retriever call and upserts once. A window of batches stays in flight, so
        one batch fetches while another embeds; it is derived from the tiler cap — more
        would only queue at the guard. The points are the ones an enroll writes (`TilePoint`),
        ids included.

        A failed batch fails every doc it touched (their later tiles are skipped).
        `per_doc` keeps batches within one document (the retry pass).
        """
        by_id = {doc.doc_id: doc for doc in docs}
        failed: set[str] = set()

        async def process(batch: list[tuple[str, int]]) -> None:
            batch = [(doc_id, n) for doc_id, n in batch if doc_id not in failed]
            if not batch:
                return

            try:
                jpegs = await gather(self.tiler.get_tile(doc_id, n, bulk=True) for doc_id, n in batch)
                b64s = [base64.b64encode(j).decode("ascii") for j in jpegs]
                vectors = await self.retriever.embed_images(b64s, bulk=True)

                points = [
                    TilePoint(doc_id, n, hashlib.sha256(j).hexdigest(), by_id[doc_id].render_version)
                    for (doc_id, n), j in zip(batch, jpegs)
                ]

                await self.store.upsert(vectors, points, collection=collection)

            except Exception as e:  # noqa: BLE001 - a failed batch fails its docs, not the stream
                touched = {doc_id for doc_id, _ in batch}
                failed.update(touched)
                log.error("reembed: batch failed for doc(s) %s: %r", sorted(touched), e)

        if per_doc:
            batches = (
                batch
                for doc_id, doc in by_id.items()
                for batch in chunked(((doc_id, n) for n in range(doc.tile_count)), self.batch_size)
            )
        else:
            tiles = ((doc_id, n) for doc_id, doc in by_id.items() for n in range(doc.tile_count))
            batches = chunked(tiles, self.batch_size)

        window = max(1, (self.tiler.max_concurrency or self.UNCAPPED_TILES) // self.batch_size) + 1
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
