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
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import aclosing
from typing import AsyncIterator

from virag.clients.qdrant_store import QdrantStore, TilePoint
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import Begin, Committed, Event, Tile, TilerClient
from virag.coordination import Coordinator
from virag.errors import ApiError
from virag.settings import EnrollSettings


log = logging.getLogger("virag.enroll")


class Enroller:
    """Writes one document's points from a tile stream of the tiler."""

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
