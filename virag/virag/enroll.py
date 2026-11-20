"""Enroll: how a document's points are written into the serving collection.

The Enroller consumes a tile stream of the tiler — `begin`, a `tile` per tile, then
`committed` — and turns it into points:

  1. `begin` names the document. The enroll takes it (`Coordinator.enrolling`): a
     second enroll of the same document is refused with 409 until this one ends, and
     reconcile and a revision leave the document alone meanwhile.
  2. The document is marked pending in the tiler's metadata (`virag_enrolled_at = 0`)
     and its earlier points are dropped. A crash from here leaves a clean, detectable
     state that reconcile finishes or rolls back.
  3. Tiles are embedded and upserted batch by batch, into the collection the serving
     alias resolved to when the document was taken.
  4. `committed` promotes the document (`virag_enrolled_at > 0`).

A stream that fails or ends without `committed` rolls the document back out of Qdrant —
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
import time
import uuid
from collections.abc import Callable
from contextlib import aclosing
from typing import AsyncIterator

from virag.clients.qdrant_store import QdrantStore
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import TilerService
from virag.coordination import Coordinator
from virag.errors import ApiError
from virag.settings import EnrollSettings


log = logging.getLogger("virag.enroll")

_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "virag.pixelrag")


def point_id(doc_id: str, tile_num: int) -> str:
    """Deterministic point id so re-enrolling a document overwrites its own tiles."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{tile_num}"))


class Enroller:
    """Writes one document's points from a tile stream of the tiler."""

    def __init__(
        self,
        store: QdrantStore,
        coordinator: Coordinator,
        tiler: TilerService,
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
        events: AsyncIterator[dict],
        *,
        metadata: Callable[[], dict | None] | None = None,
    ) -> dict:
        """Enroll the document `events` streams and return its summary.

        `metadata` gives the document's metadata; it is asked at `begin` — an upload
        relayed to the tiler has only then gone by in full. The metadata goes to the
        tiler along with the pending mark: the tiler keeps it for everyone (points carry
        none) and leaves the metadata of bytes it already holds as it is, so a re-enroll
        updates it here.

        Raises what the tiler refused the document with (before `begin`),
        EnrollInProgress when another enroll holds it, and whatever broke the write —
        after rolling it back.
        """
        # Closed with the enroll: a stream it gives up on ends in the tiler at once.
        async with aclosing(events):
            begin = await self._begin(events)
            doc_id = begin["doc_id"]

            async with self.coordinator.enrolling(doc_id) as collection:
                given = metadata() if metadata is not None else None
                await self.tiler.patch_metadata(doc_id, {**(given or {}), "virag_enrolled_at": 0})

                try:
                    await self.store.delete_by_doc(doc_id, collection=collection)

                    enrolled, committed = await self._write(begin, events, collection)

                    # Promote: the enroll is durable end-to-end — clear pending.
                    await self.tiler.patch_metadata(doc_id, {"virag_enrolled_at": int(time.time())})

                except Exception:
                    await self._rollback(doc_id, collection)
                    raise

        log.info("enrolled: doc_id=%s points=%d", doc_id, enrolled)

        return {
            "doc_id": doc_id,
            "doc_type": begin["doc_type"],
            "render_version": begin["render_version"],
            "tile_count": committed.get("tile_count", enrolled),
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


    async def _begin(self, events: AsyncIterator[dict]) -> dict:
        """The stream's first event, which names the document."""
        try:
            event = await anext(events)

        except StopAsyncIteration:
            raise ApiError(status_code=502, message="tiler stream ended before it began") from None

        if event.get("event") == "error":
            raise self._stream_error(event)

        if event.get("event") != "begin":
            raise ApiError(status_code=502, message=f"tiler stream began with {event.get('event')!r}")

        return event


    async def _write(self, begin: dict, events: AsyncIterator[dict], collection: str) -> tuple[int, dict]:
        """Embed and upsert the stream's tiles batch by batch, up to its `committed`
        event. Returns how many points were written, and that event."""
        batch: list[dict] = []
        enrolled = 0

        async for event in events:
            kind = event.get("event")

            if kind == "tile":
                batch.append(event)

                if len(batch) >= self.batch_size:
                    enrolled += await self._flush(begin, batch, collection)
                    batch = []

            elif kind == "committed":
                enrolled += await self._flush(begin, batch, collection)

                return enrolled, event

            elif kind == "error":
                raise self._stream_error(event)

        raise ApiError(status_code=502, message="tiler stream ended without a committed event")


    async def _flush(self, begin: dict, batch: list[dict], collection: str) -> int:
        """Embed one batch of tiles and upsert it. The tiler knows the doc since `begin`
        (Qdrant ⊆ Tiler) and serves its tiles once the ingest commits."""
        if not batch:
            return 0

        doc_id = begin["doc_id"]
        vectors = await self.retriever.embed_images([tile["data"] for tile in batch], bulk=True)

        payloads = [
            {
                "doc_id": doc_id,
                "tile_num": tile["tile_num"],
                "image_hash": tile["image_hash"],
                "render_version": begin["render_version"],
            }
            for tile in batch
        ]
        ids = [point_id(doc_id, tile["tile_num"]) for tile in batch]

        await self.store.upsert(vectors, payloads, ids=ids, collection=collection)

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


    @staticmethod
    def _stream_error(event: dict) -> ApiError:
        """An in-stream `error` event of the tiler as the error the enroll ends with."""
        return ApiError(
            status_code=502,
            message=f"tiler stream error: {event.get('message')}",
            details={"tiler_code": event.get("code")},
        )
