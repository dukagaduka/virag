"""Ingest: a document's way into the store, and out of it.

An ingest is a stream of events (`events.py`) — `Begin`, a `Tile` per tile, then
`Committed`:

  1. The upload waits in `.incoming` while it renders (a URL is captured into one
     first); its sha256 is the doc_id.
  2. Bytes the store already holds and serves are not rendered again: the stored doc is
     replayed (`Tiles.replay`).
  3. A new doc gets its row first, incomplete, so the consumer can refer to it from
     `begin`; the row's primary key settles who ingests the doc when two requests bring
     the same bytes (the other gets 409 INGEST_IN_PROGRESS).
  4. The tiles are rendered and kept in memory; the blob and the tile rows reach the
     store together, in one commit. Nothing of a stored doc changes before it.

Whatever ends an ingest without that commit — a failure, a client that went away —
takes the spooled upload and the row it inserted along.
"""

from __future__ import annotations

import logging
from contextlib import aclosing
from typing import Any, AsyncIterator

import anyio

from . import events, media, metrics
from .capture import UrlCapture
from .config import Settings
from .errors import ApiError, bad_request, conflict
from .meta.schema import MetadataSchema
from .models import DocType
from .render.renderer import Renderer
from .store.blobs import BlobStore, Incoming
from .store.meta import DocRow, MetaStore, TileRow
from .tiles import Tiles

log = logging.getLogger("tiler.ingest")


class Ingester:
    """Brings documents into the store, and takes them out of it."""


    def __init__(
        self,
        store: MetaStore,
        blobs: BlobStore,
        renderer: Renderer,
        capture: UrlCapture,
        tiles: Tiles,
        schema: MetadataSchema,
        settings: Settings,
    ):

        self.store    = store
        self.blobs    = blobs
        self.renderer = renderer
        self.capture  = capture
        self.tiles    = tiles
        self.schema   = schema

        self.render_version = settings.render_version


    async def ingest(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
    ) -> events.Committed:
        """Ingest a document into the store and return its summary — the event its
        stream ends with.

        Runs the same pipeline as `ingest_stream`, only without emitting the tiles: bytes
        the store already serves return at once instead of being replayed.
        """
        committed = None

        async for event in self._ingest(
            doc_type=doc_type, upload=upload, url=url, metadata=metadata, replay=False
        ):
            if isinstance(event, events.Committed):
                committed = event

        if committed is None:  # pragma: no cover
            raise RuntimeError("ingest finished without a committed event")

        return committed


    def ingest_stream(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[events.Event]:
        """Ingest a document as a stream of events: `Begin`, a `Tile` per tile, then
        `Committed`. Failures raise."""
        return self._ingest(doc_type=doc_type, upload=upload, url=url, metadata=metadata, replay=True)


    async def delete(self, doc_id: str) -> None:
        """Delete a document's rows, blob and cached tiles (404 for a doc the store
        does not have). A doc still being ingested is its ingest's to remove: 409."""
        row = await self.store.require_doc(doc_id)

        if not row.complete:
            raise self._in_progress(doc_id)

        if await self.store.delete_doc(doc_id):
            await self.blobs.delete(doc_id)
            await self.tiles.forget(doc_id)
            metrics.DOCS.set(await self.store.count_docs())


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    async def _ingest(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        replay: bool,
    ) -> AsyncIterator[events.Event]:
        """The ingest pipeline: `Begin`, a `Tile` per tile, `Committed`.

        The upload (taken over here) waits in `.incoming` while it renders. Bytes the
        store already holds and serves are not rendered again: their tiles are replayed
        when `replay` is set, and `Committed` follows at once otherwise. Whatever ends
        the ingest, the spooled upload is gone with it. Failures raise.
        """
        source_url: str | None = None

        try:
            if doc_type == DocType.URL:
                if not url:
                    raise bad_request("url is required when doc_type=URL", "MISSING_URL")

                upload = await self.blobs.spool(await self.capture.capture(url))
                effective_type = DocType.MHTML
                source_url = url
                mime = "multipart/related"

            else:
                if upload is None:
                    raise bad_request(f"document is required when doc_type={doc_type}", "MISSING_FILE")

                effective_type = doc_type
                mime = media.check_and_mime(doc_type, await self.blobs.head(upload))

            existing = await self.store.get_doc(upload.doc_id)

            # Same hash, same bytes: a doc that is stored and servable is not rendered again.
            if existing is not None and existing.servable(self.render_version):
                metrics.INGESTS.labels(doc_type=effective_type, result="idempotent").inc()
                stream = self.tiles.replay(existing, render=replay)

            else:
                stream = self._render_into_store(upload, effective_type, source_url, mime, metadata, existing)

            try:
                async with aclosing(stream):
                    async for event in stream:
                        yield event

            except ApiError as exc:
                log.warning("ingest failed for %s: %s", upload.doc_id, exc.message)
                raise

        finally:
            if upload is not None:
                await self._drop_upload(upload)


    async def _render_into_store(
        self,
        upload: Incoming,
        effective_type: DocType,
        source_url: str | None,
        mime: str,
        metadata: dict[str, Any] | None,
        existing: DocRow | None,
    ) -> AsyncIterator[events.Event]:
        """Render an upload and commit it, ending with `Committed`.

        A new doc gets its row first, incomplete, so the consumer can refer to it while
        its tiles stream; the row's primary key also settles who ingests the doc when
        two requests bring the same bytes. Its tiles are kept in memory and reach the
        store together with the blob, in one commit. A doc that exists but is not
        servable (quarantined) is rendered the same way, and nothing of it changes
        before that commit. An ingest that ends without it takes the row it inserted along.
        """
        doc_id = upload.doc_id

        if existing is None:
            if metadata is not None:
                metadata = self.schema.validated(metadata)

            created = await self.store.begin_doc(
                doc_id=doc_id,
                doc_type=effective_type,
                blob_rel_path=self.blobs.rel_path(doc_id),
                source_url=source_url,
                mime_type=mime,
                metadata=metadata,
                render_version=self.render_version,
            )

            if not created:  # another request inserted the row first
                raise self._in_progress(doc_id)

        elif not existing.complete:
            raise self._in_progress(doc_id)

        begin = events.Begin(doc_id, effective_type, source_url, self.render_version)
        committed = False

        try:
            # Identity up front, so the consumer can build point ids / mark pending before
            # any tile arrives.
            yield begin

            staged: list[TileRow] = []

            async with self.renderer.open(effective_type, upload.path, purpose="ingest", bulk=True) as doc:
                async with aclosing(doc.tiles()) as tiles:
                    async for tile in tiles:
                        staged.append(TileRow.from_rendered(doc_id, tile))
                        yield events.Tile(tile)

                height = doc.height

            # The blob first: one without tiles is inert, tiles without their blob are not.
            await self.blobs.commit(upload)
            await self.store.replace_render(
                doc_id,
                height=height,
                tile_count=len(staged),
                render_version=self.render_version,
                tiles=staged,
            )
            committed = True

        finally:
            if existing is None and not committed:
                with anyio.CancelScope(shield=True):
                    await self.store.delete_doc(doc_id)

        await self.tiles.forget(doc_id)

        metrics.DOCS.set(await self.store.count_docs())
        metrics.INGESTS.labels(doc_type=effective_type, result="rendered").inc()

        yield begin.committed(height=height, tile_count=len(staged), result="rendered")


    async def _drop_upload(self, upload: Incoming) -> None:
        """Remove a spooled upload (a no-op once committed) and the page capture cached
        under its path — even when the ingest is being cancelled."""
        with anyio.CancelScope(shield=True):
            await self.blobs.discard(upload)
            await self.renderer.forget(upload.path)


    @staticmethod
    def _in_progress(doc_id: str) -> ApiError:
        """409 for a doc another request is ingesting right now."""
        return conflict(
            f"doc_id {doc_id!r} is being ingested by another request", "INGEST_IN_PROGRESS", doc_id=doc_id
        )
