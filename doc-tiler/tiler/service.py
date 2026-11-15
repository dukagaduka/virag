"""Application core: ingest, regeneration and revision over the stores.

The API modules stay thin — parse, delegate here, serialise. The service stores what a
document renders; what a document is, how it is cut into tiles and where it renders
(PDF raster, image, Chromium) is the render layer's business (`render/documents.py`).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import aclosing, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import anyio
import orjson

from . import media, metrics
from .capture import UrlCapture
from .config import Settings
from .errors import ApiError, bad_request, conflict, gone, not_found, unprocessable
from .hashing import sha256_hex
from .meta.schema import MetadataSchema, MetadataViolation, validate_metadata
from .models import DocType
from .render.documents import DOCUMENTS, Document
from .render.geometry import Tile
from .render.pool import RenderPool
from .render.types import RenderedTile
from .revision import Revision
from .store.blobs import BlobStore, Incoming
from .store.cache import AsyncLRU
from .store.meta import DocRow, MetaStore, TileRow

log = logging.getLogger("tiler.service")


class TilerService:
    """Owns the stores, the render pool and the tile cache; runs ingest and regeneration."""


    def __init__(self, settings: Settings, schema: MetadataSchema):
        self.settings = settings
        self.schema = schema
        self.store = MetaStore(settings.db_path)
        self.blobs = BlobStore(settings.blob_dir)
        self.render = RenderPool(settings)
        self.capture = UrlCapture(settings, self.render)
        self.tile_cache = AsyncLRU(settings.render_cache_size, "tile")
        self.revision = Revision(self)
        self.started_at = time.time()
        self.volume_id = ""  # set by bootstrap from service_meta


    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start_pools(self) -> None:
        """Start the SSRF guard proxy and the render pool."""
        await self.capture.start()
        await self.render.start(
            browser=self.settings.browser_enabled, per_context_proxy=self.capture.proxied
        )


    async def stop_pools(self) -> None:
        """Tear down the render pool and the guard proxy on shutdown."""
        await self.render.stop()
        await self.capture.stop()


    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _document(
        self, doc_type: DocType, path: Path, *, bulk: bool = False, height: int | None = None
    ) -> Document:
        """The document in the file at `path` (a stored blob or a spooled upload),
        opened for rendering."""
        return DOCUMENTS[doc_type](self.render, doc_type, str(path), bulk=bulk, height=height)


    @contextmanager
    def _measured(self, doc_type: DocType, path: str, count: list[int]) -> Iterator[None]:
        """Record one document render (`count[0]` holds its tile count when done)."""
        started = time.perf_counter()

        try:
            yield

        finally:
            metrics.RENDER_SECONDS.labels(doc_type=doc_type, path=path).observe(time.perf_counter() - started)

        metrics.RENDERS.labels(doc_type=doc_type, path=path).inc()
        metrics.TILES.labels(doc_type=doc_type, path=path).inc(count[0])


    # ------------------------------------------------------------------ #
    # ingest
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Ingest a document into the store and return its summary (no tile bytes).

        Runs the same pipeline as `ingest_stream`, only without emitting the tiles: an
        idempotent hit returns at once instead of replaying them, and a failure raises.
        """
        summary = None

        async for event in self._ingest(
            doc_type=doc_type, upload=upload, url=url, metadata=metadata, replay=False
        ):
            if event["event"] == "committed":
                summary = {key: value for key, value in event.items() if key != "event"}

        if summary is None:  # pragma: no cover
            raise RuntimeError("ingest finished without a committed event")

        return summary


    async def ingest_stream(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an ingest as NDJSON events: `begin`, one `tile` per tile, then `committed`.

        The response is already 200 when the first event leaves, so a failure surfaces as
        an `error` event; the absence of `committed` is the client's signal to roll back.
        """
        events = self._ingest(doc_type=doc_type, upload=upload, url=url, metadata=metadata, replay=True)

        try:
            # Closed with this stream: a client that goes away ends the ingest at once.
            async with aclosing(events):
                async for event in events:
                    if event["event"] == "tile":
                        tile: RenderedTile = event["tile"]
                        event = {
                            "event": "tile",
                            **self._tile_meta(tile.tile_num, tile.height, tile.image_hash, tile.jpeg),
                        }

                    yield event

        except ApiError as exc:
            yield {"event": "error", "code": exc.code, "message": exc.message}

        except Exception as exc:  # noqa: BLE001 - the stream is already 200; report in-band
            log.exception("streaming ingest crashed")
            yield {"event": "error", "code": "INTERNAL_ERROR", "message": repr(exc)}


    async def _ingest(
        self,
        *,
        doc_type: DocType,
        upload: Incoming | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        replay: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """The ingest pipeline: `begin`, `tile` events carrying a `RenderedTile`, `committed`.

        The upload (taken over here) waits in `.incoming` while it renders. Bytes the
        store already holds and serves are not rendered again: their tiles are replayed
        when `replay` is set, and `committed` follows at once otherwise. Whatever ends
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
            if self._servable(existing):
                metrics.INGESTS.labels(doc_type=effective_type, result="idempotent").inc()
                events = self._replay_from_store(existing, replay)

            else:
                events = self._render_into_store(upload, effective_type, source_url, mime, metadata, existing)

            try:
                async with aclosing(events):
                    async for event in events:
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Render an upload and commit it, ending with `committed`.

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
                metadata = self.validate_metadata_payload(metadata)

            created = await self.store.begin_doc(
                doc_id=doc_id,
                doc_type=effective_type,
                blob_rel_path=self.blobs.rel_path(doc_id),
                source_url=source_url,
                mime_type=mime,
                metadata=metadata,
                render_version=self.settings.render_version,
            )

            if not created:  # another request inserted the row first
                raise self._in_progress(doc_id)

        elif not existing.complete:
            raise self._in_progress(doc_id)

        committed = False

        try:
            # Identity up front, so the consumer can build point ids / mark pending before
            # any tile arrives.
            yield {
                "event": "begin",
                "doc_id": doc_id,
                "doc_type": effective_type,
                "source_url": source_url,
                "render_version": self.settings.render_version,
            }

            staged: list[TileRow] = []
            count = [0]

            with self._measured(effective_type, "ingest", count):
                async with self._document(effective_type, upload.path, bulk=True) as doc:
                    async with aclosing(doc.render_all(await doc.plan())) as batches:
                        async for batch in batches:
                            for tile in batch:
                                staged.append(self._tile_row(doc_id, tile))
                                count[0] += 1
                                yield {"event": "tile", "tile": tile}

                    height = doc.height

            # The blob first: one without tiles is inert, tiles without their blob are not.
            await self.blobs.commit(upload)
            await self.store.replace_render(
                doc_id,
                height=height,
                tile_count=len(staged),
                render_version=self.settings.render_version,
                tiles=staged,
            )
            committed = True

        finally:
            if existing is None and not committed:
                with anyio.CancelScope(shield=True):
                    await self.store.delete_doc(doc_id)

        await self._invalidate(doc_id)

        metrics.DOCS.set(await self.store.count_docs())
        metrics.INGESTS.labels(doc_type=effective_type, result="rendered").inc()

        yield {
            "event": "committed",
            "doc_id": doc_id,
            "doc_type": effective_type,
            "source_url": source_url,
            "render_version": self.settings.render_version,
            "height": height,
            "tile_count": len(staged),
            "result": "rendered",
        }


    async def _drop_upload(self, upload: Incoming) -> None:
        """Remove a spooled upload (a no-op once committed) and the page capture cached
        under its path — even when the ingest is being cancelled."""
        path = str(upload.path)

        with anyio.CancelScope(shield=True):
            await self.blobs.discard(upload)
            await self.render.page_cache.invalidate(lambda k: k == path)


    async def _replay_from_store(self, row: DocRow, replay: bool) -> AsyncIterator[dict[str, Any]]:
        """An idempotent hit: `begin`, the stored tiles regenerated (only when `replay`
        is set), then `committed`."""
        yield {
            "event": "begin",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
        }

        if replay:
            for tile in await self.store.get_tiles(row.doc_id):
                jpeg, _, _ = await self.get_tile(row.doc_id, tile.tile_num, verify=True)
                yield {
                    "event": "tile",
                    "tile": RenderedTile(
                        tile_num=tile.tile_num,
                        offset=tile.offset,
                        height=tile.height,
                        jpeg=jpeg,
                        image_hash=tile.image_hash,
                    ),
                }

        yield {
            "event": "committed",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
            "height": row.height,
            "tile_count": row.tile_count,
            "result": "idempotent",
        }


    def _servable(self, row: DocRow | None) -> bool:
        """A stored doc an ingest of the same bytes may replay: complete and rendered by
        this render_version. A quarantined doc (still at an older one) is rendered again."""
        return (
            row is not None
            and row.complete
            and row.render_version == self.settings.render_version
            and not row.quarantined
        )


    async def revise_doc(self, row: DocRow) -> None:
        """Re-render one stored doc from its blob and swap its tiles and height in place
        at the current render_version (doc_id and metadata are untouched)."""
        self._require_blob(row)
        staged: list[TileRow] = []
        count = [0]

        with self._measured(row.doc_type, "revise", count):
            async with self._document(row.doc_type, self.blobs.path(row.doc_id)) as doc:
                async for batch in doc.render_all(await doc.plan()):
                    staged.extend(self._tile_row(row.doc_id, tile) for tile in batch)

                height = doc.height

            count[0] = len(staged)

        await self.store.replace_render(
            row.doc_id,
            height=height,
            tile_count=len(staged),
            render_version=self.settings.render_version,
            tiles=staged,
        )
        await self._invalidate(row.doc_id)


    def validate_metadata_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata against the schema, raising 422 on any violation."""
        try:
            validated = validate_metadata(metadata, self.schema)

        except MetadataViolation as exc:
            raise unprocessable(str(exc), "METADATA_SCHEMA_VIOLATION", key=exc.key) from exc

        try:
            orjson.dumps(validated)

        except orjson.JSONEncodeError as exc:  # e.g. an integer beyond 64 bits
            raise unprocessable(f"metadata is not storable as JSON: {exc}", "INVALID_METADATA_JSON") from exc

        return validated


    def _tile_meta(
        self, tile_num: int, height: int, image_hash: str, jpeg: bytes | None
    ) -> dict[str, Any]:
        """Build one tile's manifest entry, base64-encoding its bytes when present."""
        meta: dict[str, Any] = {
            "tile_num": tile_num,
            "width": self.settings.tile_width,  # always TILE_WIDTH
            "height": height,
            "image_hash": image_hash,
        }

        if jpeg is not None:
            meta["data"] = base64.b64encode(jpeg).decode("ascii")

        return meta


    # ------------------------------------------------------------------ #
    # regeneration
    # ------------------------------------------------------------------ #
    async def get_tile(self, doc_id: str, tile_num: int, verify: bool) -> tuple[bytes, TileRow, DocRow]:
        """Return a single tile's bytes, regenerating and caching it on demand."""
        row = await self.store.get_doc(doc_id)

        if row is None:
            raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)

        if not row.complete:
            raise not_found(
                f"doc_id {doc_id!r} is still being ingested", "DOC_INCOMPLETE", doc_id=doc_id
            )

        if row.quarantined:
            raise gone(
                f"doc_id {doc_id!r} is quarantined: {row.quarantine_reason}",
                doc_id=doc_id,
                reason=row.quarantine_reason,
            )

        if tile_num < 0 or tile_num >= row.tile_count:
            raise bad_request(
                f"tile_num {tile_num} out of range (tile_count={row.tile_count})",
                "TILE_OUT_OF_RANGE",
                doc_id=doc_id,
                tile_count=row.tile_count,
            )

        tile_row = await self.store.get_tile(doc_id, tile_num)

        if tile_row is None:  # pragma: no cover - implies a partial write
            raise not_found(f"tile {tile_num} of {doc_id!r} is missing", "UNKNOWN_TILE")

        key = (doc_id, tile_num)
        jpeg = await self.tile_cache.get_or_create(key, lambda: self._regenerate(row, tile_row))

        if verify:
            actual = sha256_hex(jpeg)

            # Every served doc was rendered by this render_version (revision guarantees
            # it), so a mismatch here is non-determinism — a bug, never expected drift.
            if actual != tile_row.image_hash:
                metrics.MISMATCHES.labels(doc_type=row.doc_type).inc()

                message = (
                    f"regenerated tile ({doc_id}, {tile_num}) hashes to {actual} but "
                    f"{tile_row.image_hash} was stored at render_version {row.render_version}"
                )
                log.warning("%s — refusing", message)

                raise conflict(message, "RENDER_MISMATCH", doc_id=doc_id, tile_num=tile_num)

        return jpeg, tile_row, row


    async def _regenerate(self, row: DocRow, tile_row: TileRow) -> bytes:
        """Re-render one tile from the stored blob: the rows it was cut from at ingest."""
        self._require_blob(row)
        tile = Tile(tile_row.tile_num, tile_row.offset, tile_row.height)

        with self._measured(row.doc_type, "regenerate", [1]):
            async with self._document(row.doc_type, self.blobs.path(row.doc_id), height=row.height) as doc:
                [image] = await doc.render([tile])

        return image.jpeg


    def _require_blob(self, row: DocRow) -> None:
        """Fail with 404 when a document's blob is gone from storage."""
        if not self.blobs.exists(row.doc_id):
            raise not_found(f"blob for {row.doc_id!r} is missing from storage", "MISSING_BLOB")


    async def _invalidate(self, doc_id: str) -> None:
        """Drop every cached tile and page capture belonging to one document."""
        path = str(self.blobs.path(doc_id))

        await self.tile_cache.invalidate(lambda k: isinstance(k, tuple) and k[0] == doc_id)
        await self.render.page_cache.invalidate(lambda k: k == path)


    # ------------------------------------------------------------------ #
    async def delete(self, doc_id: str) -> bool:
        """Delete a document's rows, blob and cached tiles; return whether it existed.
        A doc still being ingested is its ingest's to remove: 409."""
        row = await self.store.get_doc(doc_id)

        if row is not None and not row.complete:
            raise self._in_progress(doc_id)

        removed = await self.store.delete_doc(doc_id)

        if removed:
            await self.blobs.delete(doc_id)
            await self._invalidate(doc_id)
            metrics.DOCS.set(await self.store.count_docs())

        return removed


    async def ready(self) -> tuple[bool, dict[str, Any]]:
        """Probe the DB, blob store, browser and revision gate for the readiness endpoint."""
        checks: dict[str, Any] = {}
        ok = True

        try:
            await self.store.db.execute("SELECT 1")
            checks["db"] = "ok"

        except Exception as exc:
            checks["db"] = f"error: {exc}"
            ok = False

        try:
            await asyncio.to_thread(self.blobs.ensure_root)
            checks["blobs"] = "ok"

        except Exception as exc:
            checks["blobs"] = f"error: {exc}"
            ok = False

        pool_ok = await self.render.healthy()
        checks["render_pool"] = "ok" if pool_ok else "down"

        if not pool_ok:
            ok = False

        browser = self.render.browser

        if self.settings.browser_enabled:
            checks["browser"] = "ok" if browser.available else f"unavailable: {browser.start_error}"

            if not browser.available:
                ok = False

        else:
            checks["browser"] = "disabled"

        # Not ready while docs from another render_version remain (see GET /revision).
        checks["revision"] = self.revision.status["state"]

        if not self.revision.serving:
            ok = False

        # Counted live: a re-ingest lifts a quarantine without any revision running.
        quarantined = await self.store.count_quarantined()
        metrics.QUARANTINED.set(quarantined)
        checks["quarantined"] = quarantined

        return ok, checks


    @staticmethod
    def _tile_row(doc_id: str, tile: RenderedTile) -> TileRow:
        """What the store keeps of a rendered tile: its place and its hash, not its bytes."""
        return TileRow(
            doc_id=doc_id,
            tile_num=tile.tile_num,
            offset=tile.offset,
            height=tile.height,
            image_hash=tile.image_hash,
        )


    @staticmethod
    def _in_progress(doc_id: str) -> ApiError:
        """409 for a doc another request is ingesting right now."""
        return conflict(
            f"doc_id {doc_id!r} is being ingested by another request", "INGEST_IN_PROGRESS", doc_id=doc_id
        )


__all__ = ["TilerService", "RenderedTile", "ApiError"]
