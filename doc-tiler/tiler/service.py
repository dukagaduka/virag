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
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator

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
from .store.blobs import BlobStore
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
        # doc_ids with a streaming ingest in flight — the sweeper must not reap these.
        self._inflight: set[str] = set()


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
        self, doc_type: DocType, doc_id: str, *, bulk: bool = False, height: int | None = None
    ) -> Document:
        """A stored document's blob, opened for rendering."""
        path = str(self.blobs.path(doc_id))

        return DOCUMENTS[doc_type](self.render, doc_type, path, bulk=bulk, height=height)


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
        data: bytes | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        force: bool,
    ) -> dict[str, Any]:
        """Ingest a document into the store and return its summary (no tile bytes).

        Runs the same pipeline as `ingest_stream`, only without emitting the tiles: an
        idempotent hit returns at once instead of replaying them, and a failure raises.
        """
        async for event in self._ingest(
            doc_type=doc_type, data=data, url=url, metadata=metadata, force=force, replay=False
        ):
            if event["event"] == "committed":
                return {key: value for key, value in event.items() if key != "event"}

        raise RuntimeError("ingest finished without a committed event")  # pragma: no cover


    async def ingest_stream(
        self,
        *,
        doc_type: DocType,
        data: bytes | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        force: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an ingest as NDJSON events: `begin`, one `tile` per tile, then `committed`.

        The response is already 200 when the first event leaves, so a failure surfaces as
        an `error` event; the absence of `committed` is the client's signal to roll back.
        """
        try:
            async for event in self._ingest(
                doc_type=doc_type, data=data, url=url, metadata=metadata, force=force, replay=True
            ):
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
        data: bytes | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        force: bool,
        replay: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """The ingest pipeline: `begin`, `tile` events carrying a `RenderedTile`, `committed`.

        Tiles are written to the store as they render (the doc stays `complete=0` until
        the last one), so an ingest that dies mid-way leaves a swept-able partial and
        never a servable half-doc. An idempotent hit replays the stored tiles when
        `replay` is set and goes straight to `committed` otherwise. Failures raise.
        """
        source_url: str | None = None

        if doc_type == DocType.URL:
            if not url:
                raise bad_request("url is required when doc_type=URL", "MISSING_URL")

            blob = await self.capture.capture(url)
            effective_type = DocType.MHTML
            source_url = url
            mime = "multipart/related"

        else:
            if data is None:
                raise bad_request(f"document is required when doc_type={doc_type}", "MISSING_FILE")

            blob = data
            effective_type = doc_type
            mime = media.check_and_mime(doc_type, blob)

        doc_id = sha256_hex(blob)

        # Idempotency: identical bytes never re-render, but only a fully-written doc
        # short-circuits; an incomplete (crashed/streaming) doc must be re-rendered.
        existing = None if force else await self.store.get_doc(doc_id)

        if self._servable(existing):
            metrics.INGESTS.labels(doc_type=effective_type, result="idempotent").inc()

            async for event in self._replay_from_store(existing, replay):
                yield event

            return

        if metadata is not None:
            metadata = self.validate_metadata_payload(metadata)

        self._inflight.add(doc_id)

        try:
            async for event in self._render_into_store(
                doc_id, effective_type, blob, source_url, mime, metadata
            ):
                yield event

        except ApiError as exc:
            log.warning("ingest failed for %s: %s", doc_id, exc.message)
            raise

        finally:
            self._inflight.discard(doc_id)


    async def _render_into_store(
        self,
        doc_id: str,
        effective_type: DocType,
        blob: bytes,
        source_url: str | None,
        mime: str,
        metadata: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Render a new doc tile by tile into the store, ending with `committed`."""
        rel = await self.blobs.put(doc_id, blob)

        await self.store.begin_doc(
            doc_id=doc_id,
            doc_type=effective_type,
            blob_rel_path=rel,
            source_url=source_url,
            height=0,
            tile_count=0,
            mime_type=mime,
            metadata=metadata,
            render_version=self.settings.render_version,
        )

        # Identity up front, so the consumer can build point ids / mark pending before
        # any tile arrives.
        yield {
            "event": "begin",
            "doc_id": doc_id,
            "doc_type": effective_type,
            "source_url": source_url,
            "render_version": self.settings.render_version,
        }

        count = [0]

        with self._measured(effective_type, "ingest", count):
            async with self._document(effective_type, doc_id, bulk=True) as doc:
                async for batch in doc.render_all(await doc.plan()):
                    await self.store.append_tiles(doc_id, batch)

                    for tile in batch:
                        count[0] += 1
                        yield {"event": "tile", "tile": tile}

                height = doc.height

        await self.store.finish_doc(doc_id, tile_count=count[0], height=height)
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
            "tile_count": count[0],
            "result": "rendered",
        }


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
        """A stored doc an idempotent ingest may replay: complete and rendered by this
        render_version. A quarantined doc (still at an older one) is re-rendered instead."""
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
        tiles: list[RenderedTile] = []
        count = [0]

        with self._measured(row.doc_type, "revise", count):
            async with self._document(row.doc_type, row.doc_id) as doc:
                async for batch in doc.render_all(await doc.plan()):
                    tiles.extend(batch)

                height = doc.height

            count[0] = len(tiles)

        await self.store.replace_render(
            row.doc_id,
            height=height,
            tile_count=len(tiles),
            render_version=self.settings.render_version,
            tiles=tiles,
        )
        await self._invalidate(row.doc_id)


    async def sweep_incomplete(self, ttl_seconds: int) -> int:
        """Delete incomplete ingests older than the TTL, skipping any in flight. Returns the count."""
        cutoff = int(time.time()) - ttl_seconds
        stale = [d for d in await self.store.list_incomplete(cutoff) if d not in self._inflight]

        for doc_id in stale:
            await self.delete(doc_id)

        if stale:
            log.info("sweeper: removed %d incomplete ingest(s)", len(stale))

        return len(stale)


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
            async with self._document(row.doc_type, row.doc_id, height=row.height) as doc:
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
        """Delete a document's rows, blob and cached tiles; return whether it existed."""
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


__all__ = ["TilerService", "RenderedTile", "ApiError"]
