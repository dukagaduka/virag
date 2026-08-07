"""Application core: wires the execution pools to the stores.

The API modules stay thin — parse, delegate here, serialise. Everything that
touches a pool, the DB or the blob store lives in `TilerService`, which is also
the single place that knows how ingest and regeneration must agree.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Sequence

from . import media, metrics
from .config import Settings
from .errors import ApiError, bad_gateway, bad_request, conflict, forbidden, gateway_timeout
from .errors import not_found, unprocessable
from .hashing import sha256_hex
from .meta.schema import MetadataSchema, MetadataViolation, validate_metadata
from .netguard import GuardProxy
from .render import browser as web
from .render import cpu
from .render.browser import BrowserPool, BrowserUnavailable, RenderFailure, RenderTimeout
from .render.geometry import Strip, slice_doc_strips, slice_strips, strip_for
from .ssrf import BlockedTarget, validate_url
from .store.blobs import BlobStore
from .store.cache import AsyncLRU
from .store.meta import DocRow, MetaStore, TileRow

log = logging.getLogger("tiler.service")

MAX_PDF_CHUNK = 16  # pages per process-pool task: bounds pickle size per hop


@dataclass(frozen=True)
class RenderedTile:
    """One finished tile: its bytes, dimensions and content hash."""

    tile_num: int
    height: int
    jpeg: bytes
    image_hash: str


@dataclass(frozen=True)
class RenderedDoc:
    """A rendered document: full-page height plus its ordered tiles.

    `page_heights` is set only for a sliced PDF (per-page resized heights, in page
    order) — get_tile replays the same per-page slicing to reverse a global tile_num.
    """

    page_height: int
    tiles: list[RenderedTile]
    page_heights: list[int] | None = None


class TilerService:
    """Owns the pools, stores and caches; runs ingest and tile regeneration."""

    def __init__(self, settings: Settings, schema: MetadataSchema):
        self.settings = settings
        self.schema = schema
        self.store = MetaStore(settings.db_path)
        self.blobs = BlobStore(settings.blob_dir)
        self.browser = BrowserPool(settings)
        self.guard_proxy: GuardProxy | None = None
        self.tile_cache = AsyncLRU(settings.render_cache_size, "tile")
        self.page_cache = AsyncLRU(max(2, settings.render_cache_size // 8), "page")
        self._cpu: concurrent.futures.ProcessPoolExecutor | None = None
        self.started_at = time.time()
        # doc_ids with a streaming ingest in flight — the sweeper must not reap these.
        self._inflight: set[str] = set()


    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start_pools(self) -> None:
        """Start the CPU process pool, SSRF proxy and browser pool."""
        import multiprocessing  # noqa: PLC0415

        # 'spawn': the parent runs an event loop and Playwright threads; forking
        # those into workers is a well-known source of deadlocks.
        self._cpu = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.settings.cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        if self.settings.url_fetch_enabled and not self.settings.url_allow_private:
            self.guard_proxy = GuardProxy(allow_private=False)
            await self.guard_proxy.start()
        if self.settings.browser_enabled:
            await self.browser.start(per_context_proxy=self.guard_proxy is not None)


    async def stop_pools(self) -> None:
        """Tear down the browser, proxy and CPU pool on shutdown."""
        await self.browser.stop()

        if self.guard_proxy is not None:
            await self.guard_proxy.stop()

        if self._cpu is not None:
            self._cpu.shutdown(wait=False, cancel_futures=True)
            self._cpu = None


    async def run_cpu(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Dispatch CPU-bound work to the process pool so the event loop never blocks."""
        if self._cpu is None:
            raise RuntimeError("CPU pool is not running")
        
        loop = asyncio.get_running_loop()
        metrics.INFLIGHT.labels(layer="cpu").inc()

        try:
            return await loop.run_in_executor(self._cpu, fn, *args)
        
        except cpu.DecodeError as exc:
            raise unprocessable(str(exc), "DECODE_FAILED") from exc
        
        except concurrent.futures.process.BrokenProcessPool as exc:  # pragma: no cover
            log.error("CPU pool broke: %s", exc)
            raise bad_gateway("render worker crashed", "WORKER_CRASH") from exc
        
        finally:
            metrics.INFLIGHT.labels(layer="cpu").dec()


    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def _strips_for(self, page_height: int) -> list[Strip]:
        """Slice a full-page height into tile strips using the configured geometry."""
        s = self.settings
        return slice_strips(page_height, s.tile_height, s.tile_overlap, s.min_tile_height)


    async def _render_pdf(self, blob: bytes) -> RenderedDoc:
        """Rasterize a PDF, chunked across the CPU pool.

        Default: one tile per page. With `pdf_slice` on: each page is sliced into
        strips (never across pages) and tiles get a contiguous global tile_num in
        page order; per-page heights are captured so get_tile can reverse-map.
        """
        s = self.settings

        pages = await self.run_cpu(cpu.pdf_page_count, blob)
        if pages <= 0:
            raise unprocessable("PDF has no pages", "EMPTY_DOCUMENT")

        chunk = max(1, min(MAX_PDF_CHUNK, math.ceil(pages / max(1, s.cpu_workers))))
        groups = [list(range(i, min(i + chunk, pages))) for i in range(0, pages, chunk)]

        if not s.pdf_slice:
            results = await asyncio.gather(
                *(
                    self.run_cpu(cpu.render_pdf_pages, blob, g, s.pdf_dpi, s.tile_width, s.jpeg_quality)
                    for g in groups
                )
            )
            tiles = [
                RenderedTile(tile_num=num, height=h, jpeg=jpeg, image_hash=sha256_hex(jpeg))
                for group in results
                for num, h, jpeg in group
            ]
            tiles.sort(key=lambda t: t.tile_num)

            return RenderedDoc(page_height=0, tiles=tiles)  # page_height is 0 for PDF

        results = await asyncio.gather(
            *(
                self.run_cpu(
                    cpu.render_pdf_page_strips, blob, g, s.pdf_dpi, s.tile_width,
                    s.pdf_tile_height, s.pdf_tile_overlap, s.min_tile_height, s.jpeg_quality,
                )
                for g in groups
            )
        )
        per_page = sorted((p for group in results for p in group), key=lambda x: x[0])
        page_heights = [ph for _, ph, _ in per_page]

        tiles = []
        running = 0
        for _, _, strips in per_page:
            for offset, sh, jpeg in strips:
                tiles.append(
                    RenderedTile(tile_num=running, height=sh, jpeg=jpeg, image_hash=sha256_hex(jpeg))
                )
                running += 1

        return RenderedDoc(page_height=0, tiles=tiles, page_heights=page_heights)


    async def _render_img(self, blob: bytes) -> RenderedDoc:
        """Resize an image to tile width and slice it into vertical strips."""
        s = self.settings
        page_height, raw = await self.run_cpu(
            cpu.tile_full_page,
            blob,
            s.tile_width,
            s.tile_height,
            s.tile_overlap,
            s.min_tile_height,
            s.jpeg_quality,
        )

        return RenderedDoc(
            page_height=page_height,
            tiles=[
                RenderedTile(tile_num=n, height=h, jpeg=j, image_hash=sha256_hex(j))
                for n, h, j in raw
            ],
        )
    

    async def _render_mhtml(self, blob: bytes) -> RenderedDoc:
        """Render a stored MHTML archive to tiles via the browser pool."""
        s = self.settings
        rendered = await self._browser_call(
            web.render_mhtml_document, self.browser, s, blob, self._strips_for
        )
        if rendered.full_png is not None:
            page_height, raw = await self.run_cpu(
                cpu.tile_full_page,
                rendered.full_png,
                s.tile_width,
                s.tile_height,
                s.tile_overlap,
                s.min_tile_height,
                s.jpeg_quality,
            )
            tiles = [
                RenderedTile(tile_num=n, height=h, jpeg=j, image_hash=sha256_hex(j))
                for n, h, j in raw
            ]

            return RenderedDoc(page_height=page_height, tiles=tiles)

        # Very tall page: clip-captured strips, one per tile.
        pngs = rendered.strip_pngs or []
        encoded = await asyncio.gather(
            *(self.run_cpu(cpu.encode_strip, png, s.tile_width, s.jpeg_quality) for png in pngs)
        )
        tiles = [
            RenderedTile(tile_num=i, height=h, jpeg=j, image_hash=sha256_hex(j))
            for i, (h, j) in enumerate(encoded)
        ]
        return RenderedDoc(page_height=rendered.css_height, tiles=tiles)


    async def render_document(self, effective_type: str, blob: bytes, path: str) -> RenderedDoc:
        """Render any supported doc type to tiles, recording render metrics."""
        started = time.perf_counter()
        try:
            if effective_type == "PDF":
                doc = await self._render_pdf(blob)
            elif effective_type == "IMG":
                doc = await self._render_img(blob)
            elif effective_type == "MHTML":
                doc = await self._render_mhtml(blob)
            else:  # pragma: no cover - guarded upstream
                raise bad_request(f"unsupported doc_type {effective_type!r}")
        finally:
            metrics.RENDER_SECONDS.labels(doc_type=effective_type, path=path).observe(
                time.perf_counter() - started
            )
        metrics.RENDERS.labels(doc_type=effective_type, path=path).inc()
        metrics.TILES.labels(doc_type=effective_type, path=path).inc(len(doc.tiles))
        return doc


    async def _browser_call(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a browser render, mapping its failures to HTTP errors."""
        metrics.INFLIGHT.labels(layer="browser").inc()
        try:
            return await fn(*args)
        except BrowserUnavailable as exc:
            raise bad_gateway(f"Chromium is unavailable: {exc}", "BROWSER_UNAVAILABLE") from exc
        except RenderTimeout as exc:
            raise gateway_timeout(f"render timed out: {exc}") from exc
        except RenderFailure as exc:
            raise bad_gateway(f"render failed: {exc}") from exc
        finally:
            metrics.INFLIGHT.labels(layer="browser").dec()


    # ------------------------------------------------------------------ #
    # ingest
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        *,
        doc_type: str,
        data: bytes | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        store: bool,
        include_bytes: bool,
        force: bool,
    ) -> dict[str, Any]:
        """Ingest a document: render it to tiles and, unless store=false, persist it."""
        if not store:
            include_bytes = True  # store=false is a pure transform: nothing is written
            metadata = None

        source_url: str | None = None
        if doc_type == "URL":
            if not url:
                raise bad_request("url is required when doc_type=URL", "MISSING_URL")
            blob = await self._capture_url(url)
            effective_type = "MHTML"
            source_url = url
            mime = "multipart/related"

        else:
            if data is None:
                raise bad_request(f"document is required when doc_type={doc_type}", "MISSING_FILE")
            
            blob = data
            effective_type = doc_type
            mime = media.check_and_mime(doc_type, blob)

        doc_id = sha256_hex(blob)

        # Idempotency: identical bytes never re-render — but only a fully-written doc
        # short-circuits; an incomplete (crashed/streaming) doc must be re-rendered.
        if store and not force:
            existing = await self.store.get_doc(doc_id)
            if existing is not None and existing.complete:
                metrics.INGESTS.labels(
                    doc_type=effective_type, stored="true", result="idempotent"
                ).inc()
                return await self._manifest_from_store(existing, include_bytes)

        if store and metadata is not None:
            metadata = self.validate_metadata_payload(metadata)

        rendered = await self.render_document(effective_type, blob, path="ingest")

        if store:
            # Blob first: a blob without a row is inert and dedups away, whereas a row
            # without a blob would make /get_tile permanently fail.
            rel = await self.blobs.put(doc_id, blob)
            await self.store.upsert_doc(
                doc_id=doc_id,
                doc_type=effective_type,
                blob_rel_path=rel,
                source_url=source_url,
                page_height=rendered.page_height,
                tile_count=len(rendered.tiles),
                mime_type=mime,
                metadata=metadata,
                render_version=self.settings.render_version,
                tiles=[(t.tile_num, t.height, t.image_hash) for t in rendered.tiles],
                page_heights=rendered.page_heights,
            )
            await self._invalidate(doc_id)
            metrics.DOCS.set(await self.store.count_docs())

        metrics.INGESTS.labels(
            doc_type=effective_type, stored=str(store).lower(), result="rendered"
        ).inc()

        return {
            "doc_id": doc_id,
            "doc_type": effective_type,
            "source_url": source_url,
            "render_version": self.settings.render_version,
            "stored": store,
            "page_height": rendered.page_height,
            "tile_count": len(rendered.tiles),
            "tiles": [
                self._tile_meta(t.tile_num, t.height, t.image_hash, t.jpeg if include_bytes else None)
                for t in rendered.tiles
            ],
        }


    # ------------------------------------------------------------------ #
    # streaming ingest
    # ------------------------------------------------------------------ #
    async def _render_pdf_stream(
        self, blob: bytes
    ) -> AsyncIterator[tuple[list[RenderedTile], list[int]]]:
        """Render a PDF group-by-group, yielding `(tiles, page_heights_delta)`.

        Default (page-per-tile): groups are consumed as they finish — order-independent,
        since tile_num == page number — and page_heights_delta is empty. With `pdf_slice`
        on: groups are consumed in page order so strip tile_nums form a contiguous global
        index, and each group reports its pages' resized heights (in order) so get_tile
        can reverse-map.
        """
        s = self.settings

        pages = await self.run_cpu(cpu.pdf_page_count, blob)
        if pages <= 0:
            raise unprocessable("PDF has no pages", "EMPTY_DOCUMENT")

        chunk = max(1, min(MAX_PDF_CHUNK, math.ceil(pages / max(1, s.cpu_workers))))
        groups = [list(range(i, min(i + chunk, pages))) for i in range(0, pages, chunk)]

        if not s.pdf_slice:
            tasks = [
                asyncio.ensure_future(
                    self.run_cpu(cpu.render_pdf_pages, blob, g, s.pdf_dpi, s.tile_width, s.jpeg_quality)
                )
                for g in groups
            ]
            try:
                for coro in asyncio.as_completed(tasks):
                    group = await coro
                    yield (
                        [
                            RenderedTile(tile_num=num, height=h, jpeg=jpeg, image_hash=sha256_hex(jpeg))
                            for num, h, jpeg in group
                        ],
                        [],
                    )
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise
            return

        # Sliced: consume in submission (page) order so strip tile_nums number globally.
        tasks = [
            asyncio.ensure_future(
                self.run_cpu(
                    cpu.render_pdf_page_strips, blob, g, s.pdf_dpi, s.tile_width,
                    s.pdf_tile_height, s.pdf_tile_overlap, s.min_tile_height, s.jpeg_quality,
                )
            )
            for g in groups
        ]
        running = 0
        try:
            for task in tasks:
                group = await task
                tiles: list[RenderedTile] = []
                heights: list[int] = []
                for _, page_height, strips in group:
                    heights.append(page_height)
                    for offset, sh, jpeg in strips:
                        tiles.append(
                            RenderedTile(tile_num=running, height=sh, jpeg=jpeg, image_hash=sha256_hex(jpeg))
                        )
                        running += 1
                yield (tiles, heights)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise


    async def ingest_stream(
        self,
        *,
        doc_type: str,
        data: bytes | None,
        url: str | None,
        metadata: dict[str, Any] | None,
        force: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream an ingest as NDJSON events: one per tile, then a terminal `committed`.

        Tiles are written to the store as they render (the doc stays `complete=0` until
        the last one), so a stream that dies mid-way leaves a swept-able partial and
        never a servable half-doc. A failure surfaces as an `error` event; the absence
        of `committed` is the client's signal to roll back.
        """
        source_url: str | None = None

        try:
            if doc_type == "URL":
                if not url:
                    raise bad_request("url is required when doc_type=URL", "MISSING_URL")
                blob = await self._capture_url(url)
                effective_type = "MHTML"
                source_url = url
                mime = "multipart/related"

            else:
                if data is None:
                    raise bad_request(f"document is required when doc_type={doc_type}", "MISSING_FILE")
                blob = data
                effective_type = doc_type
                mime = media.check_and_mime(doc_type, blob)

            doc_id = sha256_hex(blob)

            existing = None if force else await self.store.get_doc(doc_id)
            if existing is not None and existing.complete:
                metrics.INGESTS.labels(
                    doc_type=effective_type, stored="true", result="idempotent"
                ).inc()
                async for event in self._stream_from_store(existing):
                    yield event
                return

            if metadata is not None:
                metadata = self.validate_metadata_payload(metadata)

        except ApiError as exc:
            yield {"event": "error", "code": exc.code, "message": exc.message}
            return

        self._inflight.add(doc_id)

        try:
            rel = await self.blobs.put(doc_id, blob)

            await self.store.begin_doc(
                doc_id=doc_id,
                doc_type=effective_type,
                blob_rel_path=rel,
                source_url=source_url,
                page_height=0,
                tile_count=0,
                mime_type=mime,
                metadata=metadata,
                render_version=self.settings.render_version,
            )

            # Emit identity up front so the consumer can build point ids / mark pending
            # before any tile arrives.
            yield {
                "event": "begin",
                "doc_id": doc_id,
                "doc_type": effective_type,
                "source_url": source_url,
                "render_version": self.settings.render_version,
            }

            count = 0
            page_height = 0
            page_heights_acc: list[int] = []

            if effective_type == "PDF":
                started = time.perf_counter()
                async for group, heights in self._render_pdf_stream(blob):
                    page_heights_acc.extend(heights)
                    await self.store.append_tiles(
                        doc_id, [(t.tile_num, t.height, t.image_hash) for t in group]
                    )
                    for tile in group:
                        count += 1
                        yield {"event": "tile", **self._tile_meta(tile.tile_num, tile.height, tile.image_hash, tile.jpeg)}
                metrics.RENDER_SECONDS.labels(doc_type=effective_type, path="ingest").observe(
                    time.perf_counter() - started
                )
                metrics.RENDERS.labels(doc_type=effective_type, path="ingest").inc()
                metrics.TILES.labels(doc_type=effective_type, path="ingest").inc(count)

            else:
                # IMG/MHTML render as a single unit (small); stream the transport only.
                rendered = await self.render_document(effective_type, blob, path="ingest")
                page_height = rendered.page_height
                await self.store.append_tiles(
                    doc_id, [(t.tile_num, t.height, t.image_hash) for t in rendered.tiles]
                )
                for tile in rendered.tiles:
                    count += 1
                    yield {"event": "tile", **self._tile_meta(tile.tile_num, tile.height, tile.image_hash, tile.jpeg)}

            await self.store.finish_doc(
                doc_id,
                tile_count=count,
                page_height=page_height,
                page_heights=page_heights_acc or None,
            )
            await self._invalidate(doc_id)
            metrics.DOCS.set(await self.store.count_docs())
            metrics.INGESTS.labels(doc_type=effective_type, stored="true", result="rendered").inc()

            yield {
                "event": "committed",
                "doc_id": doc_id,
                "doc_type": effective_type,
                "source_url": source_url,
                "render_version": self.settings.render_version,
                "page_height": page_height,
                "tile_count": count,
            }

        except ApiError as exc:
            log.warning("streaming ingest failed for %s: %s", doc_id, exc.message)
            yield {"event": "error", "code": exc.code, "message": exc.message}

        except Exception as exc:  # noqa: BLE001 - the stream is already 200; report in-band
            log.exception("streaming ingest crashed for %s", doc_id)
            yield {"event": "error", "code": "INTERNAL_ERROR", "message": repr(exc)}

        finally:
            self._inflight.discard(doc_id)


    async def _stream_from_store(self, row: DocRow) -> AsyncIterator[dict[str, Any]]:
        """Replay a fully-stored doc as the same NDJSON event stream (idempotent hit)."""
        yield {
            "event": "begin",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
        }

        for tile in await self.store.get_tiles(row.doc_id):
            jpeg, _ = await self.get_tile(row.doc_id, tile.tile_num, verify=True)
            yield {"event": "tile", **self._tile_meta(tile.tile_num, tile.height, tile.image_hash, jpeg)}

        yield {
            "event": "committed",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
            "page_height": row.page_height,
            "tile_count": row.tile_count,
        }


    async def revise(self, *, force: bool = False) -> dict[str, Any]:
        """Re-render stored docs whose render_version is stale (or incomplete), from their blobs.

        This is the tiler half of a revision: it refreshes tiles in place (doc_id is
        stable) so `get_tile` reproduces again after a render_version change. It does not
        touch Qdrant — virag's own /revision re-embeds afterwards.
        """
        current = self.settings.render_version
        revised = 0
        skipped = 0
        failed: list[str] = []
        cursor: str | None = None

        while True:
            rows, cursor = await self.store.search([], self.schema, limit=100, cursor=cursor)

            for row in rows:
                if not (force or not row.complete or row.render_version != current):
                    skipped += 1
                    continue

                try:
                    await self._revise_one(row, current)
                    revised += 1

                except ApiError as exc:
                    log.warning("revise failed for %s: %s", row.doc_id, exc.message)
                    failed.append(row.doc_id)

            if not cursor:
                break

        if revised:
            metrics.DOCS.set(await self.store.count_docs())

        log.info("revision: revised=%d skipped=%d failed=%d", revised, skipped, len(failed))
        return {"revised": revised, "skipped": skipped, "failed": failed, "render_version": current}


    async def _revise_one(self, row: DocRow, render_version: str) -> None:
        """Re-render one stored doc and atomically replace its tiles at the new render_version."""
        blob = await self._read_blob(row)
        rendered = await self.render_document(row.doc_type, blob, path="revise")

        await self.store.upsert_doc(
            doc_id=row.doc_id,
            doc_type=row.doc_type,
            blob_rel_path=row.blob_rel_path,
            source_url=row.source_url,
            page_height=rendered.page_height,
            tile_count=len(rendered.tiles),
            mime_type=row.mime_type,
            metadata=row.metadata,
            render_version=render_version,
            tiles=[(t.tile_num, t.height, t.image_hash) for t in rendered.tiles],
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


    async def _capture_url(self, url: str) -> bytes:
        """Fetch a URL through the SSRF guard and snapshot it as an MHTML blob."""
        s = self.settings

        try:
            target = await validate_url(
                url, allow_private=s.url_allow_private, enabled=s.url_fetch_enabled
            )
        except BlockedTarget as exc:
            metrics.SSRF_BLOCKS.inc()
            raise forbidden(str(exc), "SSRF_BLOCKED", url=url) from exc
        
        log.info("fetching %s (resolved to %s)", url, ",".join(target.ips))
        proxy = self.guard_proxy.url if self.guard_proxy is not None else None
        blob = await self._browser_call(web.capture_url_mhtml, self.browser, s, url, proxy)

        if self.guard_proxy is not None and self.guard_proxy.blocked:
            # Subresources may have been blocked; surface it without failing the capture.
            log.info("SSRF guard blocked %d subrequests", len(self.guard_proxy.blocked))
            self.guard_proxy.blocked.clear()

        return blob


    def validate_metadata_payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate metadata against the schema, raising 422 on any violation."""
        try:
            return validate_metadata(metadata, self.schema)
        except MetadataViolation as exc:
            raise unprocessable(str(exc), "METADATA_SCHEMA_VIOLATION", key=exc.key) from exc


    def _tile_meta(
        self, tile_num: int, height: int, image_hash: str, jpeg: bytes | None
    ) -> dict[str, Any]:
        """Build one tile's manifest entry, base64-encoding its bytes when present."""
        import base64  # noqa: PLC0415

        meta: dict[str, Any] = {
            "tile_num": tile_num,
            "width": self.settings.tile_width,  # always TILE_WIDTH
            "height": height,
            "image_hash": image_hash,
        }
        if jpeg is not None:
            meta["data"] = base64.b64encode(jpeg).decode("ascii")

        return meta


    async def _manifest_from_store(self, row: DocRow, include_bytes: bool) -> dict[str, Any]:
        """Rebuild an ingest manifest for an already-stored document."""
        tiles = await self.store.get_tiles(row.doc_id)
        out = []

        for t in tiles:
            jpeg = None

            if include_bytes:
                jpeg, _ = await self.get_tile(row.doc_id, t.tile_num, verify=True)

            out.append(self._tile_meta(t.tile_num, t.height, t.image_hash, jpeg))

        return {
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
            "stored": True,
            "page_height": row.page_height,
            "tile_count": row.tile_count,
            "tiles": out,
        }


    # ------------------------------------------------------------------ #
    # regeneration
    # ------------------------------------------------------------------ #
    async def get_tile(self, doc_id: str, tile_num: int, verify: bool) -> tuple[bytes, TileRow]:
        """Return a single tile's bytes, regenerating and caching it on demand."""
        row = await self.store.get_doc(doc_id)

        if row is None:
            raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)
        if not row.complete:
            raise not_found(
                f"doc_id {doc_id!r} is still being ingested", "DOC_INCOMPLETE", doc_id=doc_id
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
        before = self.tile_cache.hits
        jpeg = await self.tile_cache.get_or_create(key, lambda: self._regenerate(row, tile_num))
        outcome = "hit" if self.tile_cache.hits > before else "miss"
        metrics.CACHE.labels(cache="tile", outcome=outcome).inc()

        if verify:
            actual = sha256_hex(jpeg)
            if actual != tile_row.image_hash:
                metrics.MISMATCHES.labels(doc_type=row.doc_type).inc()
                message = (
                    f"regenerated tile ({doc_id}, {tile_num}) hashes to {actual} but "
                    f"{tile_row.image_hash} was stored at ingest"
                )
                if self.settings.render_mismatch_policy == "strict":
                    log.warning("%s — refusing (strict policy)", message)
                    raise conflict(message, "RENDER_MISMATCH", doc_id=doc_id, tile_num=tile_num)
                log.warning("%s — serving anyway (lenient policy)", message)
        return jpeg, tile_row


    async def _regenerate(self, row: DocRow, tile_num: int) -> bytes:
        """Re-render one tile from the stored blob, dispatched by doc type."""
        s = self.settings
        metrics.RENDERS.labels(doc_type=row.doc_type, path="regenerate").inc()
        metrics.TILES.labels(doc_type=row.doc_type, path="regenerate").inc()

        if row.doc_type == "PDF":
            blob = await self._read_blob(row)
            if row.page_heights:  # sliced PDF: reverse a global strip tile_num
                layout = slice_doc_strips(
                    row.page_heights, s.pdf_tile_height, s.pdf_tile_overlap, s.min_tile_height
                )
                page_no, strip = layout[tile_num]
                _, jpeg = await self.run_cpu(
                    cpu.crop_pdf_page_strip, blob, page_no, s.pdf_dpi, s.tile_width,
                    strip.offset, strip.height, s.jpeg_quality,
                )
                return jpeg
            out = await self.run_cpu(
                cpu.render_pdf_pages, blob, [tile_num], s.pdf_dpi, s.tile_width, s.jpeg_quality
            )
            return out[0][2]

        if row.doc_type == "IMG":
            blob = await self._read_blob(row)
            _, jpeg = await self.run_cpu(
                cpu.crop_one_tile,
                blob,
                s.tile_width,
                tile_num,
                row.page_height,
                s.tile_height,
                s.tile_overlap,
                s.jpeg_quality,
            )
            return jpeg

        # MHTML / URL
        if row.page_height > s.safe_fullpage_px:
            blob = await self._read_blob(row)
            strip = strip_for(tile_num, row.page_height, s.tile_height, s.tile_overlap)
            png = await self._browser_call(
                web.render_mhtml_region, self.browser, s, blob, strip
            )
            _, jpeg = await self.run_cpu(cpu.encode_strip, png, s.tile_width, s.jpeg_quality)
            return jpeg

        before = self.page_cache.hits
        png = await self.page_cache.get_or_create(row.doc_id, lambda: self._full_page(row))
        metrics.CACHE.labels(
            cache="page", outcome="hit" if self.page_cache.hits > before else "miss"
        ).inc()
        _, jpeg = await self.run_cpu(
            cpu.crop_one_tile,
            png,
            s.tile_width,
            tile_num,
            row.page_height,
            s.tile_height,
            s.tile_overlap,
            s.jpeg_quality,
        )
        return jpeg


    async def _full_page(self, row: DocRow) -> bytes:
        """Render a stored MHTML archive to one full-page PNG (cached per doc)."""
        blob = await self._read_blob(row)
        _, png = await self._browser_call(
            web.render_mhtml_full_png, self.browser, self.settings, blob
        )
        return png


    async def _read_blob(self, row: DocRow) -> bytes:
        """Read a document's original blob, mapping a missing file to 404."""
        try:
            return await self.blobs.read(row.doc_id)
        except FileNotFoundError as exc:
            raise not_found(
                f"blob for {row.doc_id!r} is missing from storage", "MISSING_BLOB"
            ) from exc


    async def _invalidate(self, doc_id: str) -> None:
        """Drop every cached tile and page belonging to one document."""
        await self.tile_cache.invalidate(lambda k: isinstance(k, tuple) and k[0] == doc_id)
        await self.page_cache.invalidate(lambda k: k == doc_id)


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
        """Probe the DB, blob store and browser for the readiness endpoint."""
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
        if self.settings.browser_enabled:
            checks["browser"] = "ok" if self.browser.available else (
                f"unavailable: {self.browser.start_error}"
            )
            if not self.browser.available:
                ok = False
        else:
            checks["browser"] = "disabled"
        return ok, checks


def summarize(rows: Sequence[DocRow]) -> list[dict[str, Any]]:
    """Project doc rows into the compact summaries returned by search/listing."""
    return [
        {
            "doc_id": r.doc_id,
            "doc_type": r.doc_type,
            "tile_count": r.tile_count,
            "complete": r.complete,
            "render_version": r.render_version,
            "created_at": r.created_at,
            "source_url": r.source_url,
            "metadata": r.metadata,
        }
        for r in rows
    ]


__all__ = ["TilerService", "RenderedTile", "RenderedDoc", "summarize", "ApiError"]
