"""RenderPool: where a reader's work runs — the CPU process pool or Chromium.

The service hands it a reader and the tiles it wants; the pool decides what runs where,
bounds it (bulk slots, worker memory, task time) and keeps itself alive. The process
pool is pebble's: a task that hangs or kills its worker costs that one worker, which
the pool replaces; the other tasks never notice. pebble gives the whole pool up when a
worker dies while no task runs (it takes that for a failing initializer); the
RenderPool then replaces the pool, so a stray kill of an idle worker costs a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import multiprocessing
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from typing import Any, AsyncIterator, Callable

import pebble

from .. import metrics
from ..config import Settings
from ..errors import ApiError, bad_gateway, gateway_timeout, unprocessable
from ..store.cache import AsyncLRU
from . import render, worker
from .browser import BrowserPool, BrowserUnavailable, RenderFailure, RenderTimeout
from .geometry import TileSpec
from .readers import CpuReader, DocumentReader, MhtmlReader, RenderParams
from .types import EncodedImage

log = logging.getLogger("tiler.render.pool")


class RenderPool:
    """Owns the render process pool and the Chromium pool."""

    MAX_BATCH = 16  # tiles per process-pool task: bounds the result size per hop


    def __init__(self, settings: Settings):
        self.settings = settings
        self.params = RenderParams.from_settings(settings)
        self.browser = BrowserPool(settings)
        # Full-page MHTML captures, per document: one render serves all of a doc's tiles.
        self.page_cache = AsyncLRU(max(2, settings.render_cache_size // 8), "page")
        self.worker_restarts = 0
        self._pool: pebble.ProcessPool | None = None
        self._rebuild_lock = asyncio.Lock()
        # Bulk work (ingest) holds at most the bulk slots, so get_tile from an
        # interactive caller always finds a free worker / browser.
        self._bulk_cpu = asyncio.Semaphore(settings.bulk_render_workers)
        self._bulk_browser = asyncio.Semaphore(settings.bulk_browsers)


    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self, *, browser: bool, per_context_proxy: bool) -> None:
        """Start the process pool and, when enabled, Chromium."""
        self._pool = self._new_pool()

        if browser:
            await self.browser.start(per_context_proxy=per_context_proxy)


    async def stop(self) -> None:
        """Tear down Chromium and the process pool."""
        await self.browser.stop()

        if self._pool is not None:
            pool, self._pool = self._pool, None
            await self._close(pool)


    @property
    def running(self) -> bool:
        """True while the process pool is up (workers that cannot start break it)."""
        return self._pool is not None and self._pool.active


    async def healthy(self) -> bool:
        """For readiness: a pool pebble gave up on is replaced first."""
        if self._pool is None:
            return False

        try:
            await self._live_pool()

        except ApiError:
            return False

        return self.running


    def _new_pool(self) -> pebble.ProcessPool:
        """A process pool whose workers cap their own memory."""
        # 'spawn': the parent runs an event loop and Playwright threads; forking those
        # into workers is a well-known source of deadlocks.
        return pebble.ProcessPool(
            max_workers=self.settings.render_workers,
            initializer=worker.init_worker,
            initargs=(self.settings.render_worker_memory,),
            context=multiprocessing.get_context("spawn"),
        )


    async def _live_pool(self) -> pebble.ProcessPool:
        """The process pool, replaced when pebble gave up on it."""
        pool = self._pool

        if pool is None:
            raise bad_gateway("the render pool is not running", "WORKER_CRASH")

        if not pool.active:
            await self._rebuild(pool)

        return self._pool


    async def _rebuild(self, broken: pebble.ProcessPool) -> None:
        """Replace a broken pool, once however many tasks report it, and prove the new
        one starts: workers that cannot start (WORKER_CRASH) leave it broken."""
        async with self._rebuild_lock:
            if self._pool is not broken:
                return

            log.warning("render pool broken (a worker died with no task running); restarting it")
            await self._close(broken)
            self._pool = self._new_pool()
            self.worker_restarts += 1
            metrics.WORKER_RESTARTS.labels(reason="pool_broken").inc()

            try:
                await asyncio.wait_for(asyncio.wrap_future(self._pool.schedule(abs, (0,))), 60)

            except (BrokenProcessPool, TimeoutError) as exc:
                log.critical("render workers cannot start: %s", exc)

                raise bad_gateway(f"render workers cannot start: {exc}", "WORKER_CRASH") from exc


    @staticmethod
    async def _close(pool: pebble.ProcessPool) -> None:
        """Stop a pool and reap its workers."""
        pool.stop()
        await asyncio.to_thread(pool.join, 10)


    # ------------------------------------------------------------------ #
    # CPU tasks
    # ------------------------------------------------------------------ #
    async def run(self, fn: Callable[..., Any], *args: Any, bulk: bool = False) -> Any:
        """Run `fn(*args)` in a render worker, mapping render failures to API errors."""
        async with (self._bulk_cpu if bulk else nullcontext()):
            metrics.INFLIGHT.labels(layer="cpu").inc()

            try:
                return await self._attempt(fn, args)

            except render.DecodeError as exc:
                raise unprocessable(str(exc), "DECODE_FAILED") from exc

            except MemoryError as exc:
                raise unprocessable(
                    f"the document needs more than TILER_RENDER_WORKER_MEMORY="
                    f"{self.settings.render_worker_memory} bytes to render ({exc})",
                    "RENDER_TOO_LARGE",
                ) from exc

            finally:
                metrics.INFLIGHT.labels(layer="cpu").dec()


    async def _attempt(self, fn: Callable[..., Any], args: tuple) -> Any:
        """Run the task; when its worker dies, once more.

        A worker can also die from the outside (the kernel's OOM killer, a stray kill),
        so one death does not convict the document; a second one does: RENDER_CRASHED.
        """
        try:
            return await self._submit(fn, args)

        except pebble.ProcessExpired as exc:
            self._replaced("worker_died", exc)

        try:
            return await self._submit(fn, args)

        except pebble.ProcessExpired as exc:
            self._replaced("worker_died", exc)

            raise unprocessable(
                "rendering this document crashes the render worker", "RENDER_CRASHED"
            ) from exc


    async def _submit(self, fn: Callable[..., Any], args: tuple) -> Any:
        """One task, at most TILER_RENDER_TIMEOUT_SEC of execution (queueing is free).
        Cancelling the caller stops the task: pebble kills its worker. A task the pool
        broke under runs again on its replacement."""
        timeout = self.settings.render_timeout_sec

        for attempt in range(2):
            pool = await self._live_pool()

            try:
                return await asyncio.wrap_future(pool.schedule(fn, args, timeout=timeout))

            except TimeoutError:
                self._replaced("timeout")

                raise gateway_timeout(
                    f"rendering took longer than TILER_RENDER_TIMEOUT_SEC={timeout}", "RENDER_TIMEOUT"
                ) from None

            except BrokenProcessPool as exc:
                if attempt:
                    raise bad_gateway(f"render workers cannot start: {exc}", "WORKER_CRASH") from exc


    def _replaced(self, reason: str, exc: BaseException | None = None) -> None:
        """Account for a worker pebble killed or lost and replaced."""
        self.worker_restarts += 1
        metrics.WORKER_RESTARTS.labels(reason=reason).inc()

        log.warning("render worker replaced (%s)%s", reason, f": {exc}" if exc else "")


    # ------------------------------------------------------------------ #
    # documents
    # ------------------------------------------------------------------ #
    @contextlib.asynccontextmanager
    async def open(
        self, reader: DocumentReader, *, bulk: bool = False, page_heights: list[int] | None = None
    ) -> AsyncIterator[Document]:
        """A render session over one document. `page_heights`, when already known (a
        stored doc), spares measuring them; `bulk` holds the session to bulk slots."""
        if isinstance(reader, MhtmlReader):
            reader.bind(self.browser, self.page_cache)

            async with (self._bulk_browser if bulk else nullcontext()):
                try:
                    yield BrowserDocument(self, reader, bulk, page_heights)

                finally:
                    await reader.close()

        elif isinstance(reader, CpuReader):
            yield CpuDocument(self, reader, bulk, page_heights)

        else:  # pragma: no cover - every reader is one of the two
            raise TypeError(f"no executor for {type(reader).__name__}")


    @contextlib.asynccontextmanager
    async def browser_errors(self) -> AsyncIterator[None]:
        """Map Chromium failures to API errors, counting the render as in flight."""
        metrics.INFLIGHT.labels(layer="browser").inc()

        try:
            yield

        except BrowserUnavailable as exc:
            raise bad_gateway(f"Chromium is unavailable: {exc}", "BROWSER_UNAVAILABLE") from exc

        except RenderTimeout as exc:
            raise gateway_timeout(f"render timed out: {exc}") from exc

        except RenderFailure as exc:
            raise bad_gateway(f"render failed: {exc}") from exc

        finally:
            metrics.INFLIGHT.labels(layer="browser").dec()


    async def browser_call(self, fn: Callable[..., Any], *args: Any, bulk: bool = False) -> Any:
        """Run one Chromium operation (e.g. a URL capture) under the browser limits."""
        async with (self._bulk_browser if bulk else nullcontext()):
            async with self.browser_errors():
                return await fn(*args)


class Document:
    """One document opened for rendering: its page heights and its tiles."""

    # How many batches may render at once.
    parallel: int = 1


    def __init__(self, pool: RenderPool, reader: DocumentReader, bulk: bool, page_heights: list[int] | None):
        self.pool = pool
        self.reader = reader
        self.bulk = bulk
        self._heights = page_heights


    async def page_heights(self) -> list[int]:
        """Height of every page at tile width (measured once)."""
        if self._heights is None:
            self._heights = await self._measure()

        return self._heights


    async def _measure(self) -> list[int]:
        raise NotImplementedError


    def batches(self, specs: list[TileSpec]) -> list[list[TileSpec]]:
        """How the tiles split into render tasks (in order)."""
        return [specs] if specs else []


    async def render(self, specs: list[TileSpec]) -> list[EncodedImage]:
        """Render and encode one batch of tiles, in order."""
        raise NotImplementedError


class CpuDocument(Document):
    """A PDF or an image: everything happens in the render workers."""

    reader: CpuReader


    def __init__(self, pool: RenderPool, reader: CpuReader, bulk: bool, page_heights: list[int] | None):
        super().__init__(pool, reader, bulk, page_heights)
        settings = pool.settings
        self.parallel = settings.bulk_render_workers if bulk else settings.render_workers


    async def _measure(self) -> list[int]:
        return await self.pool.run(worker.page_heights, self.reader, bulk=self.bulk)


    def batches(self, specs: list[TileSpec]) -> list[list[TileSpec]]:
        """Even chunks across the parallel slots, at most MAX_BATCH tiles each; an image
        stays one task (reading any tile decodes all of it)."""
        if not specs or not self.reader.splits_batches:
            return super().batches(specs)

        size = max(1, min(RenderPool.MAX_BATCH, math.ceil(len(specs) / max(1, self.parallel))))

        return [specs[i:i + size] for i in range(0, len(specs), size)]


    async def render(self, specs: list[TileSpec]) -> list[EncodedImage]:
        return await self.pool.run(worker.render_tiles, self.reader, specs, bulk=self.bulk)


class BrowserDocument(Document):
    """An MHTML archive: Chromium captures on the loop, the workers encode."""

    reader: MhtmlReader


    async def _measure(self) -> list[int]:
        async with self.pool.browser_errors():
            height = await self.reader.css_height()

            if not self.reader.full_page_mode(height):
                return [height]

            png = await self.reader.full_page()

        return [await self.pool.run(render.png_page_height, png, self.pool.params.tile_width, bulk=self.bulk)]


    def batches(self, specs: list[TileSpec]) -> list[list[TileSpec]]:
        """A whole-page capture encodes in one task; clip captures go one by one."""
        if self._heights is not None and not self.reader.full_page_mode(self._heights[0]):
            return [[spec] for spec in specs]

        return super().batches(specs)


    async def render(self, specs: list[TileSpec]) -> list[EncodedImage]:
        heights = await self.page_heights()
        p = self.pool.params

        if self.reader.full_page_mode(heights[0]):
            before = self.pool.page_cache.hits

            async with self.pool.browser_errors():
                png = await self.reader.full_page()

            outcome = "hit" if self.pool.page_cache.hits > before else "miss"
            metrics.CACHE.labels(cache="page", outcome=outcome).inc()

            rows = [(s.offset, s.offset + s.height) for s in specs]

            return await self.pool.run(render.crop_page_png, png, rows, p.tile_width, p.jpeg_quality, bulk=self.bulk)

        out = []

        for spec in specs:
            async with self.pool.browser_errors():
                png = await self.reader.clip(spec.offset, spec.height)

            out.append(await self.pool.run(render.encode_strip, png, p.tile_width, p.jpeg_quality, bulk=self.bulk))

        return out
