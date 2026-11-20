"""RenderPool: where rendering runs — the process pool of render workers and Chromium.

It only executes: a document (`documents.py`) hands it a task for the workers or takes
a Chromium slot, and the pool bounds the work (bulk slots, worker memory, task time)
and keeps itself alive. It knows nothing about document types.

The process pool is pebble's: a task that hangs or kills its worker costs that one
worker, which the pool replaces; the other tasks never notice. pebble gives the whole
pool up when a worker dies while no task runs (it takes that for a failing
initializer); the RenderPool then replaces the pool, so a stray kill of an idle worker
costs a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractAsyncContextManager, nullcontext
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pebble

from .. import metrics
from ..config import Settings
from ..errors import ApiError, bad_gateway, gateway_timeout, unprocessable
from ..store.cache import AsyncLRU
from . import render, worker
from .browser import BrowserPool, BrowserUnavailable, RenderFailure, RenderTimeout
from .readers import RenderParams

log = logging.getLogger("tiler.render.pool")


class RenderPool:
    """Owns the render process pool and the Chromium pool."""


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
        """A process pool whose workers cap their own memory and hold the render params."""
        # 'spawn': the parent runs an event loop and Playwright threads; forking those
        # into workers is a well-known source of deadlocks.
        return pebble.ProcessPool(
            max_workers=self.settings.render_workers,
            initializer=worker.init_worker,
            initargs=(self.settings.render_worker_memory, self.params),
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

            except render.EmptyDocument as exc:
                raise unprocessable(str(exc), "EMPTY_DOCUMENT") from exc

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
    # Chromium
    # ------------------------------------------------------------------ #
    def browser_slot(self, bulk: bool) -> AbstractAsyncContextManager:
        """What bulk work holds while it uses Chromium (interactive work holds nothing)."""
        return self._bulk_browser if bulk else nullcontext()


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
        async with self.browser_slot(bulk), self.browser_errors():
            return await fn(*args)


    async def forget(self, path: str | Path) -> None:
        """Drop the page capture cached for the document in the file at `path`."""
        key = str(path)

        await self.page_cache.invalidate(lambda k: k == key)
