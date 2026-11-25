"""Chromium: the browser pool and what it renders — URL captures and MHTML archives.

Playwright's async API drives Chromium over CDP on the event loop — no extra
threads. Concurrency is capped by a semaphore; a fresh BrowserContext per request
gives isolation; the browser process is recycled after `RECYCLE_AFTER` renders.

`BrowserPool` loads a document into a tab — a URL to snapshot it (`capture_url`), a
stored MHTML archive to render it (`open_mhtml`) — and a `Tab` is that loaded document:
its height, its capture whole or by rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import tempfile
from typing import Any, AsyncIterator

from ..config import RenderSettings
from ..metrics import BROWSER_RECYCLES

log = logging.getLogger("tiler.browser")


class BrowserUnavailable(RuntimeError):
    """Chromium could not be started or has died (maps to 502)."""


class RenderTimeout(RuntimeError):
    """Navigation or capture exceeded NAV_TIMEOUT_MS (maps to 504)."""


class RenderFailure(RuntimeError):
    """Navigation or capture failed (maps to 502)."""


class BrowserPool:
    """One long-lived Chromium, `N_BROWSERS` concurrent contexts, periodic recycle."""

    LAUNCH_ARGS = [
        "--hide-scrollbars",
        "--disable-gpu",
        "--force-color-profile=srgb",
        "--font-render-hinting=none",
        "--disable-lcd-text",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-ipc-flooding-protection",
        "--disable-dev-shm-usage",
        "--mute-audio",
    ]


    def __init__(self, settings: RenderSettings, *, viewport: tuple[int, int], temp_dir: str):
        self._s = settings
        self.width, self.height = viewport  # a tile's
        self.temp_dir = temp_dir            # where an archive gets the name Chromium opens it by
        self._pw: Any = None
        self._browser: Any = None
        self._sem = asyncio.Semaphore(settings.browsers)
        self._cond = asyncio.Condition()
        self._inflight = 0
        self._renders = 0
        self._recycling = False
        self._start_error: str | None = None
        self.per_context_proxy = False


    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        """True when a live, connected Chromium is present."""
        return self._browser is not None and self._browser.is_connected()


    @property
    def start_error(self) -> str | None:
        """The reason startup failed, or None if the browser is up."""
        return self._start_error


    async def start(self, *, per_context_proxy: bool = False) -> None:
        """Start Playwright and launch Chromium; record any failure instead of raising."""
        self.per_context_proxy = per_context_proxy

        try:
            # imported lazily: playwright is optional (PDF/IMG-only deployments run without it)
            from playwright.async_api import async_playwright  # noqa: PLC0415

            self._pw = await async_playwright().start()
            await self._launch()
            self._start_error = None

        except Exception as exc:
            self._start_error = f"{type(exc).__name__}: {exc}"
            log.error("Chromium unavailable: %s", self._start_error)

            with contextlib.suppress(Exception):
                if self._pw is not None:
                    await self._pw.stop()

            self._pw = None
            self._browser = None


    async def stop(self) -> None:
        """Close Chromium and stop Playwright, ignoring shutdown errors."""
        with contextlib.suppress(Exception):
            if self._browser is not None:
                await self._browser.close()

        with contextlib.suppress(Exception):
            if self._pw is not None:
                await self._pw.stop()

        self._browser = None
        self._pw = None


    # ------------------------------------------------------------------ #
    # documents
    # ------------------------------------------------------------------ #
    async def capture_url(self, url: str, proxy: str | None) -> bytes:
        """Navigate to `url` and snapshot it as MHTML — the archived blob we store.

        The blob, not the live page, is what later renders produce tiles from, so
        ingest and a tile request share one source of truth.
        """
        async with self._context(proxy=proxy) as ctx:
            tab = await self._load(ctx, url)

            return await tab.mhtml()


    @contextlib.asynccontextmanager
    async def open_mhtml(self, path: str) -> AsyncIterator[Tab]:
        """Load a stored MHTML archive in a fresh tab (Chromium reads MHTML natively).

        Chromium recognises the archive by its `.mhtml` suffix, which a content-addressed
        blob lacks, so it is opened through a hard link (a copy across file systems).
        """
        os.makedirs(self.temp_dir, exist_ok=True)
        fd, link = tempfile.mkstemp(suffix=".mhtml", dir=self.temp_dir)
        os.close(fd)
        os.unlink(link)

        try:
            try:
                os.link(path, link)

            except OSError:
                shutil.copyfile(path, link)

            async with self._context() as ctx:
                yield await self._load(ctx, f"file://{link}")

        finally:
            with contextlib.suppress(OSError):
                os.unlink(link)


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    async def _launch(self) -> None:
        """Launch a fresh Chromium process and reset the render counter."""
        kwargs: dict[str, Any] = {"headless": True, "args": list(self.LAUNCH_ARGS)}

        if self.per_context_proxy:
            # Chromium requires a browser-level proxy for per-context proxies to work.
            kwargs["proxy"] = {"server": "per-context"}

        self._browser = await self._pw.chromium.launch(**kwargs)
        self._renders = 0

        log.info("Chromium launched (version=%s)", self._browser.version)


    async def _recycle(self) -> None:
        """Close the current Chromium and launch a replacement."""
        log.info("recycling Chromium after %d renders", self._renders)

        BROWSER_RECYCLES.inc()

        with contextlib.suppress(Exception):
            if self._browser is not None:
                await self._browser.close()

        await self._launch()


    @contextlib.asynccontextmanager
    async def _context(self, *, proxy: str | None = None):
        """Lease a fresh BrowserContext; handles recycling and crash recovery."""
        if self._pw is None:
            raise BrowserUnavailable(self._start_error or "Chromium is not enabled")

        async with self._sem:
            async with self._cond:
                while self._recycling:
                    await self._cond.wait()

                needs_recycle = (
                    not self.available
                    or self._renders >= self._s.recycle_after
                )

                if needs_recycle:
                    self._recycling = True

                    try:
                        while self._inflight > 0:
                            await self._cond.wait()

                        await self._recycle()

                    finally:
                        self._recycling = False
                        self._cond.notify_all()

                self._renders += 1
                self._inflight += 1

            ctx = None

            try:
                opts: dict[str, Any] = {
                    "viewport": {"width": self.width, "height": self.height},
                    "device_scale_factor": 1,
                    "locale": "en-US",
                    "timezone_id": "UTC",
                    "color_scheme": "light",
                    "reduced_motion": "reduce",
                    "forced_colors": "none",
                    "java_script_enabled": True,
                }

                if proxy:
                    opts["proxy"] = {"server": proxy}

                ctx = await self._browser.new_context(**opts)
                ctx.set_default_timeout(self._s.nav_timeout_ms)
                ctx.set_default_navigation_timeout(self._s.nav_timeout_ms)

                yield ctx

            finally:
                if ctx is not None:
                    with contextlib.suppress(Exception):
                        await ctx.close()

                async with self._cond:
                    self._inflight -= 1
                    self._cond.notify_all()


    async def _load(self, ctx: Any, url: str) -> Tab:
        """`url` loaded and settled in a fresh tab of the context."""
        tab = Tab(await ctx.new_page(), self._s, self.width)

        try:
            await tab.page.goto(url, wait_until="load")
            await tab.settle()

        except Exception as exc:
            raise Tab.failure(exc) from exc

        return tab


class Tab:
    """A Chromium tab with a document loaded: what a capture reads from."""

    DETERMINISM_CSS = """
*, *::before, *::after {
  animation-play-state: paused !important;
  animation-delay: -0.0001s !important;
  animation-duration: 0.0001s !important;
  transition-duration: 0.0001s !important;
  transition-delay: 0s !important;
  caret-color: transparent !important;
}
html { scroll-behavior: auto !important; }
"""

    # Inject the determinism CSS by appending a <style> node ourselves. Playwright's
    # page.add_style_tag() waits for the injected element's load event, which never
    # fires on a static MHTML archive, so it stalls every MHTML/URL render until the
    # timeout. Building the node in-page returns immediately and applies identically.
    INJECT_CSS_JS = (
        "(css) => { const s = document.createElement('style'); s.textContent = css;"
        " (document.head || document.documentElement).appendChild(s); }"
    )


    def __init__(self, page: Any, settings: RenderSettings, width: int):
        self.page = page
        self.width = width  # of a tile
        self._s = settings


    async def settle(self) -> None:
        """Settle the page before a capture: await network idle, freeze animations, wait for fonts."""
        if self._s.wait_network_idle:
            with contextlib.suppress(Exception):
                await self.page.wait_for_load_state("networkidle")

        with contextlib.suppress(Exception):
            await self.page.evaluate(self.INJECT_CSS_JS, self.DETERMINISM_CSS)

        with contextlib.suppress(Exception):
            await self.page.evaluate("() => document.fonts.ready.then(() => true)")


    async def height(self) -> int:
        """Measure the full scroll height of the loaded page in CSS pixels."""
        try:
            return int(
                await self.page.evaluate(
                    "() => Math.ceil(Math.max("
                    "  document.documentElement.scrollHeight,"
                    "  document.body ? document.body.scrollHeight : 0,"
                    "  document.documentElement.offsetHeight))"
                )
            )

        except Exception as exc:
            raise self.failure(exc) from exc


    async def full_png(self) -> bytes:
        """Capture the whole loaded page as one PNG."""
        try:
            return await self.page.screenshot(
                full_page=True, type="png", animations="disabled", caret="hide", scale="css"
            )

        except Exception as exc:
            raise self.failure(exc) from exc


    async def clip(self, offset: int, height: int) -> bytes:
        """Screenshot rows `[offset, offset + height)` of the page as PNG bytes."""
        try:
            # Scroll the region into view first so lazy-loaded content rasterizes.
            await self.page.evaluate("y => window.scrollTo(0, y)", offset)
            await self.settle()

            # This is a viewport screenshot (not full_page — that would rasterize the whole
            # tall page and defeat the clip path), so the clip is viewport-relative. scrollY
            # can fall short of the requested offset at the bottom of the page, so subtract
            # where the page actually landed, not where we asked it to go.
            scroll_y = await self.page.evaluate("() => window.scrollY")

            return await self.page.screenshot(
                type="png",
                animations="disabled",
                caret="hide",
                scale="css",
                clip={"x": 0, "y": offset - scroll_y, "width": self.width, "height": height},
            )

        except Exception as exc:
            raise self.failure(exc) from exc


    async def mhtml(self) -> bytes:
        """Snapshot the loaded page as an MHTML archive."""
        try:
            cdp = await self.page.context.new_cdp_session(self.page)
            snapshot = await cdp.send("Page.captureSnapshot", {"format": "mhtml"})

        except Exception as exc:
            raise self.failure(exc) from exc

        data = snapshot.get("data")

        if not data:
            raise RenderFailure("Page.captureSnapshot returned no data")

        return data.encode("utf-8")


    @staticmethod
    def failure(exc: Exception) -> Exception:
        """Translate a Playwright error into a RenderTimeout or RenderFailure."""
        name = type(exc).__name__

        if "Timeout" in name or "timeout" in str(exc).lower():
            return RenderTimeout(str(exc))

        return RenderFailure(f"{name}: {exc}")
