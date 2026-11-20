"""URL capture: fetch a page through the SSRF guard and snapshot it as MHTML.

The snapshot, not the live page, is what gets stored and rendered, so ingest and
a tile request share one source of truth.
"""

from __future__ import annotations

import logging

from . import metrics
from .config import Settings
from .errors import forbidden
from .netguard import GuardProxy
from .render import browser as web
from .render.pool import RenderPool
from .ssrf import BlockedTarget, validate_url

log = logging.getLogger("tiler.capture")


class UrlCapture:
    """Owns the SSRF guard proxy the browser fetches through."""


    def __init__(self, settings: Settings, pool: RenderPool):
        self.settings = settings
        self.pool = pool
        self.guard_proxy: GuardProxy | None = None


    @property
    def proxied(self) -> bool:
        """Whether captures go through the guard proxy (Chromium needs per-context proxies)."""
        return self.guard_proxy is not None


    async def start(self) -> None:
        """Start the guard proxy when URLs may be fetched and private targets may not."""
        if self.settings.url_fetch_enabled and not self.settings.url_allow_private:
            self.guard_proxy = GuardProxy(allow_private=False)
            await self.guard_proxy.start()


    async def stop(self) -> None:
        """Stop the guard proxy."""
        if self.guard_proxy is not None:
            await self.guard_proxy.stop()


    async def capture(self, url: str) -> bytes:
        """Validate `url` against the SSRF policy and snapshot it as an MHTML blob."""
        s = self.settings

        try:
            target = await validate_url(url, allow_private=s.url_allow_private, enabled=s.url_fetch_enabled)

        except BlockedTarget as exc:
            metrics.SSRF_BLOCKS.inc()
            raise forbidden(str(exc), "SSRF_BLOCKED", url=url) from exc

        log.info("fetching %s (resolved to %s)", url, ",".join(target.ips))
        proxy = self.guard_proxy.url if self.guard_proxy is not None else None
        blob = await self.pool.browser_call(
            web.capture_url_mhtml, self.pool.browser, s, url, proxy, bulk=True
        )

        if self.guard_proxy is not None and self.guard_proxy.blocked:
            # Subresources may have been blocked; surface it without failing the capture.
            log.info("SSRF guard blocked %d subrequests", len(self.guard_proxy.blocked))
            self.guard_proxy.blocked.clear()

        return blob
