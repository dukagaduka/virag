"""SSRF policy for URL input: what a URL ingest may reach.

Only http(s), and — unless private targets are allowed — only hosts whose every address
is globally routable. The policy checks the URL itself (`check_url`) before the browser
is ever pointed at it, and resolves every host the browser then asks the guard proxy for
(`resolve`), so a redirect or a subresource is held to the same rules.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from .config import CaptureSettings


class BlockedTarget(Exception):
    """Target rejected by the SSRF policy (maps to 403)."""


@dataclass(frozen=True)
class Target:
    """A validated URL target: scheme, host, port, and resolved IPs."""

    scheme: str
    host: str
    port: int
    ips: tuple[str, ...]


class SsrfPolicy:
    """Whether URLs are fetched at all, and whether a private address may be a target."""

    ALLOWED_SCHEMES = {"http", "https"}
    DEFAULT_PORTS = {"http": 80, "https": 443}
    NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")  # RFC 6052 well-known prefix


    def __init__(self, *, enabled: bool = True, allow_private: bool = False):
        self.enabled = enabled
        self.allow_private = allow_private


    @classmethod
    def from_settings(cls, settings: CaptureSettings) -> SsrfPolicy:
        return cls(enabled=settings.url_fetch_enabled, allow_private=settings.url_allow_private)


    async def check_url(self, url: str) -> Target:
        """Validate a URL against the policy and return its resolved Target."""
        if not self.enabled:
            raise BlockedTarget("URL fetching is disabled (TILER_URL_FETCH_ENABLED=false)")

        parts = urlsplit(url)

        if parts.scheme not in self.ALLOWED_SCHEMES:
            raise BlockedTarget(f"scheme {parts.scheme!r} is not allowed (http/https only)")

        if not parts.hostname:
            raise BlockedTarget("URL has no host")

        host = parts.hostname

        try:
            port = parts.port or self.DEFAULT_PORTS[parts.scheme]

        except ValueError as exc:
            raise BlockedTarget(f"invalid port in URL: {exc}") from exc

        # A literal IP needs no DNS but the same policy.
        try:
            ipaddress.ip_address(host)

        except ValueError:
            ips = await self.resolve(host, port)
        else:
            if not self.allow_private and self.ip_is_blocked(host):
                raise BlockedTarget(f"address {host!r} is disallowed")

            ips = (host,)

        return Target(scheme=parts.scheme, host=host, port=port, ips=ips)


    async def resolve(self, host: str, port: int) -> tuple[str, ...]:
        """Resolve `host`, rejecting the whole target if *any* answer is disallowed."""
        ips = tuple(await asyncio.to_thread(self._lookup, host, port))

        if self.allow_private:
            return ips

        blocked = [ip for ip in ips if self.ip_is_blocked(ip)]

        if blocked:
            raise BlockedTarget(
                f"host {host!r} resolves to a disallowed address: {', '.join(blocked)}"
            )

        return ips


    @classmethod
    def ip_is_blocked(cls, raw: str) -> bool:
        """Reject every destination that is not globally routable (loopback, private,
        CGNAT, link-local, reserved, …) and multicast."""
        try:
            ip = ipaddress.ip_address(raw)

        except ValueError:
            return True

        if isinstance(ip, ipaddress.IPv6Address):
            # Unwrap v4-mapped, 6to4 and NAT64 forms so ::ffff:127.0.0.1 or
            # 64:ff9b::a9fe:a9fe cannot slip through as "global" IPv6.
            if ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            elif ip.sixtofour is not None:
                ip = ip.sixtofour
            elif ip in cls.NAT64_PREFIX:
                ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)

        return not ip.is_global or ip.is_multicast


    @staticmethod
    def _lookup(host: str, port: int) -> list[str]:
        """Resolve `host` to a deduplicated list of IPs, or raise BlockedTarget."""
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)

        except socket.gaierror as exc:
            raise BlockedTarget(f"cannot resolve host {host!r}: {exc}") from exc

        ips: list[str] = []

        for info in infos:
            addr = info[4][0]

            if addr not in ips:
                ips.append(addr)

        if not ips:
            raise BlockedTarget(f"host {host!r} resolved to no addresses")

        return ips
