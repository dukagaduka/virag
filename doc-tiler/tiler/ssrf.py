"""SSRF protection for URL input: scheme/host/IP policy checks."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_PORTS = {"http": 80, "https": 443}


class BlockedTarget(Exception):
    """Target rejected by the SSRF policy (maps to 403)."""


@dataclass(frozen=True)
class Target:
    """A validated URL target: scheme, host, port, and resolved IPs."""

    scheme: str
    host: str
    port: int
    ips: tuple[str, ...]


def ip_is_blocked(raw: str) -> bool:
    """Reject loopback / private / link-local / reserved / multicast destinations."""
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        # Unwrap v4-mapped/compatible forms so ::ffff:127.0.0.1 cannot slip through.
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif getattr(ip, "sixtofour", None) is not None:
            ip = ip.sixtofour
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv6Address) and ip.is_site_local)
    )


def _resolve(host: str, port: int) -> list[str]:
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


async def resolve_and_check(host: str, port: int, allow_private: bool) -> tuple[str, ...]:
    """Resolve `host`, rejecting the whole target if *any* answer is disallowed."""
    ips = tuple(await asyncio.to_thread(_resolve, host, port))
    if allow_private:
        return ips
    blocked = [ip for ip in ips if ip_is_blocked(ip)]
    if blocked:
        raise BlockedTarget(
            f"host {host!r} resolves to a disallowed address: {', '.join(blocked)}"
        )
    return ips


async def validate_url(url: str, *, allow_private: bool, enabled: bool = True) -> Target:
    """Validate a URL against the SSRF policy and return its resolved Target."""
    if not enabled:
        raise BlockedTarget("URL fetching is disabled (TILER_URL_FETCH_ENABLED=false)")
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise BlockedTarget(f"scheme {parts.scheme!r} is not allowed (http/https only)")
    if not parts.hostname:
        raise BlockedTarget("URL has no host")
    host = parts.hostname
    try:
        port = parts.port or DEFAULT_PORTS[parts.scheme]
    except ValueError as exc:
        raise BlockedTarget(f"invalid port in URL: {exc}") from exc

    # A literal IP needs no DNS but the same policy.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        ips = await resolve_and_check(host, port, allow_private)
    else:
        if not allow_private and ip_is_blocked(host):
            raise BlockedTarget(f"address {host!r} is disallowed")
        ips = (host,)
    return Target(scheme=parts.scheme, host=host, port=port, ips=ips)
