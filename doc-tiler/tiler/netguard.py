"""Loopback HTTP proxy that pins Chromium's egress to SSRF-validated IPs.

Chromium resolves DNS itself, so validating a URL before navigation cannot stop
DNS rebinding, redirects to internal hosts, or a subresource pointing at
169.254.169.254. Routing the browser context through this in-process proxy moves
resolution to us: every connection Chromium makes — top-level, redirect, or
subresource — is resolved here, checked against the policy, and dialled by IP.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable

from .ssrf import BlockedTarget, resolve_and_check

log = logging.getLogger("tiler.netguard")

_MAX_HEADER_BYTES = 64 * 1024
_HOP_BY_HOP = {
    b"proxy-connection",
    b"proxy-authorization",
    b"connection",
    b"keep-alive",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


class GuardProxy:
    """A one-request-per-connection forward proxy bound to 127.0.0.1."""

    def __init__(self, *, allow_private: bool = False, connect_timeout: float = 15.0):
        self._allow_private = allow_private
        self._connect_timeout = connect_timeout
        self._server: asyncio.AbstractServer | None = None
        self.port: int = 0
        self.blocked: list[str] = []

    @property
    def url(self) -> str:
        """The proxy's `http://127.0.0.1:<port>` address."""
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        """Bind the proxy to an ephemeral loopback port and start serving."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("SSRF guard proxy listening on %s", self.url)

    async def stop(self) -> None:
        """Close the listening socket and wait for it to shut down."""
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # pragma: no cover - shutdown races
                pass
            self._server = None

    # ------------------------------------------------------------------ #
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Parse one request line and dispatch to CONNECT or plain forwarding."""
        try:
            head = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=self._connect_timeout
            )
        except Exception:
            await _close(writer)
            return

        if len(head) > _MAX_HEADER_BYTES:
            await _respond(writer, 431, "Request Header Fields Too Large")
            return

        lines = head.split(b"\r\n")
        try:
            method, target, version = lines[0].split(b" ", 2)
        except ValueError:
            await _respond(writer, 400, "Bad Request")
            return

        try:
            if method.upper() == b"CONNECT":
                await self._do_connect(target.decode("latin-1"), reader, writer)
            else:
                await self._do_plain(method, target, version, lines[1:], reader, writer)
        except BlockedTarget as exc:
            self.blocked.append(str(exc))
            log.warning("SSRF guard blocked a browser request: %s", exc)
            await _respond(writer, 403, "Blocked by SSRF policy")
        except asyncio.TimeoutError:
            await _respond(writer, 504, "Upstream timeout")
        except Exception as exc:  # pragma: no cover - transport noise
            log.debug("proxy connection failed: %s", exc)
            await _close(writer)

    async def _dial(self, host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Resolve+validate `host`, then connect by IP (defeats DNS rebinding)."""
        ips = await resolve_and_check(host, port, self._allow_private)
        last: Exception | None = None
        for ip in ips:
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(ip, port), timeout=self._connect_timeout
                )
            except Exception as exc:  # try the next validated address
                last = exc
        raise BlockedTarget(f"cannot connect to {host}:{port}: {last}")

    async def _do_connect(
        self, authority: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a CONNECT tunnel (HTTPS): validate the target, then splice."""
        host, _, raw_port = authority.rpartition(":")
        if not host:
            host, raw_port = authority, "443"
        host = host.strip("[]")
        try:
            port = int(raw_port)
        except ValueError:
            raise BlockedTarget(f"invalid CONNECT authority {authority!r}") from None

        up_reader, up_writer = await self._dial(host, port)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await _pipe_both(reader, writer, up_reader, up_writer)

    async def _do_plain(
        self,
        method: bytes,
        target: bytes,
        version: bytes,
        header_lines: Iterable[bytes],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Forward a plain (non-CONNECT) http request to a validated upstream."""
        uri = target.decode("latin-1")
        if not uri.lower().startswith("http://"):
            raise BlockedTarget(f"proxy only forwards absolute http URIs, got {uri!r}")
        rest = uri[len("http://") :]
        authority, slash, path = rest.partition("/")
        path = f"/{path}" if slash else "/"
        host, _, raw_port = authority.rpartition(":")
        if not host or not raw_port.isdigit():
            host, port = authority, 80
        else:
            port = int(raw_port)
        host = host.strip("[]")

        up_reader, up_writer = await self._dial(host, port)
        out = [b" ".join([method, path.encode("latin-1"), version])]
        for line in header_lines:
            if not line:
                continue
            name = line.split(b":", 1)[0].strip().lower()
            if name in _HOP_BY_HOP:
                continue
            out.append(line)
        out.append(b"Connection: close")
        up_writer.write(b"\r\n".join(out) + b"\r\n\r\n")
        await up_writer.drain()
        await _pipe_both(reader, writer, up_reader, up_writer)


async def _pipe_both(
    c_reader: asyncio.StreamReader,
    c_writer: asyncio.StreamWriter,
    u_reader: asyncio.StreamReader,
    u_writer: asyncio.StreamWriter,
) -> None:
    """Splice client and upstream in both directions until either closes."""
    await asyncio.gather(
        _pump(c_reader, u_writer), _pump(u_reader, c_writer), return_exceptions=True
    )
    await asyncio.gather(_close(c_writer), _close(u_writer), return_exceptions=True)


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy bytes from `reader` to `writer` until EOF."""
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.write_eof()
        except Exception:
            pass


async def _respond(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
    """Send a minimal plaintext HTTP response and close the connection."""
    body = reason.encode()
    try:
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n"
            f"Content-Type: text/plain\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
    except Exception:
        pass
    await _close(writer)


async def _close(writer: asyncio.StreamWriter) -> None:
    """Close a writer, swallowing shutdown-race errors."""
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
