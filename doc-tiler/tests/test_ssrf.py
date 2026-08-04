"""SSRF policy and the pinning guard proxy."""

from __future__ import annotations

import asyncio

import pytest
from conftest import base_env, running_app

from tiler.netguard import GuardProxy
from tiler.ssrf import BlockedTarget, ip_is_blocked, validate_url

BLOCKED_IPS = [
    "127.0.0.1",
    "127.10.20.30",
    "10.0.0.1",
    "172.16.5.4",
    "172.31.255.255",
    "192.168.1.1",
    "169.254.169.254",  # cloud metadata
    "0.0.0.0",
    "::1",
    "fc00::1",
    "fe80::1",
    "::ffff:127.0.0.1",  # v4-mapped loopback
    "::ffff:169.254.169.254",
    "224.0.0.1",
]

ALLOWED_IPS = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]


@pytest.mark.parametrize("ip", BLOCKED_IPS)
def test_blocked_ip_ranges(ip):
    assert ip_is_blocked(ip) is True


@pytest.mark.parametrize("ip", ALLOWED_IPS)
def test_public_ips_pass(ip):
    assert ip_is_blocked(ip) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/x",
        "http://169.254.169.254/latest/meta-data/",
        "https://10.1.2.3/",
        "http://[::1]:9000/",
        "http://192.168.0.5/admin",
    ],
)
async def test_private_urls_are_rejected(url):
    with pytest.raises(BlockedTarget):
        await validate_url(url, allow_private=False)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://example.com/x", "ws://example.com"])
async def test_non_http_schemes_are_rejected(url):
    with pytest.raises(BlockedTarget):
        await validate_url(url, allow_private=False)


async def test_allow_private_opens_the_gate():
    target = await validate_url("http://127.0.0.1:8000/x", allow_private=True)
    assert target.ips == ("127.0.0.1",)


async def test_disabled_url_fetching():
    with pytest.raises(BlockedTarget):
        await validate_url("https://example.com", allow_private=False, enabled=False)


async def test_url_ingest_returns_403(tmp_path):
    env = base_env(tmp_path, TILER_URL_FETCH_ENABLED="true", TILER_BROWSER_ENABLED="false")
    async with running_app(env) as (client, _):
        r = await client.post("/tile", data={"doc_type": "URL", "url": "http://169.254.169.254/"})
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "SSRF_BLOCKED"


async def test_url_without_url_field_is_400(tmp_path):
    async with running_app(base_env(tmp_path, TILER_URL_FETCH_ENABLED="true")) as (client, _):
        r = await client.post("/tile", data={"doc_type": "URL"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "MISSING_URL"


# --------------------------------------------------------------------------- #
# The guard proxy is what actually pins Chromium's egress.
# --------------------------------------------------------------------------- #
async def test_guard_proxy_blocks_connect_to_private_host():
    proxy = GuardProxy(allow_private=False)
    await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        writer.write(b"CONNECT 169.254.169.254:443 HTTP/1.1\r\nHost: 169.254.169.254\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(64), timeout=5)
        assert b"403" in response
        writer.close()
    finally:
        await proxy.stop()
    assert proxy.blocked


async def test_guard_proxy_blocks_plain_http_to_loopback():
    proxy = GuardProxy(allow_private=False)
    await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        writer.write(b"GET http://127.0.0.1:1/x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(64), timeout=5)
        assert b"403" in response
        writer.close()
    finally:
        await proxy.stop()


async def test_guard_proxy_forwards_allowed_http():
    """A public-looking origin is reachable through the proxy: run a real server and
    allow private targets so the loopback address is not rejected."""
    async def handler(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nhi")
        await writer.drain()
        writer.close()

    origin = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = origin.sockets[0].getsockname()[1]
    proxy = GuardProxy(allow_private=True)
    await proxy.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
        writer.write(
            f"GET http://127.0.0.1:{port}/x HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()
        )
        await writer.drain()
        body = await asyncio.wait_for(reader.read(1024), timeout=5)
        assert b"200 OK" in body and body.endswith(b"hi")
        writer.close()
    finally:
        await proxy.stop()
        origin.close()
