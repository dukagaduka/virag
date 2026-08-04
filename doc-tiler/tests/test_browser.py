"""MHTML rendering: round-trip identity and re-render determinism.

Skipped automatically when the pinned Chromium is not installed.
"""

from __future__ import annotations

import hashlib

import pytest
from conftest import base_env, mhtml_form, running_app

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>t</title>
<style>
 body { margin:0; font-family: 'Liberation Serif', 'DejaVu Serif', serif; }
 .row { height: 180px; border-bottom: 2px solid #223; padding: 12px; background:#fbfbfb; }
 h1 { background:#123; color:#fff; margin:0; padding:20px; }
</style></head><body>
<h1>Tiler MHTML fixture</h1>
%s
</body></html>"""


def _html(rows: int) -> str:
    return HTML % "\n".join(f'<div class="row">row {i}</div>' for i in range(rows))


def chromium_available() -> bool:
    """Actually launch: the binary being present says nothing about its system libs."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
            browser.close()
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not chromium_available(),
    reason="Chromium cannot launch here (missing system libs) — run these in the container",
)


async def capture_fixture(svc, tmp_path, rows: int) -> bytes:
    """Snapshot a local HTML file to MHTML using the service's own browser pool."""
    from tiler.render.browser import capture_url_mhtml

    path = tmp_path / f"fixture_{rows}.html"
    path.write_text(_html(rows))
    return await capture_url_mhtml(svc.browser, svc.settings, f"file://{path}", None)


@pytest.fixture
def browser_env(tmp_path):
    return base_env(tmp_path, TILER_BROWSER_ENABLED="true", TILER_URL_FETCH_ENABLED="false")


async def test_mhtml_round_trip_identity(browser_env, tmp_path):
    async with running_app(browser_env) as (client, svc):
        blob = await capture_fixture(svc, tmp_path, rows=20)
        manifest = (await client.post("/tile", **mhtml_form(blob))).json()

        assert manifest["doc_type"] == "MHTML"
        assert manifest["page_height"] > 1024
        assert manifest["tile_count"] >= 3

        for tile in manifest["tiles"]:
            assert tile["width"] == 875
            got = await client.get(
                "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": tile["tile_num"]}
            )
            assert got.status_code == 200, got.text
            # regeneration reproduces the ingest bytes exactly.
            assert hashlib.sha256(got.content).hexdigest() == tile["image_hash"]


async def test_mhtml_render_is_deterministic(browser_env, tmp_path):
    """Same archive, same pinned Chromium → identical hashes."""
    async with running_app(browser_env) as (client, svc):
        blob = await capture_fixture(svc, tmp_path, rows=12)
        first = (await client.post("/tile", **mhtml_form(blob))).json()
        second = (
            await client.post("/tile", params={"force": "true"}, **mhtml_form(blob))
        ).json()
        assert [t["image_hash"] for t in first["tiles"]] == [
            t["image_hash"] for t in second["tiles"]
        ]
        assert first["page_height"] == second["page_height"]


async def test_sequential_strips_reuse_one_full_page_render(browser_env, tmp_path):
    """The page cache means N strips cost one browser render."""
    async with running_app(browser_env) as (client, svc):
        blob = await capture_fixture(svc, tmp_path, rows=20)
        manifest = (await client.post("/tile", **mhtml_form(blob))).json()
        await svc.tile_cache.clear()
        await svc.page_cache.clear()

        misses_before = svc.page_cache.misses
        for tile in manifest["tiles"]:
            await client.get(
                "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": tile["tile_num"]}
            )
        assert svc.page_cache.misses == misses_before + 1
        assert svc.page_cache.hits >= manifest["tile_count"] - 1


async def test_tall_page_uses_the_clip_path(tmp_path):
    """Force SAFE_FULLPAGE_PX below the fixture height so strips are clip-captured."""
    env = base_env(
        tmp_path,
        TILER_BROWSER_ENABLED="true",
        TILER_URL_FETCH_ENABLED="false",
        TILER_SAFE_FULLPAGE_PX="1024",
    )
    async with running_app(env) as (client, svc):
        blob = await capture_fixture(svc, tmp_path, rows=14)
        manifest = (await client.post("/tile", **mhtml_form(blob))).json()
        assert manifest["page_height"] > 1024
        assert manifest["tile_count"] >= 2

        for tile in manifest["tiles"]:
            got = await client.get(
                "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": tile["tile_num"]}
            )
            assert got.status_code == 200, got.text
            assert hashlib.sha256(got.content).hexdigest() == tile["image_hash"]


async def test_readyz_reports_the_browser(browser_env, tmp_path):
    async with running_app(browser_env) as (client, _):
        body = (await client.get("/readyz")).json()
        assert body["status"] == "ready"
        assert body["checks"]["browser"] == "ok"


async def test_url_ingest_end_to_end(tmp_path):
    """A real fetch through the SSRF guard proxy, against a local origin.

    TILER_URL_ALLOW_PRIVATE=true is required because the origin is on loopback; the proxy
    still mediates every connection, which is what the pinning path exercises.
    """
    import asyncio

    html = _html(6).encode()

    async def handler(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
            + f"Content-Length: {len(html)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + html
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    env = base_env(
        tmp_path,
        TILER_BROWSER_ENABLED="true",
        TILER_URL_FETCH_ENABLED="true",
        TILER_URL_ALLOW_PRIVATE="true",
    )
    try:
        async with running_app(env) as (client, svc):
            r = await client.post(
                "/tile", data={"doc_type": "URL", "url": f"http://127.0.0.1:{port}/page"}
            )
            assert r.status_code == 200, r.text
            manifest = r.json()
            assert manifest["doc_type"] == "MHTML"  # URL is stored as MHTML
            assert manifest["source_url"] == f"http://127.0.0.1:{port}/page"
            assert manifest["tile_count"] >= 1

            got = await client.get(
                "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": 0}
            )
            assert got.status_code == 200
            assert (
                hashlib.sha256(got.content).hexdigest() == manifest["tiles"][0]["image_hash"]
            )

            original = await client.get(f"/doc/{manifest['doc_id']}")
            assert original.headers["content-type"].startswith("multipart/related")
    finally:
        server.close()
