"""The event loop stays responsive and identical work coalesces."""

from __future__ import annotations

import asyncio
import time

from conftest import make_pdf, make_png, pdf_form, png_form

from tiler.store.cache import AsyncLRU


async def test_healthz_stays_fast_during_concurrent_ingests(client):
    """CPU work lives in the process pool, so liveness must not queue behind it."""
    docs = [make_pdf(pages=4, height=842 + i) for i in range(6)]
    latencies: list[float] = []
    stop = asyncio.Event()

    async def poll() -> None:
        while not stop.is_set():
            started = time.perf_counter()
            r = await client.get("/healthz")
            latencies.append(time.perf_counter() - started)
            assert r.status_code == 200
            await asyncio.sleep(0.005)

    poller = asyncio.create_task(poll())
    results = await asyncio.gather(*(client.post("/tile", **pdf_form(d)) for d in docs))
    stop.set()
    await poller

    assert all(r.status_code == 200 for r in results)
    assert len({r.json()["doc_id"] for r in results}) == 6
    assert len(latencies) > 5
    worst = max(latencies)
    assert worst < 0.05, f"/healthz worst-case {worst * 1000:.1f}ms exceeds the 50ms budget"


async def test_concurrent_tile_requests_render_once(client):
    manifest = (await client.post("/tile", **png_form(make_png(875, 3000)))).json()
    await client.service.tile_cache.clear()
    params = {"doc_id": manifest["doc_id"], "tile_num": 1}

    misses_before = client.service.tile_cache.misses
    responses = await asyncio.gather(*(client.get("/get_tile", params=params) for _ in range(8)))

    assert all(r.status_code == 200 for r in responses)
    assert len({r.content for r in responses}) == 1
    # One miss, the rest coalesced onto the same in-flight render.
    assert client.service.tile_cache.misses == misses_before + 1
    assert client.service.tile_cache.coalesced >= 1


async def test_lru_evicts_and_coalesces():
    cache = AsyncLRU(capacity=2)
    calls = {"n": 0}

    async def slow(value: str) -> str:
        calls["n"] += 1
        await asyncio.sleep(0.02)
        return value

    results = await asyncio.gather(*(cache.get_or_create("k", lambda: slow("v")) for _ in range(5)))
    assert results == ["v"] * 5
    assert calls["n"] == 1

    await cache.get_or_create("a", lambda: slow("a"))
    await cache.get_or_create("b", lambda: slow("b"))
    assert cache.peek("k") is None  # capacity 2: the oldest key was evicted
    assert cache.peek("b") == "b"


async def test_failed_render_is_not_cached():
    cache = AsyncLRU(capacity=4)
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    results = await asyncio.gather(
        *(cache.get_or_create("k", flaky) for _ in range(3)), return_exceptions=True
    )
    # The first producer fails; waiters retry rather than inheriting a poisoned entry.
    assert "ok" in results
    assert cache.peek("k") == "ok"


async def test_metrics_endpoint_exposes_counters(client):
    await client.post("/tile", **png_form(make_png(875, 1100)))
    body = (await client.get("/metrics")).text
    assert "tiler_renders_total" in body
    assert "tiler_docs" in body
    assert "tiler_cache_total" in body


async def test_info_endpoint(client):
    body = (await client.get("/info")).json()
    assert body["geometry"] == {
        "tile_width": 875,
        "tile_height": 1024,
        "tile_overlap": 0,
        "min_tile_height": 28,
    }
    assert body["render_version"]
    assert body["pools"]["render_workers"] == 2
