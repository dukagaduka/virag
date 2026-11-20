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
    await client.state.tiles.cache.clear()
    params = {"doc_id": manifest["doc_id"], "tile_num": 1}

    misses_before = client.state.tiles.cache.misses
    responses = await asyncio.gather(*(client.get("/get_tile", params=params) for _ in range(8)))

    assert all(r.status_code == 200 for r in responses)
    assert len({r.content for r in responses}) == 1
    # One miss, the rest coalesced onto the same in-flight render.
    assert client.state.tiles.cache.misses == misses_before + 1
    assert client.state.tiles.cache.coalesced >= 1


async def test_reads_never_see_a_commit_half_way(client, sample_pdf):
    """A read made while a commit is between its statements sees the doc as it was
    committed last — not the transaction's own view, with the old tiles already gone."""
    store = client.state.store
    doc_id = (await client.post("/tile", **pdf_form(sample_pdf))).json()["doc_id"]
    before = await store.get_tiles(doc_id)
    assert len(before) == 3

    replace = store._replace_tiles
    inside, release = asyncio.Event(), asyncio.Event()

    async def held(doc_id, tiles):
        await replace(doc_id, ())                                    # the old tiles are deleted…
        inside.set()
        await release.wait()                                         # …and the new ones not written yet
        await replace(doc_id, tiles)

    store._replace_tiles = held
    commit = asyncio.create_task(
        store.replace_render(doc_id, height=7, tile_count=1, render_version="rv-next", tiles=before[:1])
    )
    await inside.wait()

    try:
        assert await store.get_tiles(doc_id) == before
        assert ((await store.get_doc(doc_id)).tile_count, (await store.get_doc(doc_id)).height) != (1, 7)
        got = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 2})
        assert got.status_code == 200, got.text

    finally:
        release.set()
        await commit

    assert await store.get_tiles(doc_id) == before[:1]               # committed: visible at once
    assert (await store.get_doc(doc_id)).render_version == "rv-next"


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
