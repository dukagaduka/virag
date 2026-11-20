"""RenderPool: task timeouts, dead workers and the per-worker memory cap."""

from __future__ import annotations

import asyncio
import os
import signal
import struct
import time
import zlib
from pathlib import Path

import pytest

from conftest import base_env, png_form, running_app, make_png

from tiler.config import Settings
from tiler.errors import ApiError
from tiler.models import DocType
from tiler.render import worker
from tiler.render.geometry import Tile, Tiling
from tiler.render.pool import RenderPool
from tiler.render.readers import RenderParams


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {"n_render_workers": 2, "browser_enabled": False, **overrides}

    return Settings(db_path=str(tmp_path / "meta.sqlite"), blob_dir=str(tmp_path / "blobs"), **values)


async def _started(settings: Settings) -> RenderPool:
    pool = RenderPool(settings)
    await pool.start(browser=False, per_context_proxy=False)
    return pool


def _blank_png(path: Path, side: int) -> None:
    """A side×side grayscale PNG written without ever holding its pixels."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    packer = zlib.compressobj(9)
    row = b"\x00" * (side + 1)  # filter byte + pixels
    body = b"".join(packer.compress(row) for _ in range(side)) + packer.flush()
    header = struct.pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", body) + chunk(b"IEND", b""))


async def test_a_hung_task_times_out_alone(tmp_path):
    """Only the hung task's worker is killed: a task running beside it finishes, and a
    task that waited in the queue gets its full timeout once it starts."""
    pool = await _started(_settings(tmp_path, render_timeout_sec=2))

    try:
        hung = asyncio.create_task(pool.run(time.sleep, 30))
        bystander = asyncio.create_task(pool.run(time.sleep, 1.5))
        queued = asyncio.create_task(pool.run(time.sleep, 1.5))    # waits for a worker

        with pytest.raises(ApiError) as caught:
            await hung

        assert caught.value.status_code == 504
        assert caught.value.code == "RENDER_TIMEOUT"
        assert await bystander is None
        assert await queued is None                                  # 3 s wall, 1.5 s of run
        assert pool.worker_restarts == 1
        assert pool.running

    finally:
        await pool.stop()


async def test_a_worker_death_fails_only_its_task(tmp_path):
    pool = await _started(_settings(tmp_path))

    try:
        bystander = asyncio.create_task(pool.run(time.sleep, 1.5))
        await asyncio.sleep(0.5)                                     # let it start running

        with pytest.raises(ApiError) as caught:
            await pool.run(os._exit, 1)                              # dies, and dies again

        assert caught.value.code == "RENDER_CRASHED"
        assert caught.value.status_code == 422
        assert await bystander is None                               # never noticed
        assert pool.worker_restarts == 2
        assert await pool.run(abs, -1) == 1

    finally:
        await pool.stop()


async def test_an_idle_worker_killed_from_outside_costs_a_pool_restart(tmp_path):
    """pebble gives the pool up when a worker dies with no task running; the RenderPool
    replaces it instead of staying down."""
    pool = await _started(_settings(tmp_path))

    try:
        os.kill(await pool.run(os.getpid), signal.SIGKILL)

        for _ in range(50):                                          # pebble notices
            if not pool.running:
                break

            await asyncio.sleep(0.1)

        assert not pool.running
        assert await pool.healthy()                                  # /readyz heals it
        assert await pool.run(abs, -4) == 4
        assert pool.worker_restarts == 1

    finally:
        await pool.stop()


async def test_a_cancelled_task_stops_its_worker(tmp_path):
    pool = await _started(_settings(tmp_path, n_render_workers=1))

    try:
        task = asyncio.create_task(pool.run(time.sleep, 30))
        await asyncio.sleep(1)                                       # running by now
        task.cancel()

        started = time.perf_counter()
        assert await pool.run(abs, -2) == 2                          # the only worker is free again
        assert time.perf_counter() - started < 15

    finally:
        await pool.stop()


async def test_workers_that_cannot_start_report_worker_crash(tmp_path):
    pool = await _started(_settings(tmp_path, render_worker_memory=1 << 70))   # setrlimit rejects it

    try:
        with pytest.raises(ApiError) as caught:
            await pool.run(abs, -1)

        assert caught.value.code == "WORKER_CRASH"
        assert not pool.running

    finally:
        await pool.stop()


async def test_a_document_over_the_worker_memory_fails_alone(tmp_path):
    big = tmp_path / "big.png"
    _blank_png(big, 20_000)                                          # 400 MB once decoded
    pool = await _started(_settings(tmp_path, render_worker_memory=256 << 20))

    try:
        with pytest.raises(ApiError) as caught:
            await pool.run(worker.plan, DocType.IMG, str(big))

        assert caught.value.status_code == 422
        assert caught.value.code == "RENDER_TOO_LARGE"
        assert pool.worker_restarts == 0                             # the worker survived it

        small = tmp_path / "small.png"
        small.write_bytes(make_png(400, 300))
        assert await pool.run(worker.plan, DocType.IMG, str(small)) == (656, [Tile(0, 0, 656)])

    finally:
        await pool.stop()


async def test_ingest_reports_a_too_large_document_in_band(tmp_path):
    big = tmp_path / "big.png"
    _blank_png(big, 20_000)
    env = base_env(tmp_path, TILER_RENDER_WORKER_MEMORY=str(256 << 20))

    async with running_app(env) as (client, state):
        r = await client.post("/tile", **png_form(big.read_bytes()))
        assert r.status_code == 422, r.text
        assert r.json()["error"]["code"] == "RENDER_TOO_LARGE"

        ok = await client.post("/tile", **png_form(make_png(400, 300)))
        assert ok.status_code == 200, ok.text

        ready = (await client.get("/readyz")).json()
        assert ready["checks"]["render_pool"] == "ok"


def test_render_params_are_the_pixel_settings_and_the_tilings():
    settings = Settings(
        tile_width=900, jpeg_quality=70, pdf_dpi=150, tile_overlap=10, pdf_slice=True, pdf_tile_height=500
    )
    params = RenderParams.from_settings(settings)

    assert (params.tile_width, params.jpeg_quality, params.pdf_dpi) == (900, 70, 150)
    assert params.tiling == Tiling(1024, 10, 28)
    assert params.pdf_tiling == Tiling(500, 0, 28)
    assert RenderParams.from_settings(Settings()).pdf_tiling.whole   # unsliced: a page is a tile
