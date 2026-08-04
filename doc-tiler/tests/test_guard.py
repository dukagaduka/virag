"""The startup geometry guard."""

from __future__ import annotations

import pytest
from conftest import base_env, make_png, png_form, running_app


async def test_fresh_volume_records_geometry(tmp_path):
    async with running_app(base_env(tmp_path)) as (client, svc):
        meta = await svc.store.get_service_meta()
        assert (meta.tile_width, meta.tile_height, meta.tile_overlap, meta.min_tile_height) == (
            875,
            1024,
            0,
            28,
        )
        assert meta.render_version == svc.settings.render_version


async def test_same_geometry_reboots_cleanly(tmp_path):
    env = base_env(tmp_path)
    async with running_app(env) as (client, _):
        doc_id = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()["doc_id"]
    async with running_app(env) as (client, _):
        r = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})
        assert r.status_code == 200


@pytest.mark.parametrize(
    "changed", [{"TILER_TILE_HEIGHT": "512"}, {"TILER_TILE_WIDTH": "1024"},
                {"TILER_TILE_OVERLAP": "64"}, {"TILER_MIN_TILE_HEIGHT": "40"}]
)
async def test_geometry_change_is_fatal(tmp_path, changed):
    env = base_env(tmp_path)
    async with running_app(env) as (client, _):
        pass  # initialise the volume

    with pytest.raises(SystemExit) as exit_info:
        async with running_app(base_env(tmp_path, **changed)) as (_, _svc):
            pass
    assert exit_info.value.code == 1  # exits non-zero, serves nothing


async def test_pixel_tier_change_only_warns(tmp_path, caplog):
    env = base_env(tmp_path)
    async with running_app(env) as (client, _):
        pass
    with caplog.at_level("WARNING"):
        async with running_app(base_env(tmp_path, TILER_JPEG_QUALITY="60")) as (client, svc):
            assert (await client.get("/healthz")).status_code == 200
    assert any("Pixel params" in r.message for r in caplog.records)


async def test_invalid_config_is_fatal(tmp_path):
    with pytest.raises(SystemExit):
        async with running_app(base_env(tmp_path, TILER_TILE_OVERLAP="1024")) as (_, _svc):
            pass


def test_preflight_exits_non_zero_on_geometry_change(tmp_path):
    """The container runs this before uvicorn, so the failure is fast and unambiguous."""
    import os
    import subprocess
    import sys

    env = {**os.environ, **base_env(tmp_path)}
    first = subprocess.run([sys.executable, "-m", "tiler.preflight"], env=env, capture_output=True)
    assert first.returncode == 0, first.stderr.decode()

    env["TILER_TILE_HEIGHT"] = "900"
    second = subprocess.run([sys.executable, "-m", "tiler.preflight"], env=env, capture_output=True)
    assert second.returncode == 1
    assert b"Refusing to start" in second.stdout + second.stderr


async def test_render_version_covers_both_tiers():
    from tiler.config import Settings

    base = Settings()
    assert base.render_version != Settings(tile_height=512).render_version
    assert base.render_version != Settings(jpeg_quality=90).render_version
    assert base.render_version != Settings(pdf_dpi=150).render_version
    assert base.render_version == Settings().render_version  # stable
    assert base.render_version == Settings(max_doc_bytes=1).render_version  # not a tier
