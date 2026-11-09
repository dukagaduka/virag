"""TILER_API_KEY: every route needs the bearer key but the probes and metrics."""

from __future__ import annotations

import pytest
from conftest import base_env, make_png, png_form, running_app
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

from tiler.errors import install_error_handlers

KEY = "s3cret-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def keyed_env(tmp_path):
    return base_env(tmp_path, TILER_API_KEY=KEY)


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/info"),
        ("GET", "/schema/metadata"),
        ("GET", "/revision"),
        ("GET", "/doc/" + "0" * 64),
        ("GET", "/get_tile?doc_id=" + "0" * 64 + "&tile_num=0"),
        ("POST", "/docs/search"),
    ],
)
async def test_keyed_routes_reject_a_missing_or_wrong_key(keyed_env, method, path):
    async with running_app(keyed_env) as (client, _):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}):
            r = await client.request(method, path, headers=headers)
            assert r.status_code == 401, (path, headers)
            assert r.json()["error"]["code"] == "UNAUTHORIZED"
            assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("path", ["/healthz", "/readyz", "/metrics"])
async def test_probes_and_metrics_stay_open(keyed_env, path):
    async with running_app(keyed_env) as (client, _):
        assert (await client.get(path)).status_code == 200


async def test_the_right_key_passes(keyed_env):
    async with running_app(keyed_env) as (client, _):
        assert (await client.post("/tile", **png_form(make_png(875, 1200)))).status_code == 401

        r = await client.post("/tile", headers=AUTH, **png_form(make_png(875, 1200)))
        assert r.status_code == 200
        r = await client.get("/get_tile", headers=AUTH, params={"doc_id": r.json()["doc_id"], "tile_num": 0})
        assert r.status_code == 200


async def test_empty_key_leaves_the_routes_open(tmp_path, caplog):
    async with running_app(base_env(tmp_path)) as (client, _):
        assert (await client.get("/info")).status_code == 200
    assert any("TILER_API_KEY is empty" in r.message for r in caplog.records)


async def test_unhandled_error_hides_its_text_and_logs_it(caplog):
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("/data/blobs/private-path")

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://tiler") as client:
        r = await client.get("/boom")

    assert r.status_code == 500
    assert r.json()["error"] == {"code": "INTERNAL_ERROR", "message": "Internal error; see the tiler log."}
    assert "private-path" not in r.text
    assert any(rec.exc_info and "private-path" in str(rec.exc_info[1]) for rec in caplog.records)
