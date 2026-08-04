"""The error envelope and the /health, /ready operational endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from virag.errors import ApiError, install_error_handlers


def test_api_error_body_minimal():
    err = ApiError(status_code=422, message="bad")
    assert err.code == "UNPROCESSABLE"
    assert err.body() == {"error": {"code": "UNPROCESSABLE", "message": "bad"}}


def test_api_error_body_with_details():
    err = ApiError(status_code=502, message="down", details={"error": "conn"})
    assert err.body() == {"error": {"code": "UPSTREAM_FAILURE", "message": "down", "details": {"error": "conn"}}}


def test_error_handler_renders_envelope():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom():
        raise ApiError(status_code=422, message="nope")

    @app.get("/crash")
    async def crash():
        raise RuntimeError("unexpected")

    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/boom")
    assert r.status_code == 422
    assert r.json() == {"error": {"code": "UNPROCESSABLE", "message": "nope"}}

    # an unexpected exception is masked as a generic 500 envelope (no stack trace leaked)
    r = client.get("/crash")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL_ERROR"


def test_health_always_ok(make_client):
    client = make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_503_when_backends_missing(make_client):
    client = make_client()                       # only settings set; no backend clients
    r = client.get("/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not ready"
    assert set(body["missing"]) == {"store", "embeddings", "tiler", "vlm"}


def test_ready_200_when_all_present(make_client):
    from conftest import FakeEmbeddings, FakeStore, FakeTiler, FakeVLM

    client = make_client(store=FakeStore(), embeddings=FakeEmbeddings(),
                         tiler=FakeTiler(), vlm=FakeVLM())
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}
