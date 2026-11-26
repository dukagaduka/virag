"""The error envelope and the /health, /ready operational endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from virag import errors
from virag.errors import ApiError, install_error_handlers


def test_api_error_body_minimal():
    err = ApiError(status_code=422, message="bad")
    assert err.code == "UNPROCESSABLE"
    assert err.body() == {"error": {"code": "UNPROCESSABLE", "message": "bad"}}


def test_api_error_body_with_details():
    err = ApiError(status_code=502, message="down", details={"error": "conn"})
    assert err.body() == {"error": {"code": "UPSTREAM_FAILURE", "message": "down", "details": {"error": "conn"}}}


@pytest.mark.parametrize("factory, status, code", [
    (errors.unauthorized, 401, "UNAUTHORIZED"),
    (errors.forbidden, 403, "FORBIDDEN"),
    (errors.not_found, 404, "NOT_FOUND"),
    (errors.conflict, 409, "CONFLICT"),
    (errors.unprocessable, 422, "UNPROCESSABLE"),
    (errors.bad_gateway, 502, "UPSTREAM_FAILURE"),
    (errors.not_ready, 503, "NOT_READY"),
])
def test_a_factory_builds_the_error_of_its_status(factory, status, code):
    err = factory("what happened")

    assert (err.status_code, err.code, err.message, err.details) == (status, code, "what happened", None)
    assert err.code == ApiError(status_code=status, message="").code     # the code a bare status maps to


def test_a_factory_takes_its_own_code_and_the_details_by_name():
    err = errors.conflict("stale", "RENDER_VERSION_MISMATCH", doc_id="d", tile_num=3)

    assert err.body() == {"error": {"code": "RENDER_VERSION_MISMATCH", "message": "stale", "details": {"doc_id": "d", "tile_num": 3}}}


def test_unauthorized_carries_the_bearer_challenge():
    assert errors.unauthorized("no key").headers == {"WWW-Authenticate": "Bearer"}
    assert errors.forbidden("not here").headers is None


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
    assert set(body["missing"]) == {"store", "coordinator", "retriever", "tiler", "vlm"}


def test_ready_200_when_all_present(make_client):
    from conftest import FakeRetriever, FakeStore, FakeTiler, FakeVLM

    client = make_client(store=FakeStore(), retriever=FakeRetriever(),
                         tiler=FakeTiler(), vlm=FakeVLM())
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}
