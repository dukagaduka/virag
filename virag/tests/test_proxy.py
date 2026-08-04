"""proxy header hygiene: hop-by-hop stripping and auth injection."""

from __future__ import annotations

import types

import httpx
from starlette.datastructures import Headers

from virag.api.proxy import _forward_request_headers, _forward_response_headers


def _request(headers: dict):
    return types.SimpleNamespace(headers=Headers(headers))


def test_request_strips_hop_by_hop_host_and_length():
    req = _request({
        "host": "gateway",
        "content-length": "10",
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
        "x-keep": "yes",
        "content-type": "application/json",
    })
    out = {k.lower(): v for k, v in _forward_request_headers(req, None).items()}

    assert "host" not in out
    assert "content-length" not in out
    assert "connection" not in out
    assert "transfer-encoding" not in out
    assert out["x-keep"] == "yes"
    assert out["content-type"] == "application/json"


def test_request_injects_auth_when_key_given():
    req = _request({"x": "1"})
    out = _forward_request_headers(req, "secret-key")
    assert out["Authorization"] == "Bearer secret-key"


def test_request_no_auth_when_key_none():
    req = _request({"authorization": "Bearer client-token"})
    out = {k.lower(): v for k, v in _forward_request_headers(req, None).items()}
    # client's own auth is passed through untouched when we don't inject
    assert out.get("authorization") == "Bearer client-token"


def test_request_injected_auth_overrides_client():
    req = _request({"authorization": "Bearer client-token"})
    out = _forward_request_headers(req, "server-key")
    assert out["Authorization"] == "Bearer server-key"


def test_response_strips_hop_by_hop_and_length():
    resp = httpx.Response(
        200,
        headers={
            "content-length": "5",
            "connection": "close",
            "content-type": "text/event-stream",
            "x-doc-id": "abc",
        },
    )
    out = {k.lower(): v for k, v in _forward_response_headers(resp).items()}

    assert "content-length" not in out
    assert "connection" not in out
    assert out["content-type"] == "text/event-stream"
    assert out["x-doc-id"] == "abc"
