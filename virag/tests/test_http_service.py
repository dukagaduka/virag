"""HttpService transport: retries, backoff, error mapping, and the concrete clients on top."""

from __future__ import annotations

import asyncio

import httpx2
import pytest

from virag.clients.retriever_client import RetrieverClient
from virag.clients.http_service import HttpService
from virag.clients.tiler_client import TilerService
from virag.clients.vlm_client import VLMClient
from virag.errors import ApiError


_HTTP = dict(timeout=5, connect_timeout=5, max_connections=4, backoff_base=0, backoff_cap=0)


class _Service(HttpService):

    async def check_connection(self) -> None:
        pass


def _mock(service: HttpService, handler) -> list[httpx2.Request]:
    """Route the service's pooled client through `handler`; return the seen requests."""
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request, len(seen))

    service.http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(record),
        headers=service.http.headers,
    )

    return seen


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# send / retries
# ---------------------------------------------------------------------------

def test_retries_transient_status_then_succeeds():
    svc = _Service("http://up", max_retries=3, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(503 if n < 3 else 200))

    response = _run(svc.send("GET", "http://up/x"))

    assert response.status_code == 200
    assert len(seen) == 3


def test_exhausted_retries_return_last_response():
    svc = _Service("http://up", max_retries=2, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(502, json={"n": n}))

    response = _run(svc.send("GET", "http://up/x"))

    assert response.status_code == 502
    assert response.json() == {"n": 3}
    assert len(seen) == 3


def test_non_retryable_status_is_not_replayed():
    svc = _Service("http://up", max_retries=3, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(400))

    assert _run(svc.send("GET", "http://up/x")).status_code == 400
    assert len(seen) == 1


def test_retry_disabled_sends_once():
    svc = _Service("http://up", max_retries=3, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(503))

    assert _run(svc.send("GET", "http://up/x", retry=False)).status_code == 503
    assert len(seen) == 1


def test_connect_error_is_retried():
    svc = _Service("http://up", max_retries=2, **_HTTP)

    def handler(req, n):
        if n == 1:
            raise httpx2.ConnectError("refused", request=req)
        return httpx2.Response(200)

    seen = _mock(svc, handler)

    assert _run(svc.send("GET", "http://up/x")).status_code == 200
    assert len(seen) == 2


def test_read_timeout_is_not_retried():
    svc = _Service("http://up", max_retries=3, **_HTTP)

    def handler(req, n):
        raise httpx2.ReadTimeout("slow", request=req)

    seen = _mock(svc, handler)

    with pytest.raises(httpx2.ReadTimeout):
        _run(svc.send("GET", "http://up/x"))
    assert len(seen) == 1


def test_backoff_is_capped_and_jittered():
    svc = _Service("http://up", max_retries=0, timeout=5, connect_timeout=5, max_connections=4,
                   backoff_base=1, backoff_cap=4)

    delays = [svc._backoff(10) for _ in range(200)]

    assert all(0 <= d <= 4 for d in delays)
    assert len(set(delays)) > 1


def test_retry_after_is_honoured_and_capped():
    svc = _Service("http://up", max_retries=0, timeout=5, connect_timeout=5, max_connections=4,
                   backoff_base=1, backoff_cap=4)

    assert svc._retry_after(httpx2.Response(429, headers={"retry-after": "2"})) == 2
    assert svc._retry_after(httpx2.Response(429, headers={"retry-after": "60"})) == 4
    assert svc._retry_after(httpx2.Response(429)) is None


# ---------------------------------------------------------------------------
# _request / _json error mapping
# ---------------------------------------------------------------------------

def test_request_maps_error_status_to_api_error():
    svc = _Service("http://up", max_retries=0, **_HTTP)
    _mock(svc, lambda req, n: httpx2.Response(500, json={"detail": "boom"}))

    with pytest.raises(ApiError) as exc:
        _run(svc._request("GET", "http://up/x", action="the test"))

    assert exc.value.status_code == 503
    assert exc.value.details == {"upstream_status": 500, "upstream_error": {"detail": "boom"}}


def test_request_maps_unreachable_to_api_error():
    svc = _Service("http://up", max_retries=0, **_HTTP)

    def handler(req, n):
        raise httpx2.ConnectError("refused", request=req)

    _mock(svc, handler)

    with pytest.raises(ApiError) as exc:
        _run(svc._request("GET", "http://up/x", action="the test"))

    assert exc.value.status_code == 503
    assert "during the test" in exc.value.message


def test_json_maps_bad_structure_to_422():
    svc = _Service("http://up", max_retries=0, **_HTTP)

    with pytest.raises(ApiError) as exc:
        svc._json(httpx2.Response(200, json={}, request=httpx2.Request("GET", "http://up/x")),
                  action="the test", parse=lambda data: data["missing"])

    assert exc.value.status_code == 422


def test_gather_keeps_order():
    async def value(v, delay):
        await asyncio.sleep(delay)
        return v

    assert _run(HttpService._gather([value(1, 0.02), value(2, 0), value(3, 0.01)])) == [1, 2, 3]


def test_gather_cancels_siblings_and_raises_plain_error():
    finished = []

    async def slow():
        await asyncio.sleep(1)
        finished.append("slow")

    async def boom():
        raise ApiError(status_code=503, message="boom")

    with pytest.raises(ApiError, match="boom"):
        _run(HttpService._gather([slow(), boom()]))

    assert finished == []


# ---------------------------------------------------------------------------
# concrete clients
# ---------------------------------------------------------------------------

def test_retriever_sends_bearer_and_orders_vectors():
    retriever = RetrieverClient("http://retriever/v1", "retriever", "k-retriever", max_retries=0, **_HTTP)

    def handler(req, n):
        assert req.headers["authorization"] == "Bearer k-retriever"
        return httpx2.Response(200, json={"data": [
            {"index": 1, "embedding": [1.0]},
            {"index": 0, "embedding": [0.0]},
        ]})

    _mock(retriever, handler)

    assert _run(retriever.embed_texts(["a", "b"])) == [[0.0], [1.0]]


def test_openai_check_rejects_missing_model():
    retriever = RetrieverClient("http://retriever/v1", "retriever", "k", max_retries=0, **_HTTP)
    _mock(retriever, lambda req, n: httpx2.Response(200, json={"data": [{"id": "other"}]}))

    with pytest.raises(ApiError) as exc:
        _run(retriever.check_connection())

    assert exc.value.status_code == 404


def test_vlm_complete_retries_upstream_503_and_passes_through():
    vlm = VLMClient("http://vlm/v1", "vlm", "k", max_retries=1, **_HTTP)
    seen = _mock(vlm, lambda req, n: httpx2.Response(503, json={"error": "busy"}))

    response = _run(vlm.complete({"messages": []}))

    assert response.status_code == 503
    assert len(seen) == 2


def test_tiler_doc_exists_statuses():
    tiler = TilerService("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(206 if req.url.path.endswith("/a") else 404))

    assert _run(tiler.doc_exists("a")) is True
    assert _run(tiler.doc_exists("b")) is False


def test_tiler_stream_is_never_replayed():
    tiler = TilerService("http://tiler", max_retries=3, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(503, text="busy"))

    async def consume():
        return [event async for event in tiler.tile_stream("PDF", document=b"x")]

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert exc.value.details["upstream_status"] == 503
    assert len(seen) == 1


def test_tiler_stream_yields_events():
    tiler = TilerService("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(200, content=b'{"event":"begin"}\n\n{"event":"committed"}\n'))

    async def consume():
        return [event async for event in tiler.tile_stream("PDF", document=b"x")]

    assert _run(consume()) == [{"event": "begin"}, {"event": "committed"}]
