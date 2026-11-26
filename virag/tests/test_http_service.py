"""HttpService transport: retries, backoff, error mapping, and the concrete clients on top."""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager

import httpx2
import pytest

from virag.clients.retriever_client import RetrieverClient
from virag.clients.http_service import HttpService
from virag.clients.tiler_client import Begin, Committed, Tile, TilerClient, TilerDoc
from virag.clients.vlm_client import VLMClient
from virag.errors import ApiError
from virag.utils import gather

from conftest import make_settings


_HTTP = dict(timeout=5, connect_timeout=5, max_connections=4, max_concurrency=0, backoff_base=0, backoff_cap=0)
_RETRIEVER = dict(batch_size=32, bulk_priority=0, image_batching="auto")


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


@asynccontextmanager
async def _upstream(delay: float = 0):
    """A real HTTP upstream on localhost answering every request 200 after `delay`;
    yields its URL and a dict with the peak number of requests it served at once."""
    load = {"now": 0, "peak": 0}

    async def serve(reader, writer):
        try:
            while await reader.readuntil(b"\r\n\r\n"):
                load["now"] += 1
                load["peak"] = max(load["peak"], load["now"])
                await asyncio.sleep(delay)
                load["now"] -= 1

                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionError):
            pass

        finally:
            writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)

    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", load

    finally:
        server.close()


# ---------------------------------------------------------------------------
# connections
# ---------------------------------------------------------------------------

def test_connections_carry_tcp_keepalive():
    """What lets the kernel find a vanished upstream dead in ~25 s instead of the read timeout."""
    async def go():
        async with _upstream() as (url, _):
            svc = _Service(url, max_retries=0, **_HTTP)

            try:
                response = await svc.send("GET", f"{url}/x")
                sock = response.extensions["network_stream"].get_extra_info("socket")

                return [sock.getsockopt(level, option) for level, option, _ in HttpService.SOCKET_OPTIONS]

            finally:
                await svc.close()

    assert _run(go()) == [1, 10, 5, 3]
    assert [option for _, option, _ in HttpService.SOCKET_OPTIONS] == [
        socket.SO_KEEPALIVE, socket.TCP_KEEPIDLE, socket.TCP_KEEPINTVL, socket.TCP_KEEPCNT,
    ]


def test_the_pool_is_capped_at_max_connections():
    async def go():
        async with _upstream(delay=0.02) as (url, load):
            svc = _Service(url, max_retries=0, **{**_HTTP, "max_connections": 2})

            try:
                await gather(svc.send("GET", f"{url}/{n}") for n in range(6))

            finally:
                await svc.close()

            return load["peak"]

    assert _run(go()) == 2


# ---------------------------------------------------------------------------
# send / retries
# ---------------------------------------------------------------------------

def test_retries_transient_status_then_succeeds():
    svc = _Service("http://up", max_retries=3, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(503 if n < 3 else 200))

    response = _run(svc.send("GET", "http://up/x"))

    assert response.status_code == 200
    assert len(seen) == 3


def test_json_body_is_encoded_once_by_orjson_and_replayed():
    svc = _Service("http://up", max_retries=1, **_HTTP)
    seen = _mock(svc, lambda req, n: httpx2.Response(503 if n < 2 else 200))

    _run(svc.send("POST", "http://up/x", json={"k": "é", "n": 1}, headers={"X-Trace": "t"}))

    assert [r.content for r in seen] == ['{"k":"é","n":1}'.encode()] * 2   # compact UTF-8
    assert seen[1].headers["content-type"] == "application/json"
    assert seen[1].headers["x-trace"] == "t"                                  # caller headers kept


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


def test_read_timeout_is_retried_when_asked():
    svc = _Service("http://up", max_retries=2, **_HTTP)

    def handler(req, n):
        if n == 1:
            raise httpx2.ReadTimeout("stalled", request=req)
        return httpx2.Response(200)

    seen = _mock(svc, handler)

    assert _run(svc.send("GET", "http://up/x", retry_read_timeouts=True)).status_code == 200
    assert len(seen) == 2


def test_timeout_is_reported_as_timeout():
    svc = _Service("http://up", max_retries=0, **_HTTP)

    def handler(req, n):
        raise httpx2.ReadTimeout("stalled", request=req)

    _mock(svc, handler)

    with pytest.raises(ApiError, match="Timed out waiting for"):
        _run(svc._request("GET", "http://up/x", action="the test"))


def test_backoff_is_capped_and_jittered():
    svc = _Service("http://up", max_retries=0, timeout=5, connect_timeout=5, max_connections=4,
                   max_concurrency=0, backoff_base=1, backoff_cap=4)

    delays = [svc._backoff(10) for _ in range(200)]

    assert all(0 <= d <= 4 for d in delays)
    assert len(set(delays)) > 1


def test_retry_after_is_honoured_and_capped():
    svc = _Service("http://up", max_retries=0, timeout=5, connect_timeout=5, max_connections=4,
                   max_concurrency=0, backoff_base=1, backoff_cap=4)

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

    assert _run(gather([value(1, 0.02), value(2, 0), value(3, 0.01)])) == [1, 2, 3]


def test_gather_cancels_siblings_and_raises_plain_error():
    finished = []

    async def slow():
        await asyncio.sleep(1)
        finished.append("slow")

    async def boom():
        raise ApiError(status_code=503, message="boom")

    with pytest.raises(ApiError, match="boom"):
        _run(gather([slow(), boom()]))

    assert finished == []


# ---------------------------------------------------------------------------
# concrete clients
# ---------------------------------------------------------------------------

def test_retriever_sends_bearer_and_orders_vectors():
    retriever = RetrieverClient("http://retriever/v1", "retriever", "k-retriever", max_retries=0, **_RETRIEVER, **_HTTP)

    def handler(req, n):
        assert req.headers["authorization"] == "Bearer k-retriever"
        return httpx2.Response(200, json={"data": [
            {"index": 1, "embedding": [1.0]},
            {"index": 0, "embedding": [0.0]},
        ]})

    _mock(retriever, handler)

    assert _run(retriever.embed_texts(["a", "b"])) == [[0.0], [1.0]]


def test_openai_check_rejects_missing_model():
    retriever = RetrieverClient("http://retriever/v1", "retriever", "k", max_retries=0, **_RETRIEVER, **_HTTP)
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


@pytest.mark.parametrize("error, told", [
    (httpx2.ConnectError, "Cannot reach http://vlm/v1/chat/completions during chat completion"),
    (httpx2.ReadTimeout, "Timed out waiting for http://vlm/v1/chat/completions during chat completion"),
])
def test_vlm_that_cannot_be_reached_is_a_502_naming_the_attempts(error, told):
    vlm = VLMClient("http://vlm/v1", "vlm", "k", max_retries=2, **_HTTP)

    def handler(req, n):
        raise error("no answer", request=req)

    seen = _mock(vlm, handler)

    with pytest.raises(ApiError) as exc:
        _run(vlm.complete({"messages": []}))

    assert (exc.value.status_code, exc.value.message) == (502, told)
    assert exc.value.details["attempts"] == 3 and "no answer" in exc.value.details["error"]
    assert len(seen) == (3 if error is httpx2.ConnectError else 1)      # a read timeout is not replayed


def test_vlm_stream_that_cannot_be_opened_is_a_502():
    vlm = VLMClient("http://vlm/v1", "vlm", "k", max_retries=0, **_HTTP)

    def handler(req, n):
        raise httpx2.ConnectError("refused", request=req)

    _mock(vlm, handler)

    async def open_stream():
        async with vlm.stream({"messages": [], "stream": True}):
            pass

    with pytest.raises(ApiError, match="during streaming chat completion") as exc:
        _run(open_stream())

    assert exc.value.status_code == 502 and exc.value.details["attempts"] == 1


def test_an_image_is_a_data_url_part():
    assert VLMClient.image_part("QUJD") == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert VLMClient.image_part("QUJD", "image/png")["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_every_client_takes_its_transport_options_from_its_settings():
    settings = make_settings()
    hs = settings.http_settings

    assert HttpService.http_options(settings.tiler_settings, hs) == {
        "timeout": 120, "connect_timeout": 10, "max_connections": 100, "max_concurrency": 0,
        "max_retries": 0, "backoff_base": 0, "backoff_cap": 0,
    }

    for client, section in [
        (RetrieverClient, settings.retriever_settings),
        (TilerClient, settings.tiler_settings),
        (VLMClient, settings.vlm_settings),
    ]:
        built = client.from_settings(section, hs)

        assert (built.base_url, built.max_retries, built.max_concurrency) == (section.base_url, section.max_retries, section.max_concurrency)
        assert built.http.timeout == httpx2.Timeout(section.timeout_sec, connect=section.connect_timeout_sec)


def test_tiler_doc_exists_statuses():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(206 if req.url.path.endswith("/a") else 404))

    assert _run(tiler.doc_exists("a")) is True
    assert _run(tiler.doc_exists("b")) is False


def test_tiler_tile_is_checked_against_its_embedded_render_version():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    seen = []

    def handler(req, n):
        seen.append(str(req.url))

        return httpx2.Response(200, content=b"jpeg", headers={"X-Render-Version": "rv-new"})

    _mock(tiler, handler)

    assert _run(tiler.get_tile("a", 0, render_version="rv-new")) == b"jpeg"
    assert _run(tiler.get_tile("a", 3, verify=False)) == b"jpeg"       # nothing to check against
    assert seen == ["http://tiler/doc/a/tiles/0?verify=true", "http://tiler/doc/a/tiles/3?verify=false"]

    with pytest.raises(ApiError) as exc:
        _run(tiler.get_tile("a", 0, render_version="rv-old"))

    assert (exc.value.status_code, exc.value.code) == (409, "RENDER_VERSION_MISMATCH")
    assert exc.value.details["tiler_render_version"] == "rv-new"


# A tile stream on the wire, line by line.
_BEGIN = b'{"event":"begin","doc_id":"abc","doc_type":"PDF","source_url":null,"render_version":"rv"}'
_TILE = b'{"event":"tile","tile_num":0,"width":875,"height":1024,"image_hash":"h0","data":"b64"}'
_COMMITTED = (
    b'{"event":"committed","doc_id":"abc","doc_type":"PDF","source_url":null,"render_version":"rv",'
    b'"height":1024,"tile_count":1,"result":"rendered"}'
)

_STREAM = [
    Begin(doc_id="abc", doc_type="PDF", render_version="rv"),
    Tile(tile_num=0, image_hash="h0", data="b64", width=875, height=1024),
    Committed(doc_id="abc", tile_count=1, doc_type="PDF", render_version="rv", height=1024, result="rendered"),
]


async def _body():
    """An ingest form on its way through, in two chunks."""
    yield b"--b\r\n"
    yield b"--b--\r\n"


def test_tiler_stream_relays_the_body_with_its_headers():
    tiler = TilerClient("http://tiler", api_key="k", max_retries=0, **_HTTP)
    sent = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        sent["url"] = str(request.url)
        sent["headers"] = request.headers
        sent["body"] = request.content

        return httpx2.Response(200, content=_BEGIN + b"\n")

    tiler.http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), headers=tiler.http.headers)

    async def consume():
        stream = tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b", content_length="14")

        return [event async for event in stream]

    assert _run(consume()) == [Begin(doc_id="abc", doc_type="PDF", render_version="rv")]
    assert (sent["url"], sent["body"]) == ("http://tiler/doc?tiles=true", b"--b\r\n--b--\r\n")
    assert sent["headers"]["content-type"] == "multipart/form-data; boundary=b"
    assert sent["headers"]["content-length"] == "14" and "transfer-encoding" not in sent["headers"]
    assert sent["headers"]["authorization"] == "Bearer k"


def test_tiler_stream_is_never_replayed():
    tiler = TilerClient("http://tiler", max_retries=3, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(503, text="busy"))

    async def consume():
        return [event async for event in tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b")]

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert exc.value.details["upstream_status"] == 503
    assert len(seen) == 1


def test_tiler_stream_relays_a_document_the_tiler_refuses():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    refusal = {"error": {
        "code": "UNSUPPORTED_MEDIA",
        "message": "declared doc_type=PDF but the payload does not start with %PDF-",
        "details": {"declared": "PDF"},
    }}
    _mock(tiler, lambda req, n: httpx2.Response(415, json=refusal))

    async def consume():
        return [event async for event in tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b")]

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert (exc.value.status_code, exc.value.code) == (415, "UNSUPPORTED_MEDIA")   # the caller's to fix
    assert "does not start with %PDF-" in exc.value.message
    assert exc.value.details == {"tiler_code": "UNSUPPORTED_MEDIA", "tiler_details": {"declared": "PDF"}}


def test_tiler_stream_rejecting_our_key_is_not_the_callers_fault():
    tiler = TilerClient("http://tiler", api_key="wrong", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(401, json={"error": {"code": "UNAUTHORIZED", "message": "no"}}))

    async def consume():
        return [event async for event in tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b")]

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert exc.value.status_code == 503
    assert exc.value.details["upstream_status"] == 401


def test_tiler_stream_yields_events():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(200, content=_BEGIN + b"\n\n" + _TILE + b"\n" + _COMMITTED + b"\n"))

    async def consume():
        return [event async for event in tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b")]

    assert _run(consume()) == _STREAM


def test_tiler_stream_raises_the_error_event_the_tiler_ends_it_with():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(
        200, content=_BEGIN + b'\n{"event":"error","code":"RENDER_FAILED","message":"boom"}\n',
    ))
    seen = []

    async def consume():
        async for event in tiler.ingest_stream(_body(), content_type="multipart/form-data; boundary=b"):
            seen.append(event)

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert seen == _STREAM[:1]                                          # what came before it was delivered
    assert (exc.value.status_code, exc.value.message) == (502, "tiler stream error: boom")
    assert exc.value.details == {"tiler_code": "RENDER_FAILED"}


def test_tiler_stream_refuses_an_event_without_its_fields():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, lambda req, n: httpx2.Response(200, content=b'{"event":"begin","doc_id":"abc"}\n'))

    async def consume():
        return [event async for event in tiler.doc_stream("abc")]

    with pytest.raises(ApiError, match="malformed `begin` event") as exc:
        _run(consume())

    assert exc.value.status_code == 502


def test_tiler_doc_stream_yields_the_events_of_a_stored_doc():
    tiler = TilerClient("http://tiler", max_retries=3, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, content=_BEGIN + b"\n" + _TILE + b"\n" + _COMMITTED + b"\n"))

    async def consume():
        return [event async for event in tiler.doc_stream("abc")]

    assert _run(consume()) == _STREAM
    assert [(req.method, str(req.url)) for req in seen] == [("GET", "http://tiler/doc/abc/tiles")]


def test_tiler_doc_stream_fails_before_it_begins_for_a_doc_the_tiler_does_not_serve():
    tiler = TilerClient("http://tiler", max_retries=3, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(410, json={"error": {"code": "QUARANTINED", "message": "no"}}))

    async def consume():
        return [event async for event in tiler.doc_stream("abc")]

    with pytest.raises(ApiError) as exc:
        _run(consume())

    assert exc.value.details["upstream_status"] == 410
    assert len(seen) == 1                                               # never replayed


def test_tiler_bulk_get_tile_is_capped_interactive_is_not():
    tiler = TilerClient("http://tiler", max_retries=0, **{**_HTTP, "max_concurrency": 2})
    in_flight = {"now": 0, "peak": 0}

    async def handler(req, n):
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1

        return httpx2.Response(200, content=b"jpeg")

    _mock(tiler, handler)

    _run(gather(tiler.get_tile("d", n, bulk=True) for n in range(6)))
    assert in_flight["peak"] == 2

    in_flight["peak"] = 0
    _run(gather(tiler.get_tile("d", n) for n in range(6)))
    assert in_flight["peak"] > 2


# ---------------------------------------------------------------------------
# what virag keeps in the tiler
# ---------------------------------------------------------------------------

def _json(request: httpx2.Request):
    import json

    return json.loads(request.content)


def test_tiler_docs_follows_the_cursor_and_types_the_docs():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    pages = [
        {"items": [{"doc_id": "a", "doc_type": "PDF", "tile_count": 2, "complete": True, "quarantined": False,
                    "render_version": "rv", "created_at": 1, "source_url": None,
                    "metadata": {"name": "A", "virag_enrolled_at": 7}}],
         "next_cursor": "c1"},
        {"items": [{"doc_id": "b", "doc_type": "URL", "tile_count": 0, "complete": False, "quarantined": True,
                    "render_version": None, "created_at": 2, "metadata": {}}],
         "next_cursor": None},
    ]
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, json=pages[n - 1]))

    a, b = _run(tiler.docs())

    assert a == TilerDoc("a", "PDF", 2, "rv", True, False, {"name": "A", "virag_enrolled_at": 7})
    assert (a.embeddable, a.enrolled, a.enrolled_at) == (True, True, 7)
    assert (b.embeddable, b.enrolled, b.render_version) == (False, False, "")
    assert [_json(req) for req in seen] == [{"limit": 256}, {"limit": 256, "cursor": "c1"}]


def test_tiler_pending_docs_are_searched_by_the_enroll_marker():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, json={"items": [{"doc_id": "p"}], "next_cursor": None}))

    assert [doc.doc_id for doc in _run(tiler.pending_docs())] == ["p"]
    assert _json(seen[0])["filters"] == [{"key": "virag_enrolled_at", "op": "eq", "value": 0}]


def test_tiler_marks_a_doc_pending_with_its_metadata_then_enrolled():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, json={}))

    _run(tiler.mark_pending("abc", {"name": "A"}))
    _run(tiler.mark_enrolled("abc"))

    assert [(req.method, req.url.path) for req in seen] == [("PATCH", "/doc/abc/metadata")] * 2
    assert _json(seen[0]) == {"name": "A", "virag_enrolled_at": 0}
    assert _json(seen[1]).keys() == {"virag_enrolled_at"} and _json(seen[1])["virag_enrolled_at"] > 0


def test_tiler_render_version_is_refused_while_the_tiler_is_not_serving():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    info = {"render_version": "rv", "revision": {"state": "running", "serving": False}}
    _mock(tiler, lambda req, n: httpx2.Response(200, json=info))

    with pytest.raises(ApiError, match="not serving") as exc:
        _run(tiler.render_version())

    assert exc.value.status_code == 409

    info["revision"] = {"state": "idle", "serving": True}
    assert _run(tiler.render_version()) == "rv"


def _schema(*fields: tuple[str, str]):
    return lambda req, n: httpx2.Response(200, json={"version": 1, "fields": [{"key": k, "type": t} for k, t in fields]})


def test_tiler_schema_with_the_required_fields_passes():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, _schema(("name", "text"), ("virag_enrolled_at", "number"), ("extra", "keyword")))

    _run(tiler.check_schema())                                          # does not raise


def test_tiler_schema_missing_a_required_field_is_refused():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, _schema(("name", "text")))

    with pytest.raises(ApiError, match="missing required field 'virag_enrolled_at'") as exc:
        _run(tiler.check_schema())

    assert exc.value.status_code == 503


def test_tiler_schema_declaring_a_required_field_by_another_type_is_refused():
    tiler = TilerClient("http://tiler", max_retries=0, **_HTTP)
    _mock(tiler, _schema(("name", "text"), ("virag_enrolled_at", "text")))          # should be number

    with pytest.raises(ApiError, match="declared 'text' but virag needs 'number'"):
        _run(tiler.check_schema())
