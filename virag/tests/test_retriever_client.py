"""RetrieverClient: batching, single-item fallback, bulk concurrency/priority, startup probes."""

from __future__ import annotations

import asyncio
import json

import httpx2
import pytest

from virag.clients.retriever_client import RetrieverClient
from virag.errors import ApiError

from test_http_service import _HTTP, _mock, _run


DIM = 3


def _retriever(**kwargs) -> RetrieverClient:
    options = {"max_retries": 0, **_HTTP, **kwargs}

    return RetrieverClient("http://retriever/v1", "retriever", "k", **options)


def _count(body: dict) -> int:
    """How many vectors a well-behaved upstream returns for this request body."""
    if "input" in body:
        return len(body["input"])

    messages = body["messages"]

    return len(messages) if isinstance(messages[0], list) else 1


def _ok(body: dict, dim: int = DIM) -> httpx2.Response:
    n = _count(body)

    return httpx2.Response(200, json={"data": [
        {"index": i, "embedding": [float(i)] * dim} for i in reversed(range(n))
    ]})


def _models(req) -> httpx2.Response | None:
    if req.url.path.endswith("/models"):
        return httpx2.Response(200, json={"data": [{"id": "retriever"}]})

    return None


def _bodies(seen) -> list[dict]:
    return [json.loads(req.content) for req in seen if req.url.path.endswith("/embeddings")]


# ---------------------------------------------------------------------------
# embedding requests
# ---------------------------------------------------------------------------

def test_texts_split_into_batches_in_order():
    retriever = _retriever(batch_size=2)
    seen = _mock(retriever, lambda req, n: _ok(json.loads(req.content)))

    vectors = _run(retriever.embed_texts(["a", "b", "c", "d", "e"]))

    assert [len(body["input"]) for body in _bodies(seen)] == [2, 2, 1]
    assert vectors == [[0.0] * DIM, [1.0] * DIM, [0.0] * DIM, [1.0] * DIM, [0.0] * DIM]


def test_images_batched_as_conversations_when_supported():
    retriever = _retriever(batch_size=2, image_batching="on")
    seen = _mock(retriever, lambda req, n: _ok(json.loads(req.content)))

    vectors = _run(retriever.embed_images(["x", "y", "z"]))

    bodies = _bodies(seen)
    assert len(vectors) == 3
    assert isinstance(bodies[0]["messages"][0], list) and len(bodies[0]["messages"]) == 2
    assert bodies[1]["messages"][0]["role"] == "user"                 # lone image: plain conversation


def test_images_one_per_request_when_batching_off():
    retriever = _retriever(batch_size=8, image_batching="off")
    seen = _mock(retriever, lambda req, n: _ok(json.loads(req.content)))

    _run(retriever.embed_images(["x", "y", "z"]))

    assert len(_bodies(seen)) == 3
    assert all(body["messages"][0]["role"] == "user" for body in _bodies(seen))


def test_rejected_batch_falls_back_to_single_items():
    retriever = _retriever(batch_size=4)

    def handler(req, n):
        body = json.loads(req.content)
        return httpx2.Response(400) if len(body["input"]) > 1 else _ok(body)

    seen = _mock(retriever, handler)

    vectors = _run(retriever.embed_texts(["a", "b", "c"]))

    assert len(vectors) == 3
    assert [len(body["input"]) for body in _bodies(seen)] == [3, 1, 1, 1]


def test_rejected_single_item_is_an_error():
    retriever = _retriever(batch_size=4)

    def handler(req, n):
        body = json.loads(req.content)
        return httpx2.Response(422) if "bad" in body["input"] else _ok(body)

    _mock(retriever, handler)

    with pytest.raises(ApiError) as exc:
        _run(retriever.embed_texts(["a", "bad"]))

    assert exc.value.details["upstream_status"] == 422


def test_transient_failure_retries_the_batch_not_singles():
    retriever = _retriever(batch_size=4, max_retries=1)

    def handler(req, n):
        return httpx2.Response(503) if n == 1 else _ok(json.loads(req.content))

    seen = _mock(retriever, handler)

    _run(retriever.embed_texts(["a", "b"]))

    assert [len(body["input"]) for body in _bodies(seen)] == [2, 2]


def test_vector_count_mismatch_is_an_error():
    retriever = _retriever(batch_size=4)
    _mock(retriever, lambda req, n: httpx2.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]}))

    with pytest.raises(ApiError) as exc:
        _run(retriever.embed_texts(["a", "b"]))

    assert exc.value.status_code == 422


def test_priority_only_on_bulk_and_only_when_configured():
    retriever = _retriever(bulk_priority=10)
    seen = _mock(retriever, lambda req, n: _ok(json.loads(req.content)))

    _run(retriever.embed_texts(["query"]))
    _run(retriever.embed_texts(["ingest"], bulk=True))

    query, ingest = _bodies(seen)
    assert "priority" not in query
    assert ingest["priority"] == 10

    portable = _retriever(bulk_priority=0)
    seen = _mock(portable, lambda req, n: _ok(json.loads(req.content)))

    _run(portable.embed_texts(["ingest"], bulk=True))

    assert "priority" not in _bodies(seen)[0]


def test_bulk_calls_respect_max_concurrency_queries_bypass_it():
    retriever = _retriever(batch_size=1, max_concurrency=2, image_batching="off")
    in_flight = {"now": 0, "peak": 0}

    async def handler(req, n):
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1

        return _ok(json.loads(req.content))

    _mock(retriever, handler)

    _run(retriever.embed_images(["i"] * 8, bulk=True))
    assert in_flight["peak"] == 2

    in_flight["peak"] = 0
    _run(retriever.embed_images(["i"] * 8))
    assert in_flight["peak"] > 2


# ---------------------------------------------------------------------------
# check_connection probes
# ---------------------------------------------------------------------------

def test_check_auto_detects_batching_and_dimension():
    retriever = _retriever()
    _mock(retriever, lambda req, n: _models(req) or _ok(json.loads(req.content)))

    _run(retriever.check_connection())

    assert retriever.dimension == DIM
    assert retriever.batch_images is True


def test_check_auto_falls_back_when_batch_rejected():
    retriever = _retriever()

    def handler(req, n):
        if _models(req):
            return _models(req)

        body = json.loads(req.content)
        return httpx2.Response(400) if _count(body) > 1 and "messages" in body else _ok(body)

    _mock(retriever, handler)

    _run(retriever.check_connection())

    assert retriever.batch_images is False


def test_check_on_requires_batch_support():
    retriever = _retriever(image_batching="on")

    def handler(req, n):
        if _models(req):
            return _models(req)

        body = json.loads(req.content)
        return httpx2.Response(400) if "messages" in body and _count(body) > 1 else _ok(body)

    _mock(retriever, handler)

    with pytest.raises(ApiError, match="batched image"):
        _run(retriever.check_connection())


def test_check_rejects_model_without_image_support():
    retriever = _retriever(image_batching="off")

    def handler(req, n):
        if _models(req):
            return _models(req)

        body = json.loads(req.content)
        return httpx2.Response(400) if "messages" in body else _ok(body)

    _mock(retriever, handler)

    with pytest.raises(ApiError, match="image inputs"):
        _run(retriever.check_connection())


def test_check_rejects_mismatched_image_dimension():
    retriever = _retriever()

    def handler(req, n):
        if _models(req):
            return _models(req)

        body = json.loads(req.content)
        return _ok(body, dim=DIM + 1) if "messages" in body else _ok(body)

    _mock(retriever, handler)

    with pytest.raises(ApiError, match="-d image vectors"):
        _run(retriever.check_connection())


def test_check_rejects_unsupported_priority():
    retriever = _retriever(bulk_priority=10)

    def handler(req, n):
        if _models(req):
            return _models(req)

        body = json.loads(req.content)
        return httpx2.Response(400) if "priority" in body else _ok(body)

    _mock(retriever, handler)

    with pytest.raises(ApiError, match="scheduling-policy priority"):
        _run(retriever.check_connection())
