"""RerankerClient: both wire formats, batching with scores kept in input order, the
startup probe, and how failures are classified for the chat fallback."""

from __future__ import annotations

import json

import httpx2
import pytest

from virag.clients.reranker_client import RerankerClient
from virag.errors import ApiError

from test_http_service import _HTTP, _mock, _run


def _reranker(api: str = "score", **kwargs) -> RerankerClient:
    options = {"max_retries": 0, "batch_size": 8, **_HTTP, **kwargs}

    return RerankerClient("http://reranker", "reranker", "k", api=api, **options)


def _image(body: dict, i: int) -> str:
    return body["documents"][i]["content"][0]["image_url"]["url"]


def _scored(body: dict, api: str) -> httpx2.Response:
    """A well-behaved upstream: image `img-<n>` scores n / 10, results out of order."""
    scores = [int(_image(body, i).rsplit("img-", 1)[1]) / 10 for i in range(len(body["documents"]))]
    order = reversed(range(len(scores)))

    if api == "score":
        return httpx2.Response(200, json={"data": [{"index": i, "score": scores[i]} for i in order]})

    return httpx2.Response(200, json={"results": [{"index": i, "relevance_score": scores[i]} for i in order]})


def _handler(api: str):
    def handle(req: httpx2.Request, n: int) -> httpx2.Response:
        if req.url.path == "/v1/models":
            return httpx2.Response(200, json={"data": [{"id": "reranker"}]})

        return _scored(json.loads(req.content), api)

    return handle


@pytest.mark.parametrize("api, path", [("score", "/score"), ("rerank", "/v1/rerank")])
def test_scores_come_back_in_input_order_across_batches(api, path):
    reranker = _reranker(api, batch_size=2)
    seen = _mock(reranker, _handler(api))

    scores = _run(reranker.score("q", [f"img-{n}" for n in (3, 1, 4, 1, 5)]))

    assert scores == [0.3, 0.1, 0.4, 0.1, 0.5]
    assert [req.url.path for req in seen] == [path] * 3
    assert [len(json.loads(req.content)["documents"]) for req in seen] == [2, 2, 1]


def test_score_payload():
    reranker = _reranker("score")
    seen = _mock(reranker, _handler("score"))

    _run(reranker.score("what is q3 revenue", ["img-1"]))

    body = json.loads(seen[0].content)
    assert body["model"] == "reranker" and body["queries"] == "what is q3 revenue"
    assert _image(body, 0) == "data:image/jpeg;base64,img-1"
    assert seen[0].headers["authorization"] == "Bearer k"


def test_rerank_payload_asks_for_no_documents_and_every_score():
    reranker = _reranker("rerank")
    seen = _mock(reranker, _handler("rerank"))

    _run(reranker.score("q", ["img-1", "img-2"]))

    body = json.loads(seen[0].content)
    assert body["query"] == "q" and body["return_documents"] is False
    assert "top_n" not in body                      # the cut is made over all batches at once


def test_no_images_no_requests():
    reranker = _reranker()
    seen = _mock(reranker, _handler("score"))

    assert _run(reranker.score("q", [])) == []
    assert seen == []


def test_check_connection_scores_a_probe_image():
    reranker = _reranker()
    seen = _mock(reranker, lambda req, n: (
        httpx2.Response(200, json={"data": [{"id": "reranker"}]}) if req.url.path == "/v1/models"
        else httpx2.Response(200, json={"data": [{"index": 0, "score": 0.5}]})
    ))

    _run(reranker.check_connection())

    assert [req.url.path for req in seen] == ["/v1/models", "/score"]
    assert _image(json.loads(seen[1].content), 0).startswith("data:image/png;base64,")


def test_check_connection_fails_on_a_text_only_reranker():
    reranker = _reranker()
    _mock(reranker, lambda req, n: (
        httpx2.Response(200, json={"data": [{"id": "reranker"}]}) if req.url.path == "/v1/models"
        else httpx2.Response(400, json={"error": "images are not supported"})
    ))

    with pytest.raises(ApiError) as e:
        _run(reranker.check_connection())

    assert RerankerClient.rejected(e.value)


def test_wrong_score_count_is_a_rejection():
    reranker = _reranker()
    _mock(reranker, lambda req, n: httpx2.Response(200, json={"data": [{"index": 0, "score": 0.5}]}))

    with pytest.raises(ApiError) as e:
        _run(reranker.score("q", ["img-1", "img-2"]))

    assert e.value.status_code == 422 and RerankerClient.rejected(e.value)


@pytest.mark.parametrize("status, rejected", [(400, True), (404, True), (429, False), (500, False), (503, False)])
def test_rejected_tells_contract_errors_from_outages(status, rejected):
    reranker = _reranker()
    _mock(reranker, lambda req, n: httpx2.Response(status, json={"error": "x"}))

    with pytest.raises(ApiError) as e:
        _run(reranker.score("q", ["img-1"]))

    assert RerankerClient.rejected(e.value) is rejected


def test_unreachable_reranker_is_an_outage():
    reranker = _reranker()

    def refuse(req, n):
        raise httpx2.ConnectError("refused")

    _mock(reranker, refuse)

    with pytest.raises(ApiError) as e:
        _run(reranker.score("q", ["img-1"]))

    assert e.value.status_code == 503 and not RerankerClient.rejected(e.value)
