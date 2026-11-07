"""/v1/chat/completions with a reranker: it picks `top_n` of the retriever's candidates,
best first, and a failing reranker falls back to the retriever's order."""

from __future__ import annotations

import base64
import logging

from virag.errors import ApiError
from virag.settings import RerankerSettings

from conftest import FakeReranker, FakeRetriever, FakeStore, FakeTiler, FakeVLM, hit_for, make_settings, tile_bytes


def _store():
    # Qdrant order: tiles 0, 1, 2, 3 (best retriever score first).
    return FakeStore(hits=[hit_for("doc-a", n, score=0.9 - n / 10) for n in range(4)])


def _settings(**reranker):
    return make_settings(reranker_settings=RerankerSettings(base_url="http://reranker", **{"top_n": 2, **reranker}))


def _ask(client, rag: dict):
    return client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "what is revenue"}],
        "rag": {"include_sources": True, **rag},
    })


def _sent_tiles(vlm) -> list[int]:
    """Which tiles reached the VLM, in prompt order."""
    urls = [p["image_url"]["url"] for p in vlm.complete_bodies[0]["messages"][-1]["content"] if p["type"] == "image_url"]
    by_b64 = {base64.b64encode(tile_bytes("doc-a", n)).decode(): n for n in range(4)}

    return [by_b64[url.removeprefix("data:image/jpeg;base64,")] for url in urls]


def test_reranker_picks_top_n_best_first(make_client):
    vlm, reranker = FakeVLM(), FakeReranker(scores=[0.1, 0.7, 0.2, 0.9])
    client = make_client(settings=_settings(), vlm=vlm, store=_store(), retriever=FakeRetriever(),
                         reranker=reranker, tiler=FakeTiler())

    r = _ask(client, {})

    assert r.status_code == 200
    assert reranker.calls == [("what is revenue", 4)]               # every candidate, the query text
    assert _sent_tiles(vlm) == [3, 1]
    assert [(s["tile_num"], s["rerank_score"]) for s in r.json()["sources"]] == [(3, 0.9), (1, 0.7)]
    assert r.json()["sources"][0]["score"] == 0.9 - 3 / 10          # the retriever's score stays


def test_reranker_threshold_drops_weak_tiles(make_client):
    vlm = FakeVLM()
    client = make_client(settings=_settings(top_n=3, threshold=0.5), vlm=vlm, store=_store(),
                         retriever=FakeRetriever(), reranker=FakeReranker(scores=[0.1, 0.7, 0.2, 0.9]),
                         tiler=FakeTiler())

    r = _ask(client, {})

    assert _sent_tiles(vlm) == [3, 1]                                # 3 allowed, 2 pass the threshold


def test_request_overrides_top_n_and_threshold(make_client):
    vlm = FakeVLM()
    client = make_client(settings=_settings(), vlm=vlm, store=_store(), retriever=FakeRetriever(),
                         reranker=FakeReranker(scores=[0.1, 0.7, 0.2, 0.9]), tiler=FakeTiler())

    r = _ask(client, {"top_n": 3, "reranker_threshold": 0.15})

    assert _sent_tiles(vlm) == [3, 1, 2]


def test_top_n_above_top_k_is_422(make_client):
    client = make_client(settings=_settings(), vlm=FakeVLM(), store=_store(), retriever=FakeRetriever(),
                         reranker=FakeReranker(), tiler=FakeTiler())

    r = _ask(client, {"top_k": 2, "top_n": 3})

    assert r.status_code == 422


def test_reranker_down_falls_back_to_the_retriever_order(make_client, caplog):
    vlm = FakeVLM()
    down = FakeReranker(error=ApiError(status_code=503, message="Cannot reach http://reranker/score"))
    client = make_client(settings=_settings(), vlm=vlm, store=_store(), retriever=FakeRetriever(),
                         reranker=down, tiler=FakeTiler())

    with caplog.at_level(logging.WARNING):
        r = _ask(client, {})

    assert r.status_code == 200
    assert _sent_tiles(vlm) == [0, 1]                                # Qdrant's order, cut to top_n
    assert all("rerank_score" not in s for s in r.json()["sources"])
    assert [rec.levelno for rec in caplog.records if "rerank failed" in rec.message] == [logging.WARNING]


def test_reranker_refusal_falls_back_and_logs_an_error(make_client, caplog):
    refused = FakeReranker(error=ApiError(status_code=503, message="rejected", details={"upstream_status": 400}))
    client = make_client(settings=_settings(), vlm=FakeVLM(), store=_store(), retriever=FakeRetriever(),
                         reranker=refused, tiler=FakeTiler())

    with caplog.at_level(logging.WARNING):
        r = _ask(client, {})

    assert r.status_code == 200
    assert [rec.levelno for rec in caplog.records if "rerank failed" in rec.message] == [logging.ERROR]


def test_without_reranker_every_candidate_reaches_the_vlm(make_client):
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=_store(), retriever=FakeRetriever(), tiler=FakeTiler())

    r = _ask(client, {"top_n": 9})                                   # no reranker: top_n is moot

    assert r.status_code == 200
    assert _sent_tiles(vlm) == [0, 1, 2, 3]
    assert all("rerank_score" not in s for s in r.json()["sources"])


def test_no_candidates_no_rerank_call(make_client):
    reranker = FakeReranker()
    client = make_client(settings=_settings(), vlm=FakeVLM(), store=FakeStore(), retriever=FakeRetriever(),
                         reranker=reranker, tiler=FakeTiler())

    r = _ask(client, {})

    assert r.status_code == 200 and r.json()["sources"] == []
    assert reranker.calls == []
