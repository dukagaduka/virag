"""/v1/chat/completions end-to-end through the router with fake backends."""

from __future__ import annotations

import json

from conftest import FakeHit, FakeRetriever, FakeStore, FakeTiler, FakeVLM, hit_for


def _rag_store():
    return FakeStore(hits=[hit_for("doc-a", 0), hit_for("doc-a", 1)])


def test_passthrough_without_rag(make_client):
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=FakeStore(), retriever=FakeRetriever(), tiler=FakeTiler())

    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=body)

    assert r.status_code == 200
    # forwarded untouched, no retrieval happened
    assert vlm.complete_bodies[0]["messages"] == body["messages"]
    assert "sources" not in r.json()


def test_rag_injects_tiles(make_client):
    vlm, retriever, tiler = FakeVLM(), FakeRetriever(), FakeTiler()
    client = make_client(vlm=vlm, store=_rag_store(), retriever=retriever, tiler=tiler)

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "what is revenue"}], "rag": True,
    })

    assert r.status_code == 200
    assert retriever.text_calls == [["what is revenue"]]           # query embedded once
    assert tiler.get_tile_calls == [("doc-a", 0), ("doc-a", 1)]

    sent = vlm.complete_bodies[0]
    assert "rag" not in sent                                  # trigger stripped before upstream
    content = sent["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "FRAME"}   # default framing_text
    assert [p for p in content if p.get("type") == "image_url"].__len__() == 2
    assert content[-1] == {"type": "text", "text": "what is revenue"}


def test_rag_sources_in_json(make_client):
    client = make_client(vlm=FakeVLM(), store=_rag_store(), retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    sources = r.json()["sources"]
    assert [s["tile_num"] for s in sources] == [0, 1]
    assert sources[0] == {"name": "Doc A", "doc_id": "doc-a", "tile_num": 0, "score": 0.9}


def test_rag_integrity_mismatch_skips_tile(make_client):
    # a hit whose stored image_hash does NOT match what the tiler regenerates
    bad_hit = FakeHit(score=0.9, payload={
        "doc_id": "doc-a", "tile_num": 0, "image_hash": "deadbeef", "metadata": {"name": "A"},
    })
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=FakeStore(hits=[bad_hit]),
                         retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    # the tile was dropped: nothing injected, no sources
    assert vlm.complete_bodies[0]["messages"][-1]["content"] == "q"
    assert r.json()["sources"] == []


def test_rag_render_version_mismatch_fails_the_request(make_client):
    # embedded from rv-old, while the tiler now renders rv-test
    stale = hit_for("doc-a", 0)
    stale.payload["render_version"] = "rv-old"
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=FakeStore(hits=[hit_for("doc-a", 1), stale]),
                         retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}], "rag": {},
    })

    assert r.status_code == 409
    assert r.json()["error"]["code"] == "RENDER_VERSION_MISMATCH"
    assert vlm.complete_bodies == []                                    # never answered without tiles


def test_rag_without_query_text_is_422(make_client):
    client = make_client(vlm=FakeVLM(), store=_rag_store(), retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "system", "content": "only system"}], "rag": True,
    })

    assert r.status_code == 422
    assert r.json()["error"]["code"] == "UNPROCESSABLE"


def test_streaming_sources_spliced_before_done(make_client):
    vlm = FakeVLM(stream_chunks=[b"data: {\"choices\":[{}]}\n\n", b"data: [DONE]\n\n"])
    client = make_client(vlm=vlm, store=_rag_store(), retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "stream": True, "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    text = r.text
    assert '"sources"' in text
    assert text.index('"sources"') < text.index("[DONE]")     # spliced before the sentinel


def test_invalid_json_body_is_422(make_client):
    client = make_client(vlm=FakeVLM(), store=FakeStore(), retriever=FakeRetriever(), tiler=FakeTiler())
    r = client.post("/v1/chat/completions", content="not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_non_object_body_is_422(make_client):
    client = make_client(vlm=FakeVLM(), store=FakeStore(), retriever=FakeRetriever(), tiler=FakeTiler())
    r = client.post("/v1/chat/completions", content=json.dumps([1, 2, 3]),
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
