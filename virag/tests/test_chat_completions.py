"""/v1/chat/completions end-to-end through the router with fake backends."""

from __future__ import annotations

import json

from virag.errors import ApiError

from conftest import FakeRetriever, FakeStore, FakeTiler, FakeVLM, hit_for


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
    tiler = FakeTiler()
    tiler.doc_metadata = {"doc-a": {"name": "Doc A", "virag_enrolled_at": 5}}
    client = make_client(vlm=FakeVLM(), store=_rag_store(), retriever=FakeRetriever(), tiler=tiler)

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    sources = r.json()["sources"]
    assert [s["tile_num"] for s in sources] == [0, 1]
    assert sources[0] == {"name": "Doc A", "doc_id": "doc-a", "tile_num": 0, "score": 0.9}
    assert tiler.metadata_calls == ["doc-a"]                 # the name comes from the tiler, once per doc


def test_rag_source_without_a_name_goes_by_its_doc_id(make_client):
    tiler = FakeTiler()
    tiler.doc_metadata = {"doc-a": {"project": "x"}}         # enrolled with allow_missing_name
    store = FakeStore(hits=[hit_for("doc-a", 0), hit_for("doc-b", 0)])   # doc-b: the tiler will not say
    client = make_client(vlm=FakeVLM(), store=store, retriever=FakeRetriever(), tiler=tiler)

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    assert [s["name"] for s in r.json()["sources"]] == ["doc-a", "doc-b"]


def test_rag_without_sources_asks_the_tiler_for_no_names(make_client):
    tiler = FakeTiler()
    client = make_client(vlm=FakeVLM(), store=_rag_store(), retriever=FakeRetriever(), tiler=tiler)

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}], "rag": True,
    })

    assert r.status_code == 200 and tiler.metadata_calls == []


def test_rag_retriever_defaults_reach_qdrant(make_client):
    store = _rag_store()
    client = make_client(vlm=FakeVLM(), store=store, retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}], "rag": True,
    })

    assert r.status_code == 200
    assert store.queries == [{"top_k": 5, "score_threshold": None}]   # RETRIEVER_TOP_K / RETRIEVER_THRESHOLD


def test_rag_overrides_reach_qdrant(make_client):
    store, tiler = _rag_store(), FakeTiler()
    client = make_client(vlm=FakeVLM(), store=store, retriever=FakeRetriever(), tiler=tiler)

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"top_k": 1, "retriever_threshold": 0.5},
    })

    assert r.status_code == 200
    assert store.queries == [{"top_k": 1, "score_threshold": 0.5}]    # the threshold is Qdrant's, not ours
    assert tiler.get_tile_calls == [("doc-a", 0)]


def test_rag_integrity_mismatch_skips_tile(make_client):
    # a hit whose stored image_hash does NOT match what the tiler regenerates
    bad_hit = hit_for("doc-a", 0, image_hash="deadbeef")
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


def test_rag_tile_without_hash_is_skipped(make_client):
    # a point that carries no image_hash cannot be checked, so it is not trusted
    hashless = hit_for("doc-a", 0, image_hash=None)
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=FakeStore(hits=[hashless]),
                         retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}],
        "rag": {"include_sources": True},
    })

    assert r.status_code == 200
    assert vlm.complete_bodies[0]["messages"][-1]["content"] == "q"
    assert r.json()["sources"] == []


def test_rag_render_version_mismatch_fails_the_request(make_client):
    # embedded from rv-old, while the tiler now renders rv-test
    stale = hit_for("doc-a", 0, render_version="rv-old")
    vlm = FakeVLM()
    client = make_client(vlm=vlm, store=FakeStore(hits=[hit_for("doc-a", 1), stale]),
                         retriever=FakeRetriever(), tiler=FakeTiler())

    r = client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}], "rag": {},
    })

    assert r.status_code == 409
    assert r.json()["error"]["code"] == "RENDER_VERSION_MISMATCH"
    assert vlm.complete_bodies == []                                    # never answered without tiles


class _RefusingTiler(FakeTiler):
    """get_tile fails as HttpService would map the tiler's `status` (None = unreachable)."""

    def __init__(self, status):
        super().__init__()
        self.status = status

    async def get_tile(self, doc_id, tile_num, *, render_version=None, bulk=False) -> bytes:
        details = {"error": "ConnectError()"} if self.status is None else {"upstream_status": self.status}
        raise ApiError(status_code=503, message="Upstream failed", details=details)


def _ask(make_client, tiler, vlm):
    client = make_client(vlm=vlm, store=_rag_store(), retriever=FakeRetriever(), tiler=tiler)

    return client.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "q"}], "rag": {},
    })


def test_rag_tiler_not_ready_fails_the_request(make_client):
    for status in (503, None):                                          # revising, unreachable
        vlm = FakeVLM()
        r = _ask(make_client, _RefusingTiler(status), vlm)

        assert r.status_code == 503, status
        assert r.json()["error"]["code"] == "NOT_READY"
        assert vlm.complete_bodies == []


def test_rag_tile_refused_by_the_tiler_is_skipped(make_client):
    vlm = FakeVLM()
    r = _ask(make_client, _RefusingTiler(410), vlm)                     # e.g. quarantined

    assert r.status_code == 200
    assert vlm.complete_bodies[0]["messages"][-1]["content"] == "q"


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
