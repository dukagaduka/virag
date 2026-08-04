"""/enroll: name policy, deterministic ids, embed+upsert, zero-tile handling."""

from __future__ import annotations

import json

from virag.api.enroll import point_id

from conftest import FakeEmbeddings, FakeStore, FakeTiler, sample_tile_result


def _client(make_client, *, n_tiles=2):
    store, emb = FakeStore(), FakeEmbeddings()
    tiler = FakeTiler(result=sample_tile_result("doc-a", n_tiles=n_tiles))
    client = make_client(store=store, embeddings=emb, tiler=tiler)
    return client, store, emb, tiler


def _pdf_files():
    return {"document": ("f.pdf", b"%PDF-bytes", "application/pdf")}


def test_point_id_is_deterministic_and_unique():
    assert point_id("doc-a", 0) == point_id("doc-a", 0)
    assert point_id("doc-a", 0) != point_id("doc-a", 1)
    assert point_id("doc-a", 0) != point_id("doc-b", 0)


def test_enroll_success(make_client):
    client, store, emb, _ = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    body = r.json()
    assert body == {
        "doc_id": "doc-a", "doc_type": "PDF", "render_version": "rv-test",
        "tile_count": 2, "enrolled": 2,
    }

    assert len(store.upserts) == 1
    up = store.upserts[0]
    assert up["ids"] == [point_id("doc-a", 0), point_id("doc-a", 1)]     # deterministic ids
    assert len(emb.image_calls[0]) == 2                                  # both tiles embedded
    assert up["payloads"][0]["metadata"]["name"] == "Report A"
    assert up["payloads"][0]["image_hash"]                              # carried for integrity


def test_enroll_missing_name_is_422(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"project": "x"})},
        files=_pdf_files(),
    )

    assert r.status_code == 422
    assert store.upserts == []


def test_enroll_allow_missing_name_uses_doc_id(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll?allow_missing_name=true",
        data={"doc_type": "PDF"},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert store.upserts[0]["payloads"][0]["metadata"]["name"] == "doc-a"


def test_enroll_zero_tiles_enrolls_nothing(make_client):
    client, store, emb, _ = _client(make_client, n_tiles=0)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Empty"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert r.json()["enrolled"] == 0
    assert store.upserts == []
    assert emb.image_calls == []


def test_enroll_bad_metadata_json_is_422(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": "{not json"},
        files=_pdf_files(),
    )

    assert r.status_code == 422
    assert store.upserts == []
