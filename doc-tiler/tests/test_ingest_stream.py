"""Streaming ingest: NDJSON tile events, terminal committed, incomplete gating + sweep."""

from __future__ import annotations

import base64
import hashlib
import json

from conftest import base_env, pdf_form, running_app, make_pdf


async def test_stream_sliced_round_trip(tmp_path):
    """Streaming ingest with PDF slicing: strips stream with a contiguous global
    tile_num (consumed in page order) and each regenerates byte-identically."""
    pdf = make_pdf(pages=4)
    env = base_env(
        tmp_path,
        TILER_PDF_SLICE_ENABLE="true",
        TILER_PDF_TILE_HEIGHT="500",
        TILER_PDF_TILE_OVERLAP="100",
    )
    async with running_app(env) as (client, _service):
        r = await client.post("/tile/stream", **pdf_form(pdf))
        assert r.status_code == 200, r.text
        events = _events(r.content)

        tiles = [e for e in events if e["event"] == "tile"]
        committed = [e for e in events if e["event"] == "committed"]
        assert len(committed) == 1
        assert committed[0]["tile_count"] == len(tiles)
        assert len(tiles) > 4  # slicing → more strips than pages

        # tile_num is a contiguous global index in emission (page) order
        assert [t["tile_num"] for t in tiles] == list(range(len(tiles)))

        for t in tiles:
            data = base64.b64decode(t["data"])
            assert hashlib.sha256(data).hexdigest() == t["image_hash"]
            got = await client.get(
                "/get_tile", params={"doc_id": committed[0]["doc_id"], "tile_num": t["tile_num"]}
            )
            assert got.status_code == 200, got.text
            assert hashlib.sha256(got.content).hexdigest() == t["image_hash"]


def _events(body: bytes) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _render_count(service, doc_type: str, path: str) -> float:
    from tiler import metrics

    return metrics.RENDERS.labels(doc_type=doc_type, path=path)._value.get()


async def test_stream_emits_tiles_then_committed(client, sample_pdf):
    r = await client.post("/tile/stream", **pdf_form(sample_pdf))
    assert r.status_code == 200, r.text

    events = _events(r.content)
    tiles = [e for e in events if e["event"] == "tile"]
    committed = [e for e in events if e["event"] == "committed"]

    assert len(tiles) == 3
    assert len(committed) == 1
    assert events[-1]["event"] == "committed"          # committed is terminal

    c = committed[0]
    assert c["tile_count"] == 3
    assert c["doc_id"] == hashlib.sha256(sample_pdf).hexdigest()

    for t in tiles:
        data = base64.b64decode(t["data"])
        assert hashlib.sha256(data).hexdigest() == t["image_hash"]
        assert data[:3] == b"\xff\xd8\xff"             # JPEG

    # committed ⇒ doc is complete ⇒ tiles are servable.
    got = await client.get("/get_tile", params={"doc_id": c["doc_id"], "tile_num": 0})
    assert got.status_code == 200


async def test_stream_idempotent_replay_does_not_re_render(client, sample_pdf):
    _events((await client.post("/tile/stream", **pdf_form(sample_pdf))).content)
    before = _render_count(client.service, "PDF", "ingest")

    second = _events((await client.post("/tile/stream", **pdf_form(sample_pdf))).content)
    committed = [e for e in second if e["event"] == "committed"][0]

    assert committed["tile_count"] == 3
    assert _render_count(client.service, "PDF", "ingest") == before   # replay from store


async def test_incomplete_doc_is_hidden_and_swept(client, sample_pdf):
    svc = client.service
    doc_id = "d" * 64

    # Simulate a crashed stream: a begun-but-never-finished ingest.
    rel = await svc.blobs.put(doc_id, sample_pdf)
    await svc.store.begin_doc(
        doc_id=doc_id, doc_type="PDF", blob_rel_path=rel, source_url=None,
        page_height=0, tile_count=0, mime_type="application/pdf",
        metadata=None, render_version=svc.settings.render_version,
    )

    # An incomplete doc is not servable.
    got = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})
    assert got.status_code == 404
    assert got.json()["error"]["code"] == "DOC_INCOMPLETE"

    # The sweeper reaps it (negative ttl → cutoff in the future, so it always matches).
    removed = await svc.sweep_incomplete(-100)
    assert removed == 1
    assert await svc.store.get_doc(doc_id) is None
    assert not svc.blobs.exists(doc_id)


async def test_sweeper_skips_in_flight(client, sample_pdf):
    svc = client.service
    doc_id = "e" * 64

    rel = await svc.blobs.put(doc_id, sample_pdf)
    await svc.store.begin_doc(
        doc_id=doc_id, doc_type="PDF", blob_rel_path=rel, source_url=None,
        page_height=0, tile_count=0, mime_type="application/pdf",
        metadata=None, render_version=svc.settings.render_version,
    )

    svc._inflight.add(doc_id)                           # pretend the stream is live
    try:
        assert await svc.sweep_incomplete(-100) == 0    # in-flight is never reaped
        assert await svc.store.get_doc(doc_id) is not None
    finally:
        svc._inflight.discard(doc_id)
