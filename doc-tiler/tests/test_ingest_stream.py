"""Streaming ingest: NDJSON tile events, terminal committed, and what an ingest that
does not commit leaves behind (nothing)."""

from __future__ import annotations

import base64
import hashlib
import json

import anyio
import pytest
from conftest import base_env, pdf_form, running_app, make_pdf

from tiler.models import DocType


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
    assert c["result"] == "rendered"
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
    assert committed["result"] == "idempotent"
    assert len([e for e in second if e["event"] == "tile"]) == 3       # tiles are replayed
    assert _render_count(client.service, "PDF", "ingest") == before   # replay from store


def _spooled(svc) -> list:
    return sorted(svc.blobs.incoming_dir.iterdir())


async def _stream(svc, data: bytes):
    """The service's event stream for a PDF upload, driven by hand."""
    upload = await svc.blobs.spool(data)

    return svc.ingest_stream(doc_type=DocType.PDF, upload=upload, url=None, metadata=None)


async def test_a_doc_is_incomplete_until_its_ingest_commits(client, sample_pdf):
    svc = client.service
    doc_id = hashlib.sha256(sample_pdf).hexdigest()
    stream = await _stream(svc, sample_pdf)

    assert (await anext(stream))["event"] == "begin"

    # The row is there, so a consumer can refer to the doc; nothing of it is served yet.
    assert (await svc.store.get_doc(doc_id)).complete is False
    assert not svc.blobs.exists(doc_id) and len(_spooled(svc)) == 1    # the upload waits in .incoming

    for route, params in (("/get_tile", {"doc_id": doc_id, "tile_num": 0}), (f"/doc/{doc_id}", None)):
        got = await client.get(route, params=params)
        assert got.status_code == 404
        assert got.json()["error"]["code"] == "DOC_INCOMPLETE"

    listed = (await client.post("/docs/search", json={})).json()["items"]
    assert [(d["doc_id"], d["complete"]) for d in listed] == [(doc_id, False)]

    # Its ingest owns it: the same bytes again, or a delete, are refused meanwhile.
    for route in ("/tile", "/tile/stream"):
        late = await client.post(route, **pdf_form(sample_pdf))
        assert (late.status_code, late.json()["error"]["code"]) == (409, "INGEST_IN_PROGRESS")

    gone = await client.delete(f"/doc/{doc_id}")
    assert (gone.status_code, gone.json()["error"]["code"]) == (409, "INGEST_IN_PROGRESS")
    assert len(_spooled(svc)) == 1                                     # the latecomer's upload is gone

    events = [event async for event in stream]
    assert events[-1]["event"] == "committed"

    assert (await svc.store.get_doc(doc_id)).complete is True
    assert svc.blobs.exists(doc_id) and _spooled(svc) == []            # moved into the store
    assert (await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})).status_code == 200


async def test_what_fails_before_begin_is_an_http_error(client, sample_pdf):
    """The stream has not started, so the failure keeps its status; nothing is kept."""
    svc = client.service

    files = {"document": ("x.pdf", b"this is not a pdf", "application/pdf")}
    wrong = await client.post("/tile/stream", files=files, data={"doc_type": "PDF"})
    assert (wrong.status_code, wrong.json()["error"]["code"]) == (415, "UNSUPPORTED_MEDIA")

    invalid = await client.post("/tile/stream", **pdf_form(sample_pdf, metadata='{"year": "soon"}'))
    assert (invalid.status_code, invalid.json()["error"]["code"]) == (422, "METADATA_SCHEMA_VIOLATION")

    assert await svc.store.count_docs() == 0
    assert _spooled(svc) == []


async def test_an_abandoned_ingest_leaves_nothing(client):
    svc = client.service
    pdf = make_pdf(pages=40)
    doc_id = hashlib.sha256(pdf).hexdigest()
    stream = await _stream(svc, pdf)

    async for event in stream:
        if event["event"] == "tile":
            break

    await stream.aclose()                                              # the client went away

    assert await svc.store.get_doc(doc_id) is None
    assert not svc.blobs.exists(doc_id) and _spooled(svc) == []

    again = await client.post("/tile", **pdf_form(pdf))                # the bytes are welcome again
    assert again.status_code == 200 and again.json()["result"] == "rendered"


async def test_a_cancelled_ingest_leaves_nothing(client):
    """How a dropped connection ends a stream: its task is cancelled, and the
    cancellation keeps hitting every await made after it."""
    svc = client.service
    pdf = make_pdf(pages=40, height=843)
    doc_id = hashlib.sha256(pdf).hexdigest()
    stream = await _stream(svc, pdf)

    with anyio.CancelScope() as scope:
        async for event in stream:
            if event["event"] == "tile":
                scope.cancel()

    assert scope.cancelled_caught
    assert await svc.store.get_doc(doc_id) is None
    assert not svc.blobs.exists(doc_id) and _spooled(svc) == []


async def test_a_failed_render_leaves_nothing(client):
    svc = client.service
    broken = b"%PDF-1.7\n" + b"this is not a pdf at all " * 64
    doc_id = hashlib.sha256(broken).hexdigest()

    r = await client.post("/tile", **pdf_form(broken))
    assert (r.status_code, r.json()["error"]["code"]) == (422, "DECODE_FAILED")

    events = _events((await client.post("/tile/stream", **pdf_form(broken))).content)
    assert [e["event"] for e in events] == ["begin", "error"]
    assert events[-1]["code"] == "DECODE_FAILED"

    assert await svc.store.get_doc(doc_id) is None
    assert not svc.blobs.exists(doc_id) and _spooled(svc) == []


async def test_an_upload_is_hashed_as_it_is_spooled(client, sample_pdf):
    svc = client.service

    async def chunks():
        for i in range(0, len(sample_pdf), 100):
            yield sample_pdf[i:i + 100]

    upload = await svc.blobs.spool(chunks())

    assert upload.doc_id == hashlib.sha256(sample_pdf).hexdigest()
    assert upload.size == len(sample_pdf) and upload.path.read_bytes() == sample_pdf
    assert await svc.blobs.head(upload, 5) == b"%PDF-"

    await svc.blobs.discard(upload)
    assert _spooled(svc) == []


async def test_an_upload_that_breaks_off_is_not_kept(client):
    svc = client.service

    async def chunks():
        yield b"x" * 1000
        raise RuntimeError("the connection dropped")

    with pytest.raises(RuntimeError):
        await svc.blobs.spool(chunks())

    assert _spooled(svc) == []


async def test_startup_removes_what_a_killed_process_left(tmp_path, sample_pdf):
    env = base_env(tmp_path)
    doc_id = "d" * 64

    async with running_app(env) as (client, svc):
        kept = (await client.post("/tile", **pdf_form(sample_pdf))).json()["doc_id"]

        # A process killed mid-ingest: the row it inserted and the upload it spooled.
        await svc.store.begin_doc(
            doc_id=doc_id, doc_type="PDF", blob_rel_path=svc.blobs.rel_path(doc_id), source_url=None,
            mime_type="application/pdf", metadata=None, render_version=svc.settings.render_version,
        )
        await svc.blobs.spool(make_pdf(pages=1))

    async with running_app(env) as (client, svc):
        assert await svc.store.get_doc(doc_id) is None
        assert _spooled(svc) == []
        assert (await client.get("/get_tile", params={"doc_id": kept, "tile_num": 0})).status_code == 200


async def test_two_inserts_of_one_doc_let_one_through(client):
    store = client.service.store
    row = dict(
        doc_id="e" * 64, doc_type="PDF", blob_rel_path="x", source_url=None,
        mime_type="application/pdf", metadata=None, render_version="rv",
    )

    assert await store.begin_doc(**row) is True
    assert await store.begin_doc(**row) is False
