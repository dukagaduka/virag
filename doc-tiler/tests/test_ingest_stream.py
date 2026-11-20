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
    async with running_app(env) as (client, _state):
        r = await client.post("/doc", params={"tiles": "true"}, **pdf_form(pdf))
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
                f"/doc/{committed[0]['doc_id']}/tiles/{t['tile_num']}"
            )
            assert got.status_code == 200, got.text
            assert hashlib.sha256(got.content).hexdigest() == t["image_hash"]


def _events(body: bytes) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _render_count(state, doc_type: str, path: str) -> float:
    from tiler import metrics

    return metrics.RENDERS.labels(doc_type=doc_type, path=path)._value.get()


async def test_stream_emits_tiles_then_committed(client, sample_pdf):
    r = await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf))
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
    got = await client.get(f"/doc/{c['doc_id']}/tiles/0")
    assert got.status_code == 200


async def test_stream_idempotent_replay_does_not_re_render(client, sample_pdf):
    _events((await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf))).content)
    before = _render_count(client.state, "PDF", "ingest")

    second = _events((await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf))).content)
    committed = [e for e in second if e["event"] == "committed"][0]

    assert committed["tile_count"] == 3
    assert committed["result"] == "idempotent"
    assert len([e for e in second if e["event"] == "tile"]) == 3       # tiles are replayed
    assert _render_count(client.state, "PDF", "ingest") == before   # replay from store


async def test_a_stored_doc_streams_by_its_id(client, sample_pdf):
    """GET /doc/{id}/tiles answers what POST /doc?tiles=true answers to bytes already stored,
    in one render of the document."""
    first = _events((await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf))).content)
    replayed = _events((await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf))).content)
    doc_id = first[0]["doc_id"]
    before = _render_count(client.state, "PDF", "tiles")

    r = await client.get(f"/doc/{doc_id}/tiles")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")

    events = _events(r.content)
    assert [e["event"] for e in events] == ["begin", "tile", "tile", "tile", "committed"]
    assert events == replayed
    assert [e for e in events if e["event"] == "tile"] == [e for e in first if e["event"] == "tile"]
    assert _render_count(client.state, "PDF", "tiles") == before + 1   # not one per tile


async def test_a_doc_stream_refuses_a_doc_that_is_not_stored(client):
    for doc_id in ("0" * 64, "not-a-doc-id"):
        r = await client.get(f"/doc/{doc_id}/tiles")
        assert (r.status_code, r.json()["error"]["code"]) == (404, "UNKNOWN_DOC")


async def test_a_drifted_tile_ends_a_doc_stream_with_an_error(client, sample_pdf):
    """The stream has begun, so the tile that no longer reproduces its hash is reported
    in it — and never sent."""
    doc_id = (await client.post("/doc", **pdf_form(sample_pdf))).json()["doc_id"]
    store = client.state.store

    async with store._lock:
        await store.db.execute(
            "UPDATE tiles SET image_hash = ? WHERE doc_id = ? AND tile_num = 1", ("0" * 64, doc_id)
        )
        await store.db.commit()

    r = await client.get(f"/doc/{doc_id}/tiles")
    assert r.status_code == 200

    events = _events(r.content)
    assert [e["event"] for e in events] == ["begin", "tile", "error"]
    assert events[-1]["code"] == "RENDER_MISMATCH"


def _spooled(state) -> list:
    return sorted(state.blobs.incoming_dir.iterdir())


async def _stream(state, data: bytes):
    """The ingester's event stream for a PDF upload, driven by hand."""
    upload = await state.blobs.spool(data)

    return state.ingester.ingest_stream(doc_type=DocType.PDF, upload=upload, url=None, metadata=None)


async def test_a_doc_is_incomplete_until_its_ingest_commits(client, sample_pdf):
    state = client.state
    doc_id = hashlib.sha256(sample_pdf).hexdigest()
    stream = await _stream(state, sample_pdf)

    assert (await anext(stream))["event"] == "begin"

    # The row is there, so a consumer can refer to the doc; nothing of it is served yet.
    assert (await state.store.get_doc(doc_id)).complete is False
    assert not state.blobs.exists(doc_id) and len(_spooled(state)) == 1    # the upload waits in .incoming

    for route in (f"/doc/{doc_id}/tiles/0", f"/doc/{doc_id}", f"/doc/{doc_id}/tiles"):
        got = await client.get(route)
        assert got.status_code == 404
        assert got.json()["error"]["code"] == "DOC_INCOMPLETE"

    listed = (await client.post("/docs/search", json={})).json()["items"]
    assert [(d["doc_id"], d["complete"]) for d in listed] == [(doc_id, False)]

    # Its ingest owns it: the same bytes again, or a delete, are refused meanwhile.
    for tiles in ("false", "true"):
        late = await client.post("/doc", params={"tiles": tiles}, **pdf_form(sample_pdf))
        assert (late.status_code, late.json()["error"]["code"]) == (409, "INGEST_IN_PROGRESS")

    gone = await client.delete(f"/doc/{doc_id}")
    assert (gone.status_code, gone.json()["error"]["code"]) == (409, "INGEST_IN_PROGRESS")
    assert len(_spooled(state)) == 1                                     # the latecomer's upload is gone

    events = [event async for event in stream]
    assert events[-1]["event"] == "committed"

    assert (await state.store.get_doc(doc_id)).complete is True
    assert state.blobs.exists(doc_id) and _spooled(state) == []            # moved into the store
    assert (await client.get(f"/doc/{doc_id}/tiles/0")).status_code == 200


async def test_what_fails_before_begin_is_an_http_error(client, sample_pdf):
    """The stream has not started, so the failure keeps its status; nothing is kept."""
    state = client.state

    files = {"document": ("x.pdf", b"this is not a pdf", "application/pdf")}
    wrong = await client.post("/doc", params={"tiles": "true"}, files=files, data={"doc_type": "PDF"})
    assert (wrong.status_code, wrong.json()["error"]["code"]) == (415, "UNSUPPORTED_MEDIA")

    invalid = await client.post("/doc", params={"tiles": "true"}, **pdf_form(sample_pdf, metadata='{"year": "soon"}'))
    assert (invalid.status_code, invalid.json()["error"]["code"]) == (422, "METADATA_SCHEMA_VIOLATION")

    assert await state.store.count_docs() == 0
    assert _spooled(state) == []


async def test_an_abandoned_ingest_leaves_nothing(client):
    state = client.state
    pdf = make_pdf(pages=40)
    doc_id = hashlib.sha256(pdf).hexdigest()
    stream = await _stream(state, pdf)

    async for event in stream:
        if event["event"] == "tile":
            break

    await stream.aclose()                                              # the client went away

    assert await state.store.get_doc(doc_id) is None
    assert not state.blobs.exists(doc_id) and _spooled(state) == []

    again = await client.post("/doc", **pdf_form(pdf))                # the bytes are welcome again
    assert again.status_code == 200 and again.json()["result"] == "rendered"


async def test_a_cancelled_ingest_leaves_nothing(client):
    """How a dropped connection ends a stream: its task is cancelled, and the
    cancellation keeps hitting every await made after it."""
    state = client.state
    pdf = make_pdf(pages=40, height=843)
    doc_id = hashlib.sha256(pdf).hexdigest()
    stream = await _stream(state, pdf)

    with anyio.CancelScope() as scope:
        async for event in stream:
            if event["event"] == "tile":
                scope.cancel()

    assert scope.cancelled_caught
    assert await state.store.get_doc(doc_id) is None
    assert not state.blobs.exists(doc_id) and _spooled(state) == []


async def test_a_failed_render_leaves_nothing(client):
    state = client.state
    broken = b"%PDF-1.7\n" + b"this is not a pdf at all " * 64
    doc_id = hashlib.sha256(broken).hexdigest()

    r = await client.post("/doc", **pdf_form(broken))
    assert (r.status_code, r.json()["error"]["code"]) == (422, "DECODE_FAILED")

    events = _events((await client.post("/doc", params={"tiles": "true"}, **pdf_form(broken))).content)
    assert [e["event"] for e in events] == ["begin", "error"]
    assert events[-1]["code"] == "DECODE_FAILED"

    assert await state.store.get_doc(doc_id) is None
    assert not state.blobs.exists(doc_id) and _spooled(state) == []


async def test_an_upload_is_hashed_as_it_is_spooled(client, sample_pdf):
    state = client.state

    async def chunks():
        for i in range(0, len(sample_pdf), 100):
            yield sample_pdf[i:i + 100]

    upload = await state.blobs.spool(chunks())

    assert upload.doc_id == hashlib.sha256(sample_pdf).hexdigest()
    assert upload.size == len(sample_pdf) and upload.path.read_bytes() == sample_pdf
    assert await state.blobs.head(upload, 5) == b"%PDF-"

    await state.blobs.discard(upload)
    assert _spooled(state) == []


async def test_an_upload_that_breaks_off_is_not_kept(client):
    state = client.state

    async def chunks():
        yield b"x" * 1000
        raise RuntimeError("the connection dropped")

    with pytest.raises(RuntimeError):
        await state.blobs.spool(chunks())

    assert _spooled(state) == []


async def test_startup_removes_what_a_killed_process_left(tmp_path, sample_pdf):
    env = base_env(tmp_path)
    doc_id = "d" * 64

    async with running_app(env) as (client, state):
        kept = (await client.post("/doc", **pdf_form(sample_pdf))).json()["doc_id"]

        # A process killed mid-ingest: the row it inserted and the upload it spooled.
        await state.store.begin_doc(
            doc_id=doc_id, doc_type="PDF", blob_rel_path=state.blobs.rel_path(doc_id), source_url=None,
            mime_type="application/pdf", metadata=None, render_version=state.settings.render_version,
        )
        await state.blobs.spool(make_pdf(pages=1))

    async with running_app(env) as (client, state):
        assert await state.store.get_doc(doc_id) is None
        assert _spooled(state) == []
        assert (await client.get(f"/doc/{kept}/tiles/0")).status_code == 200


async def test_two_inserts_of_one_doc_let_one_through(client):
    store = client.state.store
    row = dict(
        doc_id="e" * 64, doc_type="PDF", blob_rel_path="x", source_url=None,
        mime_type="application/pdf", metadata=None, render_version="rv",
    )

    assert await store.begin_doc(**row) is True
    assert await store.begin_doc(**row) is False
