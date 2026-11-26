"""/enroll: the upload relayed to the tiler, name policy, deterministic ids, embed+upsert,
zero-tile handling, one enroll of a document at a time."""

from __future__ import annotations

import json

from virag.clients.tiler_client import Begin
from virag.enroll import point_id
from virag.errors import ApiError

from virag.settings import EnrollSettings

from conftest import (
    FakeRetriever,
    FakeStore,
    FakeTiler,
    lease_key,
    make_coordinator,
    make_settings,
    sample_tile_result,
)


def _client(make_client, *, n_tiles=2, batch_size=16):
    store, retriever = FakeStore(), FakeRetriever()
    tiler = FakeTiler(result=sample_tile_result("doc-a", n_tiles=n_tiles))
    settings = make_settings(enroll_settings=EnrollSettings(batch_size=batch_size))
    client = make_client(settings=settings, store=store, retriever=retriever, tiler=tiler)
    return client, store, retriever, tiler


def _pdf_files():
    return {"document": ("f.pdf", b"%PDF-bytes", "application/pdf")}


def _enroll(client):
    return client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )


def _held(client) -> dict:
    """doc_id -> the token holding it, as Valkey has it."""
    values = client.app.state.coordinator.valkey.values

    return {name.removeprefix("enroll:"): value for name, (value, _) in values.items() if name.startswith("enroll:")}


def test_point_id_is_deterministic_and_unique():
    assert point_id("doc-a", 0) == point_id("doc-a", 0)
    assert point_id("doc-a", 0) != point_id("doc-a", 1)
    assert point_id("doc-a", 0) != point_id("doc-b", 0)


def test_enroll_success(make_client):
    client, store, retriever, _ = _client(make_client)

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
    assert len(retriever.image_calls[0]) == 2                                  # both tiles embedded
    # A point carries what fetching and checking its tile takes, and no metadata.
    assert set(up["payloads"][0]) == {"doc_id", "tile_num", "image_hash", "render_version"}
    assert up["payloads"][0]["image_hash"]                              # carried for integrity
    assert _held(client) == {}                                          # the document is released


def test_enroll_missing_name_is_422(make_client):
    client, store, _, tiler = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"project": "x"})},
        files=_pdf_files(),
    )

    assert r.status_code == 422
    assert store.upserts == []
    assert tiler.uploads == []                                          # the tiler never got the form whole


def test_enroll_without_any_metadata_is_422(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post("/enroll", data={"doc_type": "PDF"}, files=_pdf_files())

    assert r.status_code == 422 and "metadata.name" in r.json()["error"]["message"]
    assert store.upserts == []


def test_enroll_that_is_not_a_multipart_form_is_422(make_client):
    client, store, _, tiler = _client(make_client)

    r = client.post("/enroll", json={"doc_type": "PDF"})

    assert r.status_code == 422 and "multipart/form-data" in r.json()["error"]["message"]
    assert tiler.uploads == [] and store.upserts == []


def test_enroll_allow_missing_name_uses_doc_id(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll?allow_missing_name=true",
        data={"doc_type": "PDF"},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert len(store.upserts) == 1


def test_enroll_zero_tiles_enrolls_nothing(make_client):
    client, store, retriever, _ = _client(make_client, n_tiles=0)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Empty"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert r.json()["enrolled"] == 0
    assert store.upserts == []
    assert retriever.image_calls == []


def test_enroll_bad_metadata_json_is_422(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": "{not json"},
        files=_pdf_files(),
    )

    assert r.status_code == 422
    assert store.upserts == []


def test_enroll_marks_pending_then_promotes(make_client):
    client, store, _, tiler = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert store.deletes == ["doc-a"]                                # cleared prior partial on start
    # first patch marks pending (enrolled_at 0), last patch promotes (enrolled_at > 0)
    assert tiler.metadata_patches[0][1]["virag_enrolled_at"] == 0
    assert tiler.metadata_patches[-1][1]["virag_enrolled_at"] > 0
    # the metadata lives in the tiler: the pending patch carries it, so a re-enroll updates it
    assert tiler.metadata_patches[0][1]["name"] == "Report A"


def test_enroll_streams_in_batches(make_client):
    client, store, retriever, _ = _client(make_client, n_tiles=3, batch_size=1)   # one tile per batch

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert r.json()["enrolled"] == 3
    assert len(store.upserts) == 3                                   # one upsert per streamed batch


def test_enroll_stream_error_rolls_back(make_client):
    client, store, _, tiler = _client(make_client)

    async def bad_stream(body, **_):
        yield Begin(doc_id="doc-a", doc_type="PDF", render_version="rv-test")
        raise ApiError(status_code=502, message="tiler stream error: boom", details={"tiler_code": "INTERNAL"})

    tiler.ingest_stream = bad_stream

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 502
    assert store.upserts == []
    assert store.deletes == ["doc-a", "doc-a"]                       # start clear + rollback


def test_enroll_relays_a_document_the_tiler_refuses(make_client):
    client, store, _, tiler = _client(make_client)

    async def refusing_stream(body, **_):
        raise ApiError(status_code=415, message="The tiler refused the document: not a PDF")
        yield  # pragma: no cover - makes this an async generator

    tiler.ingest_stream = refusing_stream

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert (r.status_code, r.json()["error"]["code"]) == (415, "UNSUPPORTED_MEDIA")
    assert store.upserts == [] and store.deletes == []                # nothing was begun
    assert tiler.metadata_patches == []


def test_enroll_rolls_back_on_upsert_failure(make_client):
    client, store, _, tiler = _client(make_client)

    async def boom(*a, **k):
        raise RuntimeError("qdrant down")

    store.upsert = boom

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 500
    assert store.deletes == ["doc-a", "doc-a"]                        # start clear + rollback
    # pending is never promoted on failure — no patch sets a non-zero enrolled_at
    assert all(p[1].get("virag_enrolled_at") == 0 for p in tiler.metadata_patches)


def test_enroll_writes_into_the_collection_pinned_at_begin(make_client):
    client, store, _, _ = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 200
    assert store.upserts[0]["collection"] == "c__v1"
    assert store.delete_collections == ["c__v1"]                      # the start clear, same one


def test_enroll_without_valkey_is_503_and_keeps_existing_points(make_client):
    client, store, _, tiler = _client(make_client)
    client.app.state.coordinator.valkey.down = True

    r = client.post(
        "/enroll",
        data={"doc_type": "PDF", "metadata": json.dumps({"name": "Report A"})},
        files=_pdf_files(),
    )

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "NOT_READY"
    assert store.deletes == []                                        # an earlier enroll's points stay
    assert tiler.metadata_patches == []


# ---------------------------------------------------------------------------
# one enroll of a document at a time
# ---------------------------------------------------------------------------

def test_enroll_of_a_document_being_enrolled_is_409(make_client):
    client, store, _, tiler = _client(make_client)
    coordinator = client.app.state.coordinator
    other = make_coordinator(store, valkey=coordinator.valkey)        # another process…
    coordinator.valkey.values["enroll:doc-a"] = (other.token, None)   # …is enrolling the doc

    r = _enroll(client)

    assert (r.status_code, r.json()["error"]["code"]) == (409, "ENROLL_IN_PROGRESS")
    assert store.deletes == [] and store.upserts == []                # nothing of the doc is touched
    assert tiler.metadata_patches == []
    assert _held(client) == {"doc-a": other.token}                    # and it stays the other's


def test_enroll_takes_over_a_document_of_a_dead_process(make_client):
    client, store, _, _ = _client(make_client)
    coordinator = client.app.state.coordinator
    dead = make_coordinator(store, valkey=coordinator.valkey)
    coordinator.valkey.values["enroll:doc-a"] = (dead.token, None)
    coordinator.valkey.expire(lease_key(dead))                        # it died mid-enroll

    r = _enroll(client)

    assert r.status_code == 200 and r.json()["enrolled"] == 2
    assert _held(client) == {}


def test_a_failed_enroll_releases_the_document(make_client):
    client, store, _, _ = _client(make_client)

    async def boom(*a, **k):
        raise RuntimeError("qdrant down")

    store.upsert = boom

    assert _enroll(client).status_code == 500
    assert _held(client) == {}


def test_a_failed_enroll_that_lost_the_document_deletes_nothing(make_client):
    """A process that stalled past its lease: another one enrolls the document meanwhile,
    and the stale enroll's rollback must leave the points that one wrote."""
    client, store, _, _ = _client(make_client)
    coordinator = client.app.state.coordinator
    other = make_coordinator(store, valkey=coordinator.valkey)

    async def boom(*a, **k):
        coordinator.valkey.expire(lease_key(coordinator))                # stalled past the lease…
        coordinator.valkey.values["enroll:doc-a"] = (other.token, None)  # …and the doc was taken over
        raise RuntimeError("tiler lost mid-stream")

    store.upsert = boom

    r = _enroll(client)

    assert r.status_code == 500
    assert store.deletes == ["doc-a"]                                 # the start clear; no rollback delete
    assert _held(client) == {"doc-a": other.token}


def test_enroll_that_cannot_mark_pending_keeps_existing_points(make_client):
    client, store, _, tiler = _client(make_client)

    async def unavailable(doc_id, patch):
        raise ApiError(status_code=503, message="tiler is unavailable")

    tiler.patch_metadata = unavailable

    r = _enroll(client)

    assert r.status_code == 503
    assert store.deletes == []                                        # the doc still reads as enrolled: its points stay
    assert _held(client) == {}


# ---------------------------------------------------------------------------
# the upload is relayed, not held
# ---------------------------------------------------------------------------

def _form(*parts, boundary="xXx") -> tuple[bytes, str]:
    """A multipart body with its parts in the given order: (name, value) for a text field,
    (name, filename, bytes) for a file. Returns the body and its Content-Type."""
    body = b""

    for name, *rest in parts:
        filename = f'; filename="{rest[0]}"' if len(rest) == 2 else ""
        value = rest[-1] if isinstance(rest[-1], bytes) else rest[-1].encode()
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"{filename}\r\n\r\n'.encode()
        body += value + b"\r\n"

    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def test_enroll_relays_the_form_to_the_tiler_as_it_came(make_client):
    client, _, _, tiler = _client(make_client)
    body, content_type = _form(
        ("doc_type", "PDF"), ("metadata", json.dumps({"name": "Report A"})), ("document", "r.pdf", b"%PDF-bytes"),
    )

    r = client.post("/enroll", content=body, headers={"Content-Type": content_type})

    assert r.status_code == 200
    assert tiler.uploads == [{"body": body, "content_type": content_type, "content_length": str(len(body))}]
    assert tiler.metadata_patches[0][1]["name"] == "Report A"            # read on the way, for the tiler's patch


def test_enroll_reads_the_metadata_wherever_it_comes_in_the_form(make_client):
    client, store, _, tiler = _client(make_client)
    named, content_type = _form(
        ("document", "r.pdf", b"%PDF-bytes"), ("doc_type", "PDF"), ("metadata", json.dumps({"name": "Last"})),
    )
    unnamed, _ = _form(("document", "r.pdf", b"%PDF-bytes"), ("doc_type", "PDF"), ("metadata", "{}"))

    ok = client.post("/enroll", content=named, headers={"Content-Type": content_type})
    refused = client.post("/enroll", content=unnamed, headers={"Content-Type": content_type})

    assert ok.status_code == 200 and tiler.metadata_patches[0][1]["name"] == "Last"
    assert refused.status_code == 422
    assert len(tiler.uploads) == 1                                      # the unnamed one never went through whole


def test_enroll_with_a_malformed_form_is_422(make_client):
    client, store, _, tiler = _client(make_client)

    r = client.post(
        "/enroll",
        content=b"--xXx\r\nthis is not a part header\r\n\r\n--xXx--\r\n",
        headers={"Content-Type": "multipart/form-data; boundary=xXx"},
    )

    assert r.status_code == 422 and "malformed" in r.json()["error"]["message"]
    assert tiler.uploads == [] and store.upserts == []


def test_enroll_by_url_needs_no_document(make_client):
    client, _, _, tiler = _client(make_client)

    r = client.post(
        "/enroll",
        data={"doc_type": "URL", "url": "https://example.com", "metadata": json.dumps({"name": "Example"})},
        files={"placeholder": (None, "")},                              # keeps the body multipart
    )

    assert r.status_code == 200
    assert b"https://example.com" in tiler.uploads[0]["body"]
