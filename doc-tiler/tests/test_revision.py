"""Revision: stored docs are brought to the current render_version before the tiler serves."""

from __future__ import annotations

import asyncio
import os
import sqlite3

import pytest
from conftest import base_env, make_pdf, make_png, pdf_form, png_form, running_app


def _render_count(doc_type: str, path: str) -> float:
    from tiler import metrics

    return metrics.RENDERS.labels(doc_type=doc_type, path=path)._value.get()


def _age_volume(db_path: str) -> None:
    """Pretend every stored doc — and the volume — was rendered by an older render_version."""
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE docs SET render_version = 'old'")
        db.execute("UPDATE service_meta SET render_version = 'old'")


async def _revised(state) -> dict:
    """Wait for the background revision to finish and return its final status."""
    if state.revision._task is not None:
        await asyncio.wait_for(asyncio.shield(state.revision._task), 60)

    return state.revision.status


async def test_startup_revision_rerenders_stale_docs(tmp_path):
    env = base_env(tmp_path)
    async with running_app(env) as (client, _):
        pdf_id = (await client.post("/tile", **pdf_form(make_pdf(), metadata='{"project": "a"}'))).json()["doc_id"]
        png_id = (await client.post("/tile", **png_form(make_png()))).json()["doc_id"]
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (client, state):
        status = await _revised(state)
        assert status["state"] == "succeeded" and status["serving"] is True
        assert (status["total"], status["revised"], status["quarantined"]) == (2, 2, 0)

        assert os.path.exists(env["TILER_DB_PATH"] + ".bak-old")      # rollback point
        assert (await state.store.get_service_meta()).render_version == state.settings.render_version

        for doc_id in (pdf_id, png_id):
            row = await state.store.get_doc(doc_id)
            assert row.render_version == state.settings.render_version
            got = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})
            assert got.status_code == 200, got.text                 # verified against the new hash

        assert (await state.store.get_doc(pdf_id)).metadata == {"project": "a"}  # untouched

    async with running_app(env) as (client, state):                       # nothing left to revise
        assert state.revision.serving and state.revision._task is None


async def test_data_routes_wait_for_revision(tmp_path):
    """With the auto revision off, the tiler stays up but serves nothing until POST /revision."""
    env = base_env(tmp_path, TILER_REVISION_ON_RENDER_MISMATCH="false")
    async with running_app(env) as (client, _):
        doc_id = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()["doc_id"]
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (client, state):
        status = (await client.get("/revision")).json()
        assert (status["state"], status["serving"], status["total"]) == ("required", False, 1)

        got = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})
        assert got.status_code == 503
        assert got.json()["error"]["code"] == "NOT_READY"
        assert got.headers["retry-after"]
        assert (await client.post("/docs/search", json={})).status_code == 503
        assert (await client.get("/readyz")).status_code == 503

        for path in ("/healthz", "/info", "/schema/metadata", "/metrics"):    # ops stay up
            assert (await client.get(path)).status_code == 200, path

        started = await client.post("/revision")
        assert started.status_code == 202 and started.json()["state"] == "running"
        assert (await _revised(state))["state"] == "succeeded"

        assert (await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})).status_code == 200
        assert (await client.get("/readyz")).status_code == 200


async def test_unrenderable_doc_is_quarantined(tmp_path):
    """A doc that cannot be re-rendered is quarantined instead of holding the service back,
    and re-ingesting its bytes brings it back."""
    env = base_env(tmp_path)
    good, bad = make_png(875, 1200), make_png(875, 1300)
    async with running_app(env) as (client, state):
        good_id = (await client.post("/tile", **png_form(good))).json()["doc_id"]
        bad_id = (await client.post("/tile", **png_form(bad))).json()["doc_id"]
        await state.blobs.delete(bad_id)                                   # its original is lost
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (client, state):
        status = await _revised(state)
        assert status["serving"] is True
        assert (status["revised"], status["quarantined"]) == (1, 1)
        assert status["quarantined_doc_ids"] == [bad_id]

        assert (await client.get("/get_tile", params={"doc_id": good_id, "tile_num": 0})).status_code == 200
        gone = await client.get("/get_tile", params={"doc_id": bad_id, "tile_num": 0})
        assert gone.status_code == 410
        assert gone.json()["error"]["code"] == "QUARANTINED"

        gone = await client.get(f"/doc/{bad_id}/tiles")
        assert (gone.status_code, gone.json()["error"]["code"]) == (410, "QUARANTINED")

        listed = (await client.post("/docs/search", json={})).json()["items"]
        assert {d["doc_id"]: d["quarantined"] for d in listed} == {good_id: False, bad_id: True}
        assert (await client.get("/readyz")).json()["checks"]["quarantined"] == 1

        again = await client.post("/tile", **png_form(bad))                # re-ingest lifts it
        assert again.status_code == 200
        assert (await client.get("/get_tile", params={"doc_id": bad_id, "tile_num": 0})).status_code == 200
        assert await state.store.count_quarantined() == 0
        assert (await client.get("/readyz")).json()["checks"]["quarantined"] == 0   # counted live


async def test_quarantined_doc_is_not_retried_at_the_same_render_version(tmp_path):
    env = base_env(tmp_path)
    async with running_app(env) as (client, state):
        doc_id = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()["doc_id"]
        await state.blobs.delete(doc_id)
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (_, state):
        assert (await _revised(state))["quarantined"] == 1

    async with running_app(env) as (_, state):                            # restart: no new job
        assert state.revision.serving and state.revision._task is None


async def test_force_revision_rerenders_every_doc(client, sample_pdf):
    state = client.state
    doc_id = (await client.post("/tile", **pdf_form(sample_pdf))).json()["doc_id"]
    before = _render_count("PDF", "revise")

    r = await client.post("/revision", params={"force": "true"})
    assert r.status_code == 202, r.text
    assert r.json()["force"] is True

    status = await _revised(state)
    assert (status["state"], status["revised"]) == ("succeeded", 1)
    assert _render_count("PDF", "revise") == before + 1
    assert (await state.store.get_doc(doc_id)).tile_count == 3


async def test_second_revision_is_refused_while_one_runs(client, sample_pdf):
    from tiler.errors import ApiError

    await client.post("/tile", **pdf_form(sample_pdf))
    client.state.revision.start(force=True)

    with pytest.raises(ApiError) as refused:
        client.state.revision.start()
    assert (refused.value.status_code, refused.value.code) == (409, "REVISION_RUNNING")

    await _revised(client.state)


async def test_an_abandoned_reingest_leaves_a_quarantined_doc_as_it_was(tmp_path):
    """Nothing of a stored doc changes before an ingest of its bytes commits."""
    from tiler.models import DocType

    env = base_env(tmp_path)
    png = make_png(875, 3000)
    async with running_app(env) as (client, state):
        doc_id = (await client.post("/tile", **png_form(png))).json()["doc_id"]
        await state.blobs.delete(doc_id)                                  # its original is lost
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (client, state):
        assert (await _revised(state))["quarantined"] == 1
        before = await state.store.get_doc(doc_id)

        stream = state.ingester.ingest_stream(
            doc_type=DocType.IMG, upload=await state.blobs.spool(png), url=None, metadata=None
        )
        assert (await anext(stream))["event"] == "begin"
        await stream.aclose()                                           # abandoned before it commits

        assert await state.store.get_doc(doc_id) == before                # still there, still quarantined
        assert before.quarantined and before.complete
        assert len(await state.store.get_tiles(doc_id)) == before.tile_count

        again = await client.post("/tile", **png_form(png))             # a finished one lifts it
        assert again.status_code == 200 and again.json()["result"] == "rendered"
        assert await state.store.count_quarantined() == 0
        assert (await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})).status_code == 200


async def test_revision_keeps_pdf_strip_slicing(tmp_path):
    """A revised sliced PDF keeps its strip layout: the height and the tile count are
    unchanged and every strip still regenerates to its stored hash."""
    env = base_env(
        tmp_path,
        TILER_PDF_SLICE_ENABLE="true",
        TILER_PDF_TILE_HEIGHT="500",
        TILER_PDF_TILE_OVERLAP="100",
    )
    async with running_app(env) as (client, _):
        manifest = (await client.post("/tile", **pdf_form(make_pdf(pages=3)))).json()
    _age_volume(env["TILER_DB_PATH"])

    async with running_app(env) as (client, state):
        assert (await _revised(state))["revised"] == 1

        row = await state.store.get_doc(manifest["doc_id"])
        assert row.height == manifest["height"]
        assert row.tile_count == manifest["tile_count"] > 3

        for tile_num in range(row.tile_count):
            got = await client.get("/get_tile", params={"doc_id": row.doc_id, "tile_num": tile_num})
            assert got.status_code == 200, got.text                    # verified against the stored hash


async def test_bulk_slots_default_to_half_the_pools(tmp_path):
    from tiler.config import Settings

    s = Settings(n_render_workers=8, n_browsers=3)
    assert (s.bulk_render_workers, s.bulk_browsers) == (4, 1)
    assert Settings(n_render_workers=1).bulk_render_workers == 1
    assert Settings(n_render_workers=8, n_bulk_render_workers=6).bulk_render_workers == 6
