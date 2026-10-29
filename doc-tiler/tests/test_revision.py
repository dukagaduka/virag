"""tiler /revision: re-render stale/incomplete docs in place at the current render_version."""

from __future__ import annotations

from conftest import base_env, make_pdf, pdf_form, running_app


def _render_count(service, doc_type: str, path: str) -> float:
    from tiler import metrics

    return metrics.RENDERS.labels(doc_type=doc_type, path=path)._value.get()


async def test_revision_force_re_renders_every_doc(client, sample_pdf):
    doc_id = (await client.post("/tile", **pdf_form(sample_pdf))).json()["doc_id"]
    before = _render_count(client.service, "PDF", "revise")

    r = await client.post("/revision", params={"force": "true"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["revised"] == 1
    assert body["failed"] == []
    assert _render_count(client.service, "PDF", "revise") == before + 1

    row = await client.service.store.get_doc(doc_id)
    assert row.complete is True
    assert row.tile_count == 3


async def test_revision_skips_current_docs(client, sample_pdf):
    await client.post("/tile", **pdf_form(sample_pdf))

    r = await client.post("/revision")             # no force, render_version already current
    assert r.status_code == 200
    body = r.json()

    assert body["revised"] == 0
    assert body["skipped"] == 1


async def test_revision_heals_incomplete_doc(client, sample_pdf):
    svc = client.service
    doc_id = "f" * 64

    rel = await svc.blobs.put(doc_id, sample_pdf)
    await svc.store.begin_doc(
        doc_id=doc_id, doc_type="PDF", blob_rel_path=rel, source_url=None,
        page_height=0, tile_count=0, mime_type="application/pdf",
        metadata=None, render_version=svc.settings.render_version,
    )
    assert (await svc.store.get_doc(doc_id)).complete is False

    r = await client.post("/revision")             # no force, but incomplete → re-rendered
    assert r.status_code == 200
    assert r.json()["revised"] == 1

    row = await svc.store.get_doc(doc_id)
    assert row.complete is True
    assert row.tile_count == 3                      # sample_pdf has 3 pages


async def test_revision_keeps_pdf_strip_slicing(tmp_path):
    """A revised sliced PDF keeps its strip layout: page_heights survive, the tile count
    is unchanged and every strip still regenerates to its stored hash."""
    env = base_env(
        tmp_path,
        TILER_PDF_SLICE_ENABLE="true",
        TILER_PDF_TILE_HEIGHT="500",
        TILER_PDF_TILE_OVERLAP="100",
    )
    async with running_app(env) as (client, svc):
        manifest = (await client.post("/tile", **pdf_form(make_pdf(pages=3)))).json()
        doc_id = manifest["doc_id"]

        r = await client.post("/revision", params={"force": "true"})
        assert r.status_code == 200, r.text
        assert r.json()["revised"] == 1

        row = await svc.store.get_doc(doc_id)
        assert row.page_heights                                     # still sliced
        assert row.tile_count == manifest["tile_count"] > 3

        for tile_num in range(row.tile_count):
            got = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": tile_num})
            assert got.status_code == 200, got.text                # verified against the stored hash
