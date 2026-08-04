"""IMG ingest: strip geometry, round-trip identity, page_height."""

from __future__ import annotations

import hashlib

from conftest import base_env, make_png, png_form, running_app

W, TH, MIN = 875, 1024, 28


async def test_img_strip_geometry_and_round_trip(client):
    src = make_png(width=1000, height=2600)  # → 875 wide, 2275 tall
    manifest = (await client.post("/tile", **png_form(src))).json()

    expected_h = round(2600 * W / 1000)
    assert manifest["doc_type"] == "IMG"
    assert manifest["page_height"] == expected_h
    assert manifest["tile_count"] == 3  # 1024 + 1024 + 227

    heights = [t["height"] for t in manifest["tiles"]]
    assert heights[:2] == [TH, TH]
    assert heights[2] == expected_h - 2 * TH

    for tile in manifest["tiles"]:
        got = await client.get(
            "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": tile["tile_num"]}
        )
        assert got.status_code == 200
        assert hashlib.sha256(got.content).hexdigest() == tile["image_hash"]


async def test_img_final_sliver_is_dropped(client):
    # 875 x (2048 + 10) → the 10px remainder is below MIN_TILE_HEIGHT.
    manifest = (await client.post("/tile", **png_form(make_png(width=875, height=2058)))).json()
    assert manifest["page_height"] == 2058
    assert manifest["tile_count"] == 2
    assert [t["height"] for t in manifest["tiles"]] == [TH, TH]


async def test_img_smaller_than_tile_width_is_upscaled(client):
    manifest = (await client.post("/tile", **png_form(make_png(width=400, height=300)))).json()
    assert manifest["tile_count"] == 1
    assert manifest["tiles"][0]["width"] == 875
    assert manifest["tiles"][0]["height"] == round(300 * 875 / 400)


async def test_overlap_changes_tile_count(tmp_path):
    src = make_png(width=875, height=4096)
    async with running_app(base_env(tmp_path, TILER_TILE_OVERLAP="100")) as (client, _):
        manifest = (await client.post("/tile", **png_form(src))).json()
        # STEP = 924: offsets 0, 924, 1848, 2772, 3696(h=400)
        assert manifest["tile_count"] == 5
        assert [t["height"] for t in manifest["tiles"]] == [1024, 1024, 1024, 1024, 400]
        for tile in manifest["tiles"]:
            got = await client.get(
                "/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": tile["tile_num"]}
            )
            assert hashlib.sha256(got.content).hexdigest() == tile["image_hash"]


async def test_cache_serves_repeated_tiles(client):
    manifest = (await client.post("/tile", **png_form(make_png(875, 2600)))).json()
    params = {"doc_id": manifest["doc_id"], "tile_num": 1}
    await client.get("/get_tile", params=params)
    hits_before = client.service.tile_cache.hits
    await client.get("/get_tile", params=params)
    assert client.service.tile_cache.hits == hits_before + 1


async def test_verify_false_skips_hash_check(client):
    manifest = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()
    got = await client.get(
        "/get_tile",
        params={"doc_id": manifest["doc_id"], "tile_num": 0, "verify": "false"},
    )
    assert got.status_code == 200


async def test_strict_policy_rejects_drifted_tile(client):
    """A poisoned image_hash simulates render drift → 409 under strict policy."""
    manifest = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()
    doc_id = manifest["doc_id"]
    store = client.service.store
    async with store._lock:
        await store.db.execute(
            "UPDATE tiles SET image_hash = ? WHERE doc_id = ? AND tile_num = 0",
            ("0" * 64, doc_id),
        )
        await store.db.commit()
    await client.service.tile_cache.clear()

    r = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "RENDER_MISMATCH"

    ok = await client.get("/get_tile", params={"doc_id": doc_id, "tile_num": 0, "verify": "false"})
    assert ok.status_code == 200


async def test_lenient_policy_serves_drifted_tile(tmp_path):
    async with running_app(base_env(tmp_path, TILER_RENDER_MISMATCH_POLICY="lenient")) as (client, svc):
        manifest = (await client.post("/tile", **png_form(make_png(875, 1200)))).json()
        async with svc.store._lock:
            await svc.store.db.execute(
                "UPDATE tiles SET image_hash = ? WHERE doc_id = ?", ("0" * 64, manifest["doc_id"])
            )
            await svc.store.db.commit()
        await svc.tile_cache.clear()
        r = await client.get("/get_tile", params={"doc_id": manifest["doc_id"], "tile_num": 0})
        assert r.status_code == 200
