"""Tiling geometry: how a run of rows is cut into tiles."""

from __future__ import annotations

import random

import pytest

from tiler.render.geometry import Tile, Tiling, resized_height

TH = 1024
MIN = 28


def _cut(rows: int, overlap: int = 0) -> list[Tile]:
    return Tiling(TH, overlap, MIN).cut(rows)


def test_exact_multiple_of_tile_height():
    assert _cut(2048) == [Tile(0, 0, 1024), Tile(1, 1024, 1024)]


def test_final_sliver_below_min_is_dropped():
    tiles = _cut(1030)  # 6px remainder
    assert len(tiles) == 1
    assert tiles[0].height == 1024


def test_final_tile_at_min_is_kept():
    assert [t.height for t in _cut(1024 + MIN)] == [1024, MIN]


def test_overlap_increases_tile_count():
    plain = _cut(4096)
    overlapped = _cut(4096, overlap=100)
    assert len(overlapped) > len(plain)
    assert [t.offset for t in overlapped[:3]] == [0, 924, 1848]


def test_rows_shorter_than_min_yield_no_tiles():
    assert _cut(20) == []
    assert _cut(0) == []


@pytest.mark.parametrize("overlap", [0, 1, 100, 512, 1023])
def test_tiles_on_random_heights(overlap):
    rng = random.Random(1337 + overlap)
    # Keep the sample space bounded: with overlap=1023 the step is 1px, so a
    # 200k-tall image would mean 200k tiles per sample.
    ceiling = 200_000 if overlap <= 100 else 20_000
    heights = [rng.randint(1, ceiling) for _ in range(120)]
    heights += [TH - 1, TH, TH + 1, MIN, MIN - 1, 2 * TH, 2 * TH - overlap]
    for height in heights:
        tiles = _cut(height, overlap)
        # Offsets are exactly n*STEP and every tile stays inside the image.
        step = TH - overlap
        for tile in tiles:
            assert tile.offset == tile.num * step
            assert tile.offset + tile.height <= height
            assert MIN <= tile.height <= TH
        # No coverage gap between consecutive tiles.
        for a, b in zip(tiles, tiles[1:]):
            assert b.offset <= a.offset + a.height


def test_a_cut_is_placed_at_top_and_numbered_from_first():
    tiling = Tiling(TH, 100, MIN)
    plain = tiling.cut(3000)
    moved = tiling.cut(3000, top=5000, first=7)
    assert moved == [Tile(t.num + 7, t.offset + 5000, t.height) for t in plain]


@pytest.mark.parametrize("overlap", [0, 100, 300])
def test_runs_cut_one_after_another_number_globally_and_never_cross(overlap):
    """What a PDF does with its pages: each cut where it sits, numbered on from the last."""
    rng = random.Random(99 + overlap)
    tiling = Tiling(600, overlap, MIN)
    pages = [rng.randint(1, 5000) for _ in range(40)]

    tiles: list[Tile] = []
    top = 0
    for rows in pages:
        own = tiling.cut(rows, top, first=len(tiles))
        # Cut where it sits == cut in isolation, moved: nothing is stitched across runs.
        assert [(t.offset - top, t.height) for t in own] == [(t.offset, t.height) for t in tiling.cut(rows)]
        assert all(top <= t.offset and t.offset + t.height <= top + rows for t in own)
        tiles += own
        top += rows

    assert [t.num for t in tiles] == list(range(len(tiles)))


def test_a_whole_run_is_one_tile_whatever_its_height():
    whole = Tiling(TH, 0, MIN, whole=True)
    assert whole.cut(1200, top=300, first=2) == [Tile(2, 300, 1200)]
    assert whole.cut(10) == [Tile(0, 0, 10)]                         # never dropped as a sliver
    assert whole.cut(0) == []


def test_an_overlap_that_leaves_no_step_is_refused():
    with pytest.raises(ValueError):
        Tiling(100, 100, MIN).cut(500)


def test_resized_height_is_proportional():
    assert resized_height(1000, 2000, 875) == 1750
    assert resized_height(3, 1, 875) == 292  # rounds, never zero
    assert resized_height(10_000, 1, 875) == 1
