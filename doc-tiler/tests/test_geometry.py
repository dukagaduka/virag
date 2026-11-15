"""Tiling geometry, including the forward/reverse agreement property."""

from __future__ import annotations

import random

import pytest

from tiler.render.geometry import (
    TileSpec,
    bands_for,
    plan_tiles,
    resized_height,
    slice_doc_strips,
    slice_strips,
)

TH = 1024
MIN = 28


def test_exact_multiple_of_tile_height():
    strips = slice_strips(2048, TH, 0, MIN)
    assert [(s.tile_num, s.offset, s.height) for s in strips] == [(0, 0, 1024), (1, 1024, 1024)]


def test_final_sliver_below_min_is_dropped():
    strips = slice_strips(1030, TH, 0, MIN)  # 6px remainder
    assert len(strips) == 1
    assert strips[0].height == 1024


def test_final_strip_at_min_is_kept():
    strips = slice_strips(1024 + MIN, TH, 0, MIN)
    assert [s.height for s in strips] == [1024, MIN]


def test_overlap_increases_tile_count():
    plain = slice_strips(4096, TH, 0, MIN)
    overlapped = slice_strips(4096, TH, 100, MIN)
    assert len(overlapped) > len(plain)
    assert [s.offset for s in overlapped[:3]] == [0, 924, 1848]


def test_page_shorter_than_min_yields_no_tiles():
    assert slice_strips(20, TH, 0, MIN) == []


@pytest.mark.parametrize("overlap", [0, 1, 100, 512, 1023])
def test_strips_on_random_heights(overlap):
    rng = random.Random(1337 + overlap)
    # Keep the sample space bounded: with overlap=1023 the step is 1px, so a
    # 200k-tall page would mean 200k strips per sample.
    ceiling = 200_000 if overlap <= 100 else 20_000
    heights = [rng.randint(1, ceiling) for _ in range(120)]
    heights += [TH - 1, TH, TH + 1, MIN, MIN - 1, 2 * TH, 2 * TH - overlap]
    for height in heights:
        strips = slice_strips(height, TH, overlap, MIN)
        # Offsets are exactly n*STEP and every strip stays inside the page.
        step = TH - overlap
        for strip in strips:
            assert strip.offset == strip.tile_num * step
            assert strip.offset + strip.height <= height
            assert MIN <= strip.height <= TH
        if strips:
            # No coverage gap between consecutive strips.
            for a, b in zip(strips, strips[1:]):
                assert b.offset <= a.offset + a.height


@pytest.mark.parametrize("overlap", [0, 100, 300])
def test_slice_doc_strips_numbers_globally_and_slices_per_page(overlap):
    rng = random.Random(99 + overlap)
    pdf_h = 600
    page_heights = [rng.randint(1, 5000) for _ in range(40)]
    layout = slice_doc_strips(page_heights, pdf_h, overlap, MIN)

    # Contiguous global tile numbers, in page order.
    assert [strip.tile_num for _, strip in layout] == list(range(len(layout)))
    assert [page_no for page_no, _ in layout] == sorted(page_no for page_no, _ in layout)

    # Each page's strips equal slicing that page in isolation (no stitching across pages).
    by_page: dict[int, list] = {}
    for page_no, strip in layout:
        by_page.setdefault(page_no, []).append((strip.offset, strip.height))
    for page_no, strips in by_page.items():
        expected = slice_strips(page_heights[page_no], pdf_h, overlap, MIN)
        assert strips == [(e.offset, e.height) for e in expected]
        for off, h in strips:  # every strip stays inside its own page
            assert off + h <= page_heights[page_no]


def test_slice_doc_strips_edges():
    assert slice_doc_strips([], 600, 0, MIN) == []
    assert slice_doc_strips([20, 20], 600, 0, MIN) == []  # both pages below MIN → nothing


def test_resized_height_is_proportional():
    assert resized_height(1000, 2000, 875) == 1750
    assert resized_height(3, 1, 875) == 292  # rounds, never zero
    assert resized_height(10_000, 1, 875) == 1


def test_plan_tiles_whole_pages_is_one_tile_per_page():
    assert plan_tiles([1200, 900], TH, 0, MIN, whole_pages=True) == [
        TileSpec(tile_num=0, page=0, offset=0, height=1200),
        TileSpec(tile_num=1, page=1, offset=0, height=900),
    ]


def test_plan_tiles_slices_each_page_and_numbers_globally():
    specs = plan_tiles([2058, 1500], TH, 0, MIN)
    assert [(t.tile_num, t.page, t.offset, t.height) for t in specs] == [
        (0, 0, 0, 1024),
        (1, 0, 1024, 1024),          # the 10 px rest of page 0 is a dropped sliver
        (2, 1, 0, 1024),
        (3, 1, 1024, 476),
    ]


def test_bands_take_the_rest_of_the_tile_last():
    assert bands_for(1024, 300) == [(0, 300), (300, 600), (600, 900), (900, 1024)]
    assert bands_for(1024, 1024) == [(0, 1024)]
    assert bands_for(1024, 5000) == [(0, 1024)]
    assert bands_for(7, 1) == [(i, i + 1) for i in range(7)]


@pytest.mark.parametrize("band", [1, 7, 100, 1023, 1024, 1025])
def test_bands_cover_the_tile_exactly(band):
    for height in (1, 28, 999, 1024, 4097):
        bands = bands_for(height, band)
        assert bands[0][0] == 0 and bands[-1][1] == height
        assert all(a[1] == b[0] for a, b in zip(bands, bands[1:]))
        assert all(0 < end - start <= band for start, end in bands)


def test_bands_reject_an_empty_band():
    with pytest.raises(ValueError):
        bands_for(10, 0)
