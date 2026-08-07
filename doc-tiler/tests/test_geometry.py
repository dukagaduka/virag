"""Tiling geometry, including the forward/reverse agreement property."""

from __future__ import annotations

import random

import pytest

from tiler.render.geometry import (
    resized_height,
    slice_doc_strips,
    slice_strips,
    strip_for,
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
def test_forward_and_reverse_agree_on_random_heights(overlap):
    rng = random.Random(1337 + overlap)
    # Keep the sample space bounded: with overlap=1023 the step is 1px, so a
    # 200k-tall page would mean 200k strips per sample.
    ceiling = 200_000 if overlap <= 100 else 20_000
    heights = [rng.randint(1, ceiling) for _ in range(120)]
    heights += [TH - 1, TH, TH + 1, MIN, MIN - 1, 2 * TH, 2 * TH - overlap]
    for height in heights:
        strips = slice_strips(height, TH, overlap, MIN)
        for strip in strips:
            reverse = strip_for(strip.tile_num, height, TH, overlap)
            assert reverse == strip, f"H={height} overlap={overlap} tile={strip.tile_num}"
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
