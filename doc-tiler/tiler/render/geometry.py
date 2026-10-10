"""Tiling geometry — the single source of truth for forward and reverse mapping.

Forward (ingest) and reverse (get_tile) MUST agree byte-for-byte or `image_hash`
comparison in /get_tile breaks. Both live here and are property-tested together.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strip:
    """One tile's placement: its index, top offset, and height in the full-page image."""

    tile_num: int
    offset: int
    height: int


def slice_strips(
    page_height: int, tile_height: int, overlap: int, min_tile_height: int
) -> list[Strip]:
    """Slice a full-page image of height `page_height` into ordered strips.

    A final strip shorter than `min_tile_height` is dropped — and only ever the
    final one, which is what keeps the reverse mapping consistent.
    """
    step = tile_height - overlap

    if step <= 0:
        raise ValueError("TILE_OVERLAP must be < TILE_HEIGHT")
    
    strips: list[Strip] = []

    if page_height <= 0:
        return strips
    
    offset = 0

    while True:
        h = min(tile_height, page_height - offset)

        if h >= min_tile_height:
            strips.append(Strip(tile_num=len(strips), offset=offset, height=h))
        if offset + tile_height >= page_height:
            break
        
        offset += step

    return strips


def slice_doc_strips(
    page_heights: list[int], tile_height: int, overlap: int, min_tile_height: int
) -> list[tuple[int, Strip]]:
    """Global strip layout for a multi-page document, sliced per page (pages are
    never stitched: slicing stays within one page's bounds).

    Returns ``[(page_no, Strip(tile_num=<global>, offset, height))]`` with contiguous
    global tile numbers assigned in page order. Ingest and get_tile both call this
    with the same ``page_heights`` and params, so the forward and reverse mappings are
    the same computation by construction — no separate inverse to keep in sync.
    """
    out: list[tuple[int, Strip]] = []
    running = 0

    for page_no, page_height in enumerate(page_heights):
        for s in slice_strips(page_height, tile_height, overlap, min_tile_height):
            out.append((page_no, Strip(tile_num=running, offset=s.offset, height=s.height)))
            running += 1

    return out


def strip_for(
    tile_num: int, page_height: int, tile_height: int, overlap: int
) -> Strip:
    """Compute a single tile's placement from its index (inverse of slice_strips).

    Valid for `tile_num < len(slice_strips(...))`. It agrees with the forward pass
    because a short strip is only ever dropped as the final element, so every
    surviving tile keeps the same offset `tile_num * step` it had going forward.
    """
    step = tile_height - overlap
    offset = tile_num * step
    height = min(tile_height, page_height - offset)

    return Strip(tile_num=tile_num, offset=offset, height=height)


def resized_height(src_width: int, src_height: int, target_width: int) -> int:
    """Height after proportionally resizing to `target_width`. Never returns 0."""
    if src_width <= 0 or src_height <= 0:
        raise ValueError("source dimensions must be positive")
    
    return max(1, round(src_height * target_width / src_width))
