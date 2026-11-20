"""Tiling geometry: how a document image is cut into tiles.

A document is one image `TILE_WIDTH` wide; a tile is a run of its rows. The cut is made
once, at ingest, and stored with the tile (`tiles.offset`, `tiles.height`), so
a tile request reads the tile's rows back instead of cutting the document again.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """Rows `[offset, offset + height)` of the document image."""

    num: int
    offset: int
    height: int


@dataclass(frozen=True)
class Tiling:
    """How a run of rows is cut into tiles."""

    height: int          # tile height
    overlap: int
    min_height: int      # a last tile shorter than this is dropped
    whole: bool = False  # the run is one tile whatever its height (a PDF page, unsliced)


    def cut(self, rows: int, top: int = 0, first: int = 0) -> list[Tile]:
        """Tiles of rows `[top, top + rows)`, numbered from `first`.

        A final tile shorter than `min_height` is dropped — and only ever the final
        one, so every tile sits `height - overlap` rows below the previous.
        """
        if rows <= 0:
            return []

        if self.whole:
            return [Tile(first, top, rows)]

        step = self.height - self.overlap

        if step <= 0:
            raise ValueError("the tile overlap must be smaller than the tile height")

        tiles: list[Tile] = []
        offset = 0

        while True:
            h = min(self.height, rows - offset)

            if h >= self.min_height:
                tiles.append(Tile(first + len(tiles), top + offset, h))

            if offset + self.height >= rows:
                break

            offset += step

        return tiles


def resized_height(src_width: int, src_height: int, target_width: int) -> int:
    """Height after proportionally resizing to `target_width`. Never returns 0."""
    if src_width <= 0 or src_height <= 0:
        raise ValueError("source dimensions must be positive")

    return max(1, round(src_height * target_width / src_width))
