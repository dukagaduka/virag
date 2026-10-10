"""Result types shared by the render workers, the service and the store.

The `Encoded*` ones are what the process-pool workers in `render` return, so they must
stay plain module-level dataclasses (picklable). `RenderedTile`/`RenderedDoc` are what
the service assembles from them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hashing import sha256_hex


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """A cropped/encoded image with no placement: just its height and JPEG bytes."""

    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class EncodedTile:
    """One page-per-tile PDF page, numbered by its page index."""

    tile_num: int
    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class EncodedStrip:
    """One vertical strip of a page: top offset in the resized page, height, JPEG bytes."""

    offset: int
    height: int
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class EncodedPage:
    """A page sliced into strips: its index, resized height and ordered strips."""

    page_num: int
    page_height: int
    strips: list[EncodedStrip]


@dataclass(frozen=True)
class RenderedTile:
    """One finished tile: its bytes, dimensions and content hash."""

    tile_num: int
    height: int
    jpeg: bytes
    image_hash: str

    @classmethod
    def build(cls, tile_num: int, height: int, jpeg: bytes) -> RenderedTile:
        """Hash `jpeg` and wrap it as a tile."""
        return cls(tile_num=tile_num, height=height, jpeg=jpeg, image_hash=sha256_hex(jpeg))


@dataclass(frozen=True)
class RenderedDoc:
    """A rendered document: full-page height plus its ordered tiles.

    `page_heights` is set only for a sliced PDF (per-page resized heights, in page
    order) — get_tile replays the same per-page slicing to reverse a global tile_num.
    """

    page_height: int
    tiles: list[RenderedTile]
    page_heights: list[int] | None = None
