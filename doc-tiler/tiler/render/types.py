"""Result types shared by the render workers, the documents and the store.

`EncodedImage` is what the render workers return, so it must stay a plain module-level
dataclass (picklable). `RenderedTile` is what a document assembles from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hashing import sha256_hex
from .geometry import Tile


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """A cropped/encoded image with no placement: just its height and JPEG bytes."""

    height: int
    jpeg: bytes


@dataclass(frozen=True)
class RenderedTile:
    """One finished tile: its place in the document image, its size, its bytes and
    content hash."""

    tile_num: int
    offset: int
    width: int  # always TILE_WIDTH
    height: int
    jpeg: bytes
    image_hash: str


    @classmethod
    def build(cls, tile: Tile, image: EncodedImage, width: int) -> RenderedTile:
        """Hash the encoded image of `tile` and wrap the two as a finished tile."""
        return cls(
            tile_num=tile.num,
            offset=tile.offset,
            width=width,
            height=image.height,
            jpeg=image.jpeg,
            image_hash=sha256_hex(image.jpeg),
        )
