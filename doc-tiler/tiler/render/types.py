"""Result types shared by the render workers, the service and the store.

`EncodedImage` is what the render workers return, so it must stay a plain module-level
dataclass (picklable). `RenderedTile` is what the service assembles from it.
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
    """One finished tile: its place in the document image, its bytes and content hash."""

    tile_num: int
    offset: int
    height: int
    jpeg: bytes
    image_hash: str


    @classmethod
    def build(cls, tile: Tile, image: EncodedImage) -> RenderedTile:
        """Hash the encoded image of `tile` and wrap the two as a finished tile."""
        return cls(
            tile_num=tile.num,
            offset=tile.offset,
            height=image.height,
            jpeg=image.jpeg,
            image_hash=sha256_hex(image.jpeg),
        )
