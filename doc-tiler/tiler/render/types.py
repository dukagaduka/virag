"""Result types shared by the render workers, the service and the store.

`EncodedImage` is what the render workers return, so it must stay a plain module-level
dataclass (picklable). `RenderedTile` is what the service assembles from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hashing import sha256_hex


@dataclass(frozen=True, slots=True)
class EncodedImage:
    """A cropped/encoded image with no placement: just its height and JPEG bytes."""

    height: int
    jpeg: bytes


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
