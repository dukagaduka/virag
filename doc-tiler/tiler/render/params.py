"""RenderParams: what a document renders by.

The geometry and pixel settings in the form the render layer uses — small and picklable,
unlike the settings themselves: every render worker gets them once, at start.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MAX_SUPERSAMPLE, GeometrySettings, PixelSettings
from .geometry import Tiling


@dataclass(frozen=True)
class RenderParams:
    """What a document renders by — its geometry and pixel settings; small and picklable,
    unlike the settings themselves."""

    tile_width: int
    jpeg_quality: int
    pdf_dpi: int
    tiling: Tiling
    pdf_tiling: Tiling
    max_supersample: int = MAX_SUPERSAMPLE


    @classmethod
    def from_settings(cls, geometry: GeometrySettings, pixel: PixelSettings) -> RenderParams:
        """The render params of the running configuration."""
        g = geometry

        return cls(
            tile_width=g.tile_width,
            jpeg_quality=pixel.jpeg_quality,
            pdf_dpi=pixel.pdf_dpi,
            tiling=Tiling(g.tile_height, g.tile_overlap, g.min_tile_height),
            pdf_tiling=Tiling(g.pdf_tile_height, g.pdf_tile_overlap, g.min_tile_height, whole=not g.pdf_slice),
        )
