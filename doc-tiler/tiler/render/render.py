"""Pixel primitives shared by every reader: decode, resize, flatten, JPEG encode.

Everything here runs inside the render workers, never on the event loop, and must stay
picklable: plain functions over the bytes or paths handed in, no app state.

Ingest and /get_tile deliberately share these functions — same decode, same resize,
same JPEG encoder settings — so a regenerated tile is byte-identical to the one hashed
at ingest.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps

from ..config import JPEG_PROGRESSIVE, JPEG_SUBSAMPLING
from .geometry import resized_height
from .types import EncodedImage

# Memory is bounded per render worker (TILER_RENDER_WORKER_MEMORY), so Pillow's own
# pixel-count guard would only reject documents the worker can afford.
Image.MAX_IMAGE_PIXELS = None


class DecodeError(ValueError):
    """The document could not be decoded (maps to 422 DECODE_FAILED)."""


class RenderTooLarge(MemoryError):
    """A render needed more memory than one worker may take (maps to 422 RENDER_TOO_LARGE)."""


def flatten(img: Image.Image) -> Image.Image:
    """RGB with transparency composited over white — deterministic across formats."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):

        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])

        return canvas

    return img.convert("RGB") if img.mode != "RGB" else img


def to_jpeg(img: Image.Image, quality: int) -> bytes:
    """Flatten and encode an image to JPEG bytes with fixed encoder settings."""
    buf = io.BytesIO()

    flatten(img).save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=JPEG_SUBSAMPLING,
        progressive=JPEG_PROGRESSIVE,
        optimize=False,
    )

    return buf.getvalue()


def open_image(source: str | bytes, min_width: int | None = None) -> Image.Image:
    """Decode an image file or bytes and normalise its EXIF rotation.

    With `min_width`, a JPEG is decoded straight at the smallest 1/2^k scale that still
    keeps both sides at least that wide, so a large photo never decodes at full size.
    Raises DecodeError when the data is not a readable image.
    """
    try:
        img = Image.open(source if isinstance(source, str) else io.BytesIO(source))

        if min_width is not None and img.format == "JPEG":
            img.draft(None, (min_width, min_width))

        img.load()

    except MemoryError:
        raise

    except Exception as exc:
        raise DecodeError(f"cannot decode image: {exc}") from exc

    # Normalise EXIF rotation at decode so ingest and get_tile agree.
    return ImageOps.exif_transpose(img) or img


def fit_width(img: Image.Image, width: int) -> Image.Image:
    """Proportionally resize an image to exactly `width` (no-op if already that wide)."""
    if img.width == width:
        return img

    return img.resize((width, resized_height(img.width, img.height, width)), Image.Resampling.LANCZOS)


def crop_rows(img: Image.Image, y0: int, y1: int, what: str) -> Image.Image:
    """Rows `[y0, y1)` of an image, clipped to it; DecodeError when nothing is left."""
    top = min(max(y0, 0), img.height)
    bottom = min(y1, img.height)

    if bottom <= top:
        raise DecodeError(f"{what} [{y0}, {y1}) falls outside the rendered page (height={img.height})")

    return img.crop((0, top, img.width, bottom))


# --------------------------------------------------------------------------- #
# Chromium captures (MHTML): encoded in the render workers, captured on the loop.
# --------------------------------------------------------------------------- #
def png_page_height(png: bytes, width: int) -> int:
    """Height a full-page capture has after proportional resize to `width`."""
    return fit_width(open_image(png), width).height


def crop_page_png(png: bytes, rows: list[tuple[int, int]], width: int, quality: int) -> list[EncodedImage]:
    """Encode several row ranges of one full-page capture (decoded once)."""
    page = fit_width(open_image(png), width)
    out = []

    for y0, y1 in rows:
        tile = crop_rows(page, y0, y1, "tile")
        out.append(EncodedImage(height=tile.height, jpeg=to_jpeg(tile, quality)))

    return out


def encode_strip(png: bytes, width: int, quality: int) -> EncodedImage:
    """Encode an already-cropped capture (tall-page clip path)."""
    img = flatten(open_image(png))

    if img.width > width:
        img = img.crop((0, 0, width, img.height))

    elif img.width < width:  # narrower than the viewport: pad, never rescale
        padded = Image.new("RGB", (width, img.height), (255, 255, 255))
        padded.paste(img, (0, 0))
        img = padded

    return EncodedImage(height=img.height, jpeg=to_jpeg(img, quality))
