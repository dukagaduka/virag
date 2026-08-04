"""CPU-bound rendering primitives (PDF raster, image resize/slice, JPEG encode).

Every function here must stay importable and picklable: they run inside
`ProcessPoolExecutor` workers, never on the event loop. No app state, no I/O
besides the bytes handed in.

Ingest and /get_tile deliberately share these functions — same decode, same
resize, same JPEG encoder settings — so a regenerated tile is byte-identical to
the one hashed at ingest.
"""

from __future__ import annotations

import io

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from ..config import JPEG_PROGRESSIVE, JPEG_SUBSAMPLING
from .geometry import resized_height, slice_strips, strip_for

# Guard against decompression-bomb DoS while still allowing very tall pages.
Image.MAX_IMAGE_PIXELS = 512_000_000


class DecodeError(ValueError):
    """The document could not be decoded (maps to 422)."""


def _flatten(img: Image.Image) -> Image.Image:
    """RGB with transparency composited over white — deterministic across formats."""

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):

        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])

        return canvas
    
    return img.convert("RGB") if img.mode != "RGB" else img


def _to_jpeg(img: Image.Image, quality: int) -> bytes:
    """Flatten and encode an image to JPEG bytes with fixed encoder settings."""
    buf = io.BytesIO()

    _flatten(img).save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=JPEG_SUBSAMPLING,
        progressive=JPEG_PROGRESSIVE,
        optimize=False,
    )

    return buf.getvalue()


def _open_image(data: bytes) -> Image.Image:
    """Decode image bytes and normalise EXIF rotation; raise DecodeError on failure."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()

    except Exception as exc:
        raise DecodeError(f"cannot decode image: {exc}") from exc
    
    # Normalise EXIF rotation at decode so ingest and get_tile agree.
    return ImageOps.exif_transpose(img) or img


def _fit_width(img: Image.Image, width: int) -> Image.Image:
    """Proportionally resize an image to exactly `width` (no-op if already that wide)."""
    if img.width == width:
        return img
    
    new_h = resized_height(img.width, img.height, width)

    return img.resize((width, new_h), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# PDF — one tile per page, no slicing.
# --------------------------------------------------------------------------- #
def pdf_page_count(blob: bytes) -> int:
    """Number of pages in a PDF blob; raise DecodeError if it cannot be opened."""
    try:
        with fitz.open(stream=blob, filetype="pdf") as doc:
            if doc.needs_pass:
                raise DecodeError("PDF is password protected")
            return doc.page_count
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"cannot open PDF: {exc}") from exc


def render_pdf_pages(
    blob: bytes,
    page_nums: list[int] | None,
    dpi: int,
    width: int,
    quality: int,
) -> list[tuple[int, int, bytes]]:
    """Return `[(page_num, height, jpeg)]` for the requested pages (all when None)."""
    try:
        doc = fitz.open(stream=blob, filetype="pdf")
    except Exception as exc:
        raise DecodeError(f"cannot open PDF: {exc}") from exc
    
    with doc:
        if doc.needs_pass:
            raise DecodeError("PDF is password protected")
        
        pages = list(range(doc.page_count)) if page_nums is None else page_nums
        out: list[tuple[int, int, bytes]] = []

        for pno in pages:
            if not 0 <= pno < doc.page_count:
                raise DecodeError(f"page {pno} is out of range")
            
            try:
                pix = doc.load_page(pno).get_pixmap(dpi=dpi, alpha=False)
            except Exception as exc:
                raise DecodeError(f"cannot rasterize page {pno}: {exc}") from exc
            
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img = _fit_width(img, width)

            out.append((pno, img.height, _to_jpeg(img, quality)))

        return out


# --------------------------------------------------------------------------- #
# Full-page sources: an IMG blob, or a full-page PNG screenshot.
# --------------------------------------------------------------------------- #
def page_height_of(data: bytes, width: int) -> int:
    """Height the source has after proportional resize to `width`."""
    return _fit_width(_open_image(data), width).height


def tile_full_page(
    data: bytes,
    width: int,
    tile_height: int,
    overlap: int,
    min_tile_height: int,
    quality: int,
) -> tuple[int, list[tuple[int, int, bytes]]]:
    """Slice a full-page source into strips → `(page_height, [(tile_num, height, jpeg)])`."""
    img = _fit_width(_open_image(data), width)
    page_height = img.height
    tiles = [
        (
            s.tile_num,
            s.height,
            _to_jpeg(img.crop((0, s.offset, width, s.offset + s.height)), quality),
        )
        for s in slice_strips(page_height, tile_height, overlap, min_tile_height)
    ]
    return page_height, tiles


def crop_one_tile(
    data: bytes,
    width: int,
    tile_num: int,
    page_height: int,
    tile_height: int,
    overlap: int,
    quality: int,
) -> tuple[int, bytes]:
    """Crop and encode a single tile by its index → `(height, jpeg)`."""
    img = _fit_width(_open_image(data), width)
    strip = strip_for(tile_num, page_height, tile_height, overlap)
    top = min(max(strip.offset, 0), img.height)
    bottom = min(strip.offset + strip.height, img.height)
    
    if bottom <= top:
        raise DecodeError(
            f"tile {tile_num} falls outside the rendered page "
            f"(page_height={page_height}, rendered={img.height})"
        )
    
    return bottom - top, _to_jpeg(img.crop((0, top, width, bottom)), quality)


def encode_strip(data: bytes, width: int, quality: int) -> tuple[int, bytes]:
    """Encode an already-cropped capture (tall-page clip path) → `(height, jpeg)`."""
    img = _flatten(_open_image(data))

    if img.width > width:
        img = img.crop((0, 0, width, img.height))

    elif img.width < width:  # narrower than the viewport: pad, never rescale
        padded = Image.new("RGB", (width, img.height), (255, 255, 255))
        padded.paste(img, (0, 0))
        img = padded

    return img.height, _to_jpeg(img, quality)
