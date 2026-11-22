"""Document readers: a stored document read as one image `TILE_WIDTH` wide.

A reader knows one format. It answers how tall the document image is and how it is cut
into tiles (`layout`), and hands out rows `[y0, y1)` of it (`read`). How it gets them is
its own business — a PDF stacks its pages and rasterizes just the rows asked for, an
image crops its decoded pixels.

Readers live inside the render workers: a worker creates one from the document's type
and path, keeps it open between the tasks of one document, and nothing of it ever
crosses a process boundary. MHTML has no reader — Chromium runs on the event loop
(`documents.MhtmlDocument`).
"""

from __future__ import annotations

import bisect
import contextlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, ClassVar

import pymupdf
from PIL import Image

from ..config import MAX_SUPERSAMPLE, GeometrySettings, PixelSettings
from ..models import DocType
from .geometry import Tile, Tiling, resized_height
from .pixels import DecodeError, EmptyDocument, RenderTooLarge, crop_rows, fit_width, open_image


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


class DocumentReader(ABC):
    """A stored document at `path`, read as one image `params.tile_width` wide.

    Whatever it opens it keeps until `trim()` (between tasks) or `close()`.
    """

    # Whether one task may take only part of the document's tiles. False when reading
    # any tile costs decoding the whole document anyway (an image).
    splits_batches: ClassVar[bool] = True


    def __init__(self, path: str, params: RenderParams):
        self.path = path
        self.params = params


    @abstractmethod
    def layout(self) -> tuple[int, list[Tile]]:
        """The height of the document image and the tiles it is cut into."""


    @abstractmethod
    def read(self, y0: int, y1: int) -> Image.Image:
        """Rows `[y0, y1)` of the document image."""


    def trim(self) -> None:
        """Drop per-task state (decoded pixels, display lists) between tasks."""


    def close(self) -> None:
        """Release everything the reader holds."""
        self.trim()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
class PdfReader(DocumentReader):
    """Pages stacked top to bottom at tile width; where each page sits is known only here.

    A page renders at `pdf_dpi`, capped at `max_supersample` tile widths, from its
    display list (the page is parsed once), and every read rasterizes only its rows
    (plus the resampling margin) before downscaling them to tile width.
    """

    @dataclass(frozen=True)
    class _Page:
        """Where one page sits in the document image and how it is rasterized."""

        number: int
        top: int      # first row in the document image
        height: int   # rows at tile width
        matrix: Any   # page space -> raster pixels
        box: Any      # the page's integer pixel rect at `matrix`


    def __init__(self, path: str, params: RenderParams):
        super().__init__(path, params)
        self._doc: Any = None
        self._pages: list[PdfReader._Page] | None = None
        self._drawn: tuple[int, Any] | None = None  # (page number, its display list)


    def layout(self) -> tuple[int, list[Tile]]:
        """Each page cut on its own (or kept whole), so no tile crosses a page boundary."""
        pages = self._layout_pages()

        if not pages:
            raise EmptyDocument("PDF has no pages")

        tiles: list[Tile] = []

        for page in pages:
            tiles += self.params.pdf_tiling.cut(page.height, page.top, first=len(tiles))

        return pages[-1].top + pages[-1].height, tiles


    def read(self, y0: int, y1: int) -> Image.Image:
        """Rows `[y0, y1)` of the page they fall on."""
        page = self._page_at(y0)

        if not y0 < y1 <= page.top + page.height:
            raise DecodeError(f"rows [{y0}, {y1}) do not fit page {page.number}")

        return self._rasterize(page, y0 - page.top, y1 - page.top)


    def _open(self) -> Any:
        """The open document; DecodeError when it is not a readable PDF."""
        if self._doc is None:
            try:
                doc = pymupdf.open(self.path, filetype="pdf")

            except Exception as exc:
                raise self._failure("cannot open PDF", exc) from exc

            if doc.needs_pass:
                doc.close()
                raise DecodeError("PDF is password protected")

            self._doc = doc

        return self._doc


    def _layout_pages(self) -> list[_Page]:
        """Every page's place in the document image (no rasterization), kept while the
        document is open."""
        if self._pages is None:
            doc = self._open()
            pages: list[PdfReader._Page] = []
            top = 0

            try:
                for number in range(doc.page_count):
                    page = self._plan_page(doc.load_page(number), top)
                    pages.append(page)
                    top += page.height

            except DecodeError:
                raise

            except Exception as exc:
                raise self._failure("cannot read the PDF pages", exc) from exc

            self._pages = pages

        return self._pages


    def _plan_page(self, pdf_page: Any, top: int) -> _Page:
        """Matrix and pixel box of a page at pdf_dpi, capped at max_supersample widths."""
        p = self.params
        rect = pdf_page.rect

        if rect.width <= 0 or rect.height <= 0:
            raise DecodeError(f"page {pdf_page.number} has an empty page box")

        zoom = min(p.pdf_dpi / 72, p.max_supersample * p.tile_width / rect.width)
        matrix = pymupdf.Matrix(zoom, zoom)
        box = (rect * matrix).irect

        if box.width <= 0 or box.height <= 0:
            raise DecodeError(f"page {pdf_page.number} rasterizes to nothing")

        return self._Page(
            number=pdf_page.number,
            top=top,
            height=resized_height(box.width, box.height, p.tile_width),
            matrix=matrix,
            box=box,
        )


    def _page_at(self, y: int) -> _Page:
        """The page row `y` of the document image falls on."""
        pages = self._layout_pages()
        i = bisect.bisect_right(pages, y, key=attrgetter("top")) - 1

        if i < 0 or y >= pages[i].top + pages[i].height:
            raise DecodeError(f"row {y} is outside the document")

        return pages[i]


    def _display_list(self, page: _Page) -> Any:
        """The parsed page (kept for the current page: a batch's tiles come in order)."""
        if self._drawn is None or self._drawn[0] != page.number:
            try:
                display_list = self._open().load_page(page.number).get_displaylist()

            except Exception as exc:
                raise self._failure(f"cannot rasterize page {page.number}", exc) from exc

            self._drawn = (page.number, display_list)

        return self._drawn[1]


    def _rasterize(self, page: _Page, y0: int, y1: int) -> Image.Image:
        """Rasterize just rows `[y0, y1)` of `page` (plus the filter margin), at tile width."""
        display_list = self._display_list(page)
        box = page.box
        scale = box.height / page.height
        top, bottom = y0 * scale, y1 * scale
        margin = self._margin(scale)
        first = max(0, math.floor(top) - margin)
        last = min(box.height, math.ceil(bottom) + margin)
        clip = pymupdf.Rect(box.x0, box.y0 + first, box.x1, box.y0 + last) * ~page.matrix

        try:
            pix = display_list.get_pixmap(matrix=page.matrix, clip=clip, alpha=False)
            source = Image.frombytes("RGB", (pix.width, pix.height), pix.samples_mv)
            # Where the rasterized rows actually start: MuPDF rounds the clip to pixels.
            origin = pix.y - box.y0
            del pix

        except Exception as exc:
            raise self._failure(f"cannot rasterize page {page.number}", exc) from exc

        return source.resize(
            (self.params.tile_width, y1 - y0),
            Image.Resampling.LANCZOS,
            box=(0, top - origin, source.width, bottom - origin),
        )


    @staticmethod
    def _margin(scale: float) -> int:
        """Source rows LANCZOS reads beyond the edges of the rows asked for, plus
        rounding slack."""
        return math.ceil(3 * max(scale, 1.0)) + 2


    @staticmethod
    def _failure(what: str, exc: Exception) -> Exception:
        """A MuPDF failure as DecodeError, or RenderTooLarge when it ran out of memory."""
        if isinstance(exc, MemoryError) or "malloc" in str(exc):
            return RenderTooLarge(f"{what}: out of memory ({exc})")

        return DecodeError(f"{what}: {exc}")


    def trim(self) -> None:
        """Drop the current page's display list; the document and its layout stay."""
        self._drawn = None


    def close(self) -> None:
        """Close the document."""
        self.trim()

        if self._doc is not None:
            with contextlib.suppress(Exception):
                self._doc.close()

            self._doc = None

        self._pages = None


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class ImageReader(DocumentReader):
    """The image decoded once per task (a JPEG at a reduced scale when it can be) and
    resized to tile width; a read crops its rows."""

    splits_batches = False


    def __init__(self, path: str, params: RenderParams):
        super().__init__(path, params)
        self._image: Image.Image | None = None


    def layout(self) -> tuple[int, list[Tile]]:
        """The resized image cut top to bottom."""
        height = self._fitted().height

        return height, self.params.tiling.cut(height)


    def read(self, y0: int, y1: int) -> Image.Image:
        """Rows `[y0, y1)` of the resized image."""
        return crop_rows(self._fitted(), y0, y1, "tile")


    def _fitted(self) -> Image.Image:
        """The decoded image at tile width."""
        if self._image is None:
            width = self.params.tile_width
            self._image = fit_width(open_image(self.path, min_width=width), width)

        return self._image


    def trim(self) -> None:
        """Drop the decoded pixels."""
        self._image = None


# The reader of each document type the render workers read on their own.
READERS: dict[DocType, type[DocumentReader]] = {
    DocType.PDF: PdfReader,
    DocType.IMG: ImageReader,
}
