"""Document readers: every document type seen as pages of one fixed width.

A page is one long image `TILE_WIDTH` wide; a reader answers how tall each page is and
hands out rows `[y0, y1)` of a page. How it gets them is its own business — a PDF
rasterizes just that region, an image crops its decoded pixels, an MHTML archive takes
a Chromium capture. The service plans tiles over page heights and never looks inside.

`PdfReader` and `ImageReader` are `CpuReader`s: picklable descriptors (a path and the
render params) that do their work synchronously inside a render worker. `MhtmlReader`
works on the event loop, where Chromium lives, and hands raw captures to the render
pool for encoding.
"""

from __future__ import annotations

import contextlib
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pymupdf
from PIL import Image

from ..config import BAND_BYTES, MAX_SUPERSAMPLE, Settings
from ..store.cache import AsyncLRU
from . import browser as web
from .geometry import resized_height
from .render import DecodeError, RenderTooLarge, crop_rows, fit_width, open_image


@dataclass(frozen=True)
class RenderParams:
    """The render settings a reader needs; small and picklable, unlike Settings."""

    tile_width: int
    jpeg_quality: int
    pdf_dpi: int
    band_bytes: int = BAND_BYTES
    max_supersample: int = MAX_SUPERSAMPLE


    @classmethod
    def from_settings(cls, settings: Settings) -> RenderParams:
        """The render params of the running configuration."""
        return cls(
            tile_width=settings.tile_width,
            jpeg_quality=settings.jpeg_quality,
            pdf_dpi=settings.pdf_dpi,
        )


class DocumentReader(ABC):
    """A stored document at `path`, read at the geometry of `params`."""


    def __init__(self, path: str, params: RenderParams):
        self.path = path
        self.params = params


class CpuReader(DocumentReader):
    """A reader that runs inside a render worker (synchronously).

    Pickled into the worker with no open state; whatever it opens there it keeps until
    `trim()` (between tasks) or `close()`.
    """

    # Whether one task may take only part of the document's tiles. False when reading
    # any tile costs decoding the whole document anyway (an image).
    splits_batches: ClassVar[bool] = True


    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the descriptor, never what is open."""
        return {"path": self.path, "params": self.params}


    def __setstate__(self, state: dict[str, Any]) -> None:
        """Rebuild a closed reader from its descriptor."""
        self.__init__(state["path"], state["params"])


    @abstractmethod
    def page_heights(self) -> list[int]:
        """Height of every page at tile width, in page order."""


    @abstractmethod
    def read(self, page: int, y0: int, y1: int) -> Image.Image:
        """Rows `[y0, y1)` of `page`, `tile_width` wide (clipped to the page)."""


    def band_rows(self, page: int) -> int:
        """How many rows of `page` one `read` may take at most (memory bound)."""
        return sys.maxsize


    def trim(self) -> None:
        """Drop per-task state (decoded pixels, display lists) between tasks."""


    def close(self) -> None:
        """Release everything the reader holds."""
        self.trim()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _PageRaster:
    """How one PDF page is rasterized: its matrix, pixel box and display list."""

    matrix: Any
    box: Any         # the page's integer pixel rect at `matrix`
    height: int      # page height at tile width
    display_list: Any


class PdfReader(CpuReader):
    """Rasterizes a PDF region by region from a display list (the page is parsed once).

    A page renders at `pdf_dpi`, capped at `max_supersample` tile widths, and every
    read rasterizes only its rows (plus the resampling margin) before downscaling them
    to tile width — so memory follows the band, not the page.
    """


    def __init__(self, path: str, params: RenderParams):
        super().__init__(path, params)
        self._doc: Any = None
        self._raster: tuple[int, _PageRaster] | None = None


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


    def _page(self, page: int) -> _PageRaster:
        """The raster plan of `page`, with its display list (kept for the current page)."""
        if self._raster is not None and self._raster[0] == page:
            return self._raster[1]

        doc = self._open()

        if not 0 <= page < doc.page_count:
            raise DecodeError(f"page {page} is out of range")

        try:
            pdf_page = doc.load_page(page)
            raster = self._plan(pdf_page, with_display_list=True)

        except DecodeError:
            raise

        except Exception as exc:
            raise self._failure(f"cannot rasterize page {page}", exc) from exc

        self._raster = (page, raster)

        return raster


    def _plan(self, pdf_page: Any, *, with_display_list: bool) -> _PageRaster:
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

        return _PageRaster(
            matrix=matrix,
            box=box,
            height=resized_height(box.width, box.height, p.tile_width),
            display_list=pdf_page.get_displaylist() if with_display_list else None,
        )


    def page_heights(self) -> list[int]:
        """Height of every page at tile width (no rasterization)."""
        doc = self._open()

        try:
            return [self._plan(doc.load_page(n), with_display_list=False).height for n in range(doc.page_count)]

        except DecodeError:
            raise

        except Exception as exc:
            raise self._failure("cannot read the PDF pages", exc) from exc


    def band_rows(self, page: int) -> int:
        """Rows per band so one band's source pixels stay within band_bytes."""
        raster = self._page(page)
        scale = raster.box.height / raster.height
        source_rows = self.params.band_bytes // (raster.box.width * 3)
        margin = 2 * self._margin(scale)

        return max(1, math.floor((source_rows - margin) / scale))


    def read(self, page: int, y0: int, y1: int) -> Image.Image:
        """Rasterize just rows `[y0, y1)` of `page` (plus the filter margin), at tile width."""
        raster = self._page(page)
        y1 = min(y1, raster.height)

        if y1 <= y0:
            raise DecodeError(f"rows [{y0}, {y1}) fall outside page {page} (height={raster.height})")

        box = raster.box
        scale = box.height / raster.height
        top, bottom = y0 * scale, y1 * scale
        margin = self._margin(scale)
        first = max(0, math.floor(top) - margin)
        last = min(box.height, math.ceil(bottom) + margin)
        clip = pymupdf.Rect(box.x0, box.y0 + first, box.x1, box.y0 + last) * ~raster.matrix

        try:
            pix = raster.display_list.get_pixmap(matrix=raster.matrix, clip=clip, alpha=False)
            source = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            # Where the rasterized rows actually start: MuPDF rounds the clip to pixels.
            origin = pix.y - box.y0
            del pix

        except Exception as exc:
            raise self._failure(f"cannot rasterize page {page}", exc) from exc

        return source.resize(
            (self.params.tile_width, y1 - y0),
            Image.Resampling.LANCZOS,
            box=(0, top - origin, source.width, bottom - origin),
        )


    @staticmethod
    def _margin(scale: float) -> int:
        """Source rows LANCZOS reads beyond a band's edges, plus rounding slack."""
        return math.ceil(3 * max(scale, 1.0)) + 2


    @staticmethod
    def _failure(what: str, exc: Exception) -> Exception:
        """A MuPDF failure as DecodeError, or RenderTooLarge when it ran out of memory."""
        if isinstance(exc, MemoryError) or "malloc" in str(exc):
            return RenderTooLarge(f"{what}: out of memory ({exc})")

        return DecodeError(f"{what}: {exc}")


    def trim(self) -> None:
        """Drop the current page's display list; the document stays open."""
        self._raster = None


    def close(self) -> None:
        """Close the document."""
        self.trim()

        if self._doc is not None:
            with contextlib.suppress(Exception):
                self._doc.close()

            self._doc = None


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class ImageReader(CpuReader):
    """One page: the image decoded once (a JPEG at a reduced scale when it can be) and
    resized to tile width; a read crops its rows."""

    splits_batches = False


    def __init__(self, path: str, params: RenderParams):
        super().__init__(path, params)
        self._image: Image.Image | None = None


    def _page(self) -> Image.Image:
        """The decoded image at tile width."""
        if self._image is None:
            width = self.params.tile_width
            self._image = fit_width(open_image(self.path, min_width=width), width)

        return self._image


    def page_heights(self) -> list[int]:
        """The image is one page."""
        return [self._page().height]


    def read(self, page: int, y0: int, y1: int) -> Image.Image:
        """Rows `[y0, y1)` of the resized image."""
        if page != 0:
            raise DecodeError(f"page {page} is out of range (an image has one page)")

        return crop_rows(self._page(), y0, y1, "tile")


    def trim(self) -> None:
        """Drop the decoded pixels."""
        self._image = None


# --------------------------------------------------------------------------- #
# MHTML
# --------------------------------------------------------------------------- #
class MhtmlReader(DocumentReader):
    """An MHTML archive rendered by Chromium, on the event loop.

    One page, `TILE_WIDTH` CSS px wide. Up to `SAFE_FULLPAGE_PX` the page is captured
    whole (and the capture cached per document, so a doc's tiles cost one render);
    beyond it each tile is clip-captured on its own. The tab opens lazily — a cached
    capture needs none — and closes with `close()`.
    """


    def __init__(self, path: str, params: RenderParams, settings: Settings):
        super().__init__(path, params)
        self.settings = settings
        self._browser: web.BrowserPool | None = None
        self._captures: AsyncLRU | None = None
        self._stack = contextlib.AsyncExitStack()
        self._tab: Any = None


    def bind(self, browser: web.BrowserPool, captures: AsyncLRU) -> None:
        """Attach the browser pool and the per-document full-page capture cache."""
        self._browser = browser
        self._captures = captures


    def full_page_mode(self, page_height: int) -> bool:
        """Whether a page of this height is captured whole (else tile by tile)."""
        return page_height <= self.settings.safe_fullpage_px


    async def _page(self) -> Any:
        """The loaded archive in a Chromium tab, opened on first use."""
        if self._tab is None:
            if self._browser is None:
                raise RuntimeError("MhtmlReader is not bound to a browser pool")

            self._tab = await self._stack.enter_async_context(
                web.mhtml_page(self._browser, self.settings, self.path)
            )

        return self._tab


    async def css_height(self) -> int:
        """The page's full scroll height in CSS px."""
        return await web.page_height(await self._page())


    async def full_page(self) -> bytes:
        """The whole page as one PNG (cached per document)."""
        if self._captures is None:
            return await self._capture_full_page()

        return await self._captures.get_or_create(self.path, self._capture_full_page)


    async def _capture_full_page(self) -> bytes:
        """Take a full-page capture."""
        return await web.full_page_png(await self._page())


    async def clip(self, offset: int, height: int) -> bytes:
        """Rows `[offset, offset + height)` of the page as a PNG capture."""
        return await web.clip(await self._page(), self.settings, offset, height)


    async def close(self) -> None:
        """Close the tab, if one was opened."""
        await self._stack.aclose()
        self._tab = None
