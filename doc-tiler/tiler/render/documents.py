"""Documents: a document opened for rendering (through `Renderer.open`).

Whatever its format, a document is one image `TILE_WIDTH` wide and `height` tall:
`tiles()` yields its tiles rendered — `plan()` measures it and cuts it into tiles
first, unless the cut is given — and `render()` returns the JPEGs of one batch of them.
Where that happens is the document's own business:

- `WorkerDocument` (PDF, images) — entirely inside the render workers, by the format's
  reader (`readers.py`);
- `MhtmlDocument` — Chromium captures on the event loop, the workers encode.

Both run their work through the RenderPool, which bounds it.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from abc import ABC, abstractmethod
from collections import deque
from itertools import islice
from typing import AsyncIterator, Iterator

from ..models import DocType
from . import render, worker
from .browser import Tab
from .geometry import Tile
from .pool import RenderPool
from .readers import READERS
from .types import EncodedImage, RenderedTile


class Document(ABC):
    """A stored document opened for rendering: one image `tile_width` wide.

    A session (`async with`): what it holds — a Chromium tab, a bulk slot — is released
    on exit. `bulk` marks background work (ingest), held to the pool's bulk slots;
    `height`, already known for a stored document, spares measuring it.
    """


    def __init__(
        self,
        pool: RenderPool,
        doc_type: DocType,
        path: str,
        *,
        bulk: bool = False,
        height: int | None = None,
    ):

        self.pool = pool
        self.params = pool.params
        self.doc_type = doc_type
        self.path = path
        self.bulk = bulk
        self.rendered = 0  # tiles yielded so far
        self._height = height


    async def __aenter__(self) -> Document:
        return self


    async def __aexit__(self, *exc: object) -> None:
        await self.close()


    @property
    def height(self) -> int:
        """Height of the document image: measured by `plan()`, or given for a stored doc."""
        if self._height is None:
            raise RuntimeError("the document is not planned yet")

        return self._height


    @abstractmethod
    async def plan(self) -> list[Tile]:
        """Measure the document and cut it into tiles."""


    @abstractmethod
    async def render(self, tiles: list[Tile]) -> list[EncodedImage]:
        """Render and encode the given tiles, in order."""


    async def tiles(self, cut: list[Tile] | None = None) -> AsyncIterator[RenderedTile]:
        """The document's tiles, rendered, in tile order: those of `cut` (made before,
        for a stored doc), or all of them — the document is then measured and cut first.
        They render batch by batch, up to `_parallel` batches at once."""
        if cut is None:
            cut = await self.plan()

        batches = iter(self._batches(cut))
        running: deque[tuple[list[Tile], asyncio.Task]] = deque()

        try:
            self._top_up(running, batches)

            while running:
                batch, task = running.popleft()
                encoded = await task
                self._top_up(running, batches)

                for tile, image in zip(batch, encoded):
                    self.rendered += 1

                    yield RenderedTile.build(tile, image, self.params.tile_width)

        finally:
            for _, task in running:
                task.cancel()


    async def close(self) -> None:
        """Release what the session holds."""


    @property
    def _parallel(self) -> int:
        """How many batches may render at once."""
        return 1


    def _batches(self, tiles: list[Tile]) -> list[list[Tile]]:
        """How the tiles split into render tasks (in order)."""
        return [tiles] if tiles else []


    def _top_up(self, running: deque[tuple[list[Tile], asyncio.Task]], batches: Iterator[list[Tile]]) -> None:
        """Start the next batches until `_parallel` of them are running."""
        for batch in islice(batches, self._parallel - len(running)):
            task = asyncio.ensure_future(self.render(batch))
            # A batch abandoned after an earlier one failed must not log its error.
            task.add_done_callback(lambda t: t.cancelled() or t.exception())
            running.append((batch, task))


class WorkerDocument(Document):
    """A PDF or an image: its reader does everything inside the render workers."""

    MAX_BATCH = 16  # tiles per task: bounds the result size per hop


    async def plan(self) -> list[Tile]:
        self._height, tiles = await self.pool.run(worker.plan, self.doc_type, self.path, bulk=self.bulk)

        return tiles


    async def render(self, tiles: list[Tile]) -> list[EncodedImage]:
        return await self.pool.run(worker.render, self.doc_type, self.path, tiles, bulk=self.bulk)


    @property
    def _parallel(self) -> int:
        """The worker slots this session may take: the bulk ones, or all of them."""
        s = self.pool.settings

        return s.bulk_render_workers if self.bulk else s.render_workers


    def _batches(self, tiles: list[Tile]) -> list[list[Tile]]:
        """Even chunks across the parallel slots, at most MAX_BATCH tiles each; a
        document whose reader cannot split stays one task."""
        if not tiles or not READERS[self.doc_type].splits_batches:
            return super()._batches(tiles)

        size = max(1, min(self.MAX_BATCH, math.ceil(len(tiles) / self._parallel)))

        return [tiles[i:i + size] for i in range(0, len(tiles), size)]


class MhtmlDocument(Document):
    """An MHTML archive: Chromium captures on the event loop, the workers encode.

    The page is `TILE_WIDTH` CSS px wide. Up to `SAFE_FULLPAGE_PX` tall it is captured
    whole (and the capture cached per document, so a doc's tiles cost one render);
    beyond it each tile is clip-captured on its own. The tab opens lazily — a cached
    capture needs none — and closes with the session.
    """


    def __init__(
        self,
        pool: RenderPool,
        doc_type: DocType,
        path: str,
        *,
        bulk: bool = False,
        height: int | None = None,
    ):

        super().__init__(pool, doc_type, path, bulk=bulk, height=height)
        self._stack = contextlib.AsyncExitStack()
        self._tab: Tab | None = None


    async def __aenter__(self) -> MhtmlDocument:
        await self._stack.enter_async_context(self.pool.browser_slot(self.bulk))

        return self


    async def plan(self) -> list[Tile]:
        async with self.pool.browser_errors():
            height = await (await self._opened()).height()

        # A page captured whole is as tall as its capture, not as its CSS height.
        if height <= self.pool.settings.safe_fullpage_px:
            png = await self._full_page()
            height = await self.pool.run(render.png_page_height, png, self.params.tile_width, bulk=self.bulk)

        self._height = height

        return self.params.tiling.cut(height)


    async def render(self, tiles: list[Tile]) -> list[EncodedImage]:
        p = self.params

        if self._whole:
            png = await self._full_page()
            rows = [(tile.offset, tile.offset + tile.height) for tile in tiles]

            return await self.pool.run(render.crop_page_png, png, rows, p.tile_width, p.jpeg_quality, bulk=self.bulk)

        encoded = []

        for tile in tiles:
            async with self.pool.browser_errors():
                png = await (await self._opened()).clip(tile.offset, tile.height)

            encoded.append(await self.pool.run(render.encode_strip, png, p.tile_width, p.jpeg_quality, bulk=self.bulk))

        return encoded


    async def close(self) -> None:
        """Close the tab, if one was opened, then give the bulk slot back."""
        await self._stack.aclose()
        self._tab = None


    @property
    def _whole(self) -> bool:
        """Whether the page is captured whole (else tile by tile). By the document's
        height alone, so ingest and a tile request always take the same path."""
        return self.height <= self.pool.settings.safe_fullpage_px


    def _batches(self, tiles: list[Tile]) -> list[list[Tile]]:
        """A whole-page capture encodes in one task; clip captures go one by one."""
        return super()._batches(tiles) if self._whole else [[tile] for tile in tiles]


    async def _opened(self) -> Tab:
        """The archive loaded in a Chromium tab, opened on first use."""
        if self._tab is None:
            self._tab = await self._stack.enter_async_context(self.pool.browser.open_mhtml(self.path))

        return self._tab


    async def _full_page(self) -> bytes:
        """The whole page as one PNG, cached per document."""
        async with self.pool.browser_errors():
            return await self.pool.page_cache.get_or_create(self.path, self._capture_full_page)


    async def _capture_full_page(self) -> bytes:
        return await (await self._opened()).full_png()


# The document class of each stored document type.
DOCUMENTS: dict[DocType, type[Document]] = {
    DocType.PDF: WorkerDocument,
    DocType.IMG: WorkerDocument,
    DocType.MHTML: MhtmlDocument,
}
