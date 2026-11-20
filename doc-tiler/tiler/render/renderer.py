"""Renderer: the one way from a document's file to its tiles.

Whoever needs tiles — an ingest rendering an upload, a revision rendering a stored doc
anew, a request for a stored doc's tiles — opens the document here and reads them:

    async with renderer.open(doc_type, path, purpose="ingest", bulk=True) as doc:
        async for tile in doc.tiles():
            ...

They differ only in what they pass. `bulk` marks background work, held to the pool's
bulk slots so that an interactive request never queues behind it. `doc.tiles()` measures
the document and cuts it into tiles; `doc.tiles(cut)` renders the tiles of a cut made
before — a stored doc's, whose `height` is known as well and spares the measuring. What
becomes of the tiles is the caller's business.

Every render is recorded in the metrics under its `purpose`.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from .. import metrics
from ..models import DocType
from .documents import DOCUMENTS, Document
from .pool import RenderPool

Purpose = Literal["ingest", "revise", "regenerate"]


class Renderer:
    """Opens documents for rendering in the RenderPool and records their renders."""


    def __init__(self, pool: RenderPool):
        self.pool = pool


    @asynccontextmanager
    async def open(
        self,
        doc_type: DocType,
        path: str | Path,
        *,
        purpose: Purpose,
        bulk: bool = False,
        height: int | None = None,
    ) -> AsyncIterator[Document]:
        """The document in the file at `path` (a stored blob or a spooled upload),
        opened for rendering: a session, closed on exit.

        Its wall time is recorded whatever its outcome; the render and its tiles once it
        succeeded.
        """
        doc = DOCUMENTS[doc_type](self.pool, doc_type, str(path), bulk=bulk, height=height)
        started = time.perf_counter()

        try:
            async with doc:
                yield doc

        finally:
            metrics.RENDER_SECONDS.labels(doc_type=doc_type, path=purpose).observe(time.perf_counter() - started)

        metrics.RENDERS.labels(doc_type=doc_type, path=purpose).inc()
        metrics.TILES.labels(doc_type=doc_type, path=purpose).inc(doc.rendered)


    async def forget(self, path: str | Path) -> None:
        """Drop the page capture cached for the document in the file at `path`."""
        key = str(path)

        await self.pool.page_cache.invalidate(lambda k: k == key)
