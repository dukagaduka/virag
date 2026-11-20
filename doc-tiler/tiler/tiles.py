"""Tiles: the tiles of a stored document, by its id.

Tiles are never persisted. The store keeps a tile's place in the document image
(`tiles.offset`, `tiles.height`) and the hash of its bytes; the blob and a deterministic
render give the bytes back — one tile for /get_tile, or the whole document as the event
stream its ingest produced. A tile that does not hash to what was stored is refused
(RENDER_MISMATCH).
"""

from __future__ import annotations

import logging
from contextlib import aclosing
from typing import Any, AsyncIterator

from . import metrics
from .config import Settings
from .errors import bad_request, conflict, gone, not_found
from .hashing import sha256_hex
from .render.geometry import Tile
from .render.renderer import Renderer
from .render.types import RenderedTile
from .store.blobs import BlobStore
from .store.cache import AsyncLRU
from .store.meta import DocRow, MetaStore, TileRow

log = logging.getLogger("tiler.tiles")


class Tiles:
    """Serves the tiles of stored documents, rendering them from the blobs on demand;
    owns the tile cache."""


    def __init__(self, store: MetaStore, blobs: BlobStore, renderer: Renderer, settings: Settings):
        self.store    = store
        self.blobs    = blobs
        self.renderer = renderer

        # Final JPEG bytes by (doc_id, tile_num).
        self.cache = AsyncLRU(settings.render_cache_size, "tile")


    async def get(self, doc_id: str, tile_num: int, verify: bool) -> tuple[bytes, TileRow, DocRow]:
        """Return a single tile's bytes, rendering and caching it on demand."""
        row = await self._served(doc_id)

        if tile_num < 0 or tile_num >= row.tile_count:
            raise bad_request(
                f"tile_num {tile_num} out of range (tile_count={row.tile_count})",
                "TILE_OUT_OF_RANGE",
                doc_id=doc_id,
                tile_count=row.tile_count,
            )

        tile_row = await self.store.get_tile(doc_id, tile_num)

        if tile_row is None:  # pragma: no cover - implies a partial write
            raise not_found(f"tile {tile_num} of {doc_id!r} is missing", "UNKNOWN_TILE")

        key = (doc_id, tile_num)
        jpeg = await self.cache.get_or_create(key, lambda: self._render(row, tile_row))

        if verify:
            self._verify(row, tile_row, sha256_hex(jpeg))

        return jpeg, tile_row, row


    async def stream(self, doc_id: str) -> AsyncIterator[dict[str, Any]]:
        """The events of a doc the store serves, by its id: what the ingest of its bytes
        would emit, its tiles rendered again. Nothing of the doc changes."""
        row = await self._served(doc_id)

        async with aclosing(self.replay(row)) as events:
            async for event in events:
                yield event


    async def replay(self, row: DocRow, *, render: bool = True) -> AsyncIterator[dict[str, Any]]:
        """A stored doc as an ingest's events: `begin`, its tiles rendered again (unless
        `render` is off), then `committed`."""
        yield {
            "event": "begin",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
        }

        if render:
            stored = {tile_row.tile_num: tile_row for tile_row in await self.store.get_tiles(row.doc_id)}

            # One background render of the whole doc, each tile checked against its stored hash.
            async with aclosing(self._rendered(row, list(stored.values()), bulk=True)) as rendered:
                async for tile in rendered:
                    self._verify(row, stored[tile.tile_num], tile.image_hash)

                    yield {"event": "tile", "tile": tile}

        yield {
            "event": "committed",
            "doc_id": row.doc_id,
            "doc_type": row.doc_type,
            "source_url": row.source_url,
            "render_version": row.render_version,
            "height": row.height,
            "tile_count": row.tile_count,
            "result": "idempotent",
        }


    async def forget(self, doc_id: str) -> None:
        """Drop what is cached of a document — its tiles and its page capture: the
        document was rendered anew, or is gone."""
        await self.cache.invalidate(lambda k: isinstance(k, tuple) and k[0] == doc_id)
        await self.renderer.forget(self.blobs.path(doc_id))


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    async def _served(self, doc_id: str) -> DocRow:
        """The row of a doc whose tiles are served: stored, complete, not quarantined."""
        row = await self.store.get_doc(doc_id)

        if row is None:
            raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)

        if not row.complete:
            raise not_found(
                f"doc_id {doc_id!r} is still being ingested", "DOC_INCOMPLETE", doc_id=doc_id
            )

        if row.quarantined:
            raise gone(
                f"doc_id {doc_id!r} is quarantined: {row.quarantine_reason}",
                doc_id=doc_id,
                reason=row.quarantine_reason,
            )

        return row


    async def _render(self, row: DocRow, tile_row: TileRow) -> bytes:
        """Render one tile from the stored blob: the rows it was cut from at ingest."""
        async with aclosing(self._rendered(row, [tile_row], bulk=False)) as rendered:
            [tile] = [tile async for tile in rendered]

        return tile.jpeg


    async def _rendered(
        self, row: DocRow, tile_rows: list[TileRow], *, bulk: bool
    ) -> AsyncIterator[RenderedTile]:
        """Tiles of a stored doc rendered in one render, in tile order, as they were cut
        at ingest."""
        self.blobs.require(row.doc_id)
        cut = [Tile(tile_row.tile_num, tile_row.offset, tile_row.height) for tile_row in tile_rows]

        async with self.renderer.open(
            row.doc_type, self.blobs.path(row.doc_id), purpose="regenerate", bulk=bulk, height=row.height
        ) as doc:
            async with aclosing(doc.tiles(cut)) as tiles:
                async for tile in tiles:
                    yield tile


    @staticmethod
    def _verify(row: DocRow, tile_row: TileRow, actual: str) -> None:
        """Refuse a rendered tile whose hash is not the one stored at ingest."""
        if actual == tile_row.image_hash:
            return

        # Every served doc was rendered by this render_version (revision guarantees
        # it), so a mismatch here is non-determinism — a bug, never expected drift.
        metrics.MISMATCHES.labels(doc_type=row.doc_type).inc()

        message = (
            f"regenerated tile ({row.doc_id}, {tile_row.tile_num}) hashes to {actual} but "
            f"{tile_row.image_hash} was stored at render_version {row.render_version}"
        )
        log.warning("%s — refusing", message)

        raise conflict(message, "RENDER_MISMATCH", doc_id=row.doc_id, tile_num=tile_row.tile_num)
