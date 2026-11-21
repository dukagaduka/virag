"""Events: what a document's tile stream consists of.

An ingest and the stream of a stored doc emit the same events, in this order:

  `Begin`      the document's identity — first, so a consumer can refer to the doc
               before any tile arrives;
  `Tile`       one per tile, in tile order, carrying the rendered tile;
  `Committed`  last: the doc is in the store, whole (its summary is what a plain ingest
               answers with).

A stream that fails after `Begin` ends with an `Error` instead of `Committed` — on the
wire only (the API puts it there); inside the service a failure is an exception.

`line()` is an event as it goes on the wire: one line of NDJSON, `{"event": <name>, …}`.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal

import orjson

from .models import DocType
from .render.types import RenderedTile


class Event:
    """One event of a document's tile stream."""

    name: ClassVar[str]


    def payload(self) -> dict[str, Any]:
        """The event's fields as they go on the wire."""
        return asdict(self)  # type: ignore[call-overload]


    def line(self) -> bytes:
        """The event as one NDJSON line."""
        return orjson.dumps({"event": self.name, **self.payload()}) + b"\n"


@dataclass(frozen=True)
class Begin(Event):
    """The document's identity; the doc is known to the store from here on."""

    name = "begin"

    doc_id: str
    doc_type: DocType
    source_url: str | None
    render_version: str


    def committed(self, *, height: int, tile_count: int, result: Literal["rendered", "idempotent"]) -> Committed:
        """The event this stream ends with once the doc is in the store."""
        return Committed(**asdict(self), height=height, tile_count=tile_count, result=result)


@dataclass(frozen=True)
class Tile(Event):
    """One rendered tile; on the wire its bytes are base64 JPEG."""

    name = "tile"

    tile: RenderedTile


    def payload(self) -> dict[str, Any]:
        return {
            "tile_num": self.tile.tile_num,
            "width": self.tile.width,
            "height": self.tile.height,
            "image_hash": self.tile.image_hash,
            "data": base64.b64encode(self.tile.jpeg).decode("ascii"),
        }


@dataclass(frozen=True)
class Committed(Event):
    """The doc is in the store: `rendered` by this ingest, or held already (`idempotent`)."""

    name = "committed"

    doc_id: str
    doc_type: DocType
    source_url: str | None
    render_version: str
    height: int
    tile_count: int
    result: Literal["rendered", "idempotent"]


@dataclass(frozen=True)
class Error(Event):
    """The stream failed after `Begin`: there will be no `Committed`."""

    name = "error"

    code: str
    message: str
