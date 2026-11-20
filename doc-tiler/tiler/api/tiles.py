"""A stored document's tiles, rendered from its original: one of them, or the whole stream."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Query, Request, Response
from fastapi.responses import StreamingResponse

from .deps import ERROR_RESPONSES, ndjson_response

router = APIRouter()


@router.get(
    "/doc/{doc_id}/tiles",
    responses={200: {"content": {"application/x-ndjson": {}}}, **ERROR_RESPONSES},
    response_class=StreamingResponse,
    summary="Stream a stored document's tiles as NDJSON, as the ingest of its bytes would.",
    operation_id="streamDocumentTiles",
)
async def stream_document_tiles(request: Request, doc_id: str) -> StreamingResponse:
    """The event stream `POST /doc?tiles=true` answers to bytes the store already holds
    — `begin`, the tiles rendered again, `committed` — asked for by doc_id.

    A doc that is unknown, still being ingested or quarantined is an HTTP error; from
    `begin` on errors surface as an in-stream `error` event.
    """
    state = request.app.state

    return await ndjson_response(state.tiles.stream(doc_id), tile_width=state.settings.tile_width)


@router.get(
    "/doc/{doc_id}/tiles/{tile_num}",
    responses={200: {"content": {"image/jpeg": {}}}, **ERROR_RESPONSES},
    response_class=Response,
    summary="One tile of a stored document, as JPEG.",
    operation_id="getTile",
)
async def get_tile(
    request: Request,
    doc_id: str,
    tile_num: Annotated[int, Path(ge=0)],
    verify: Annotated[bool, Query()] = True,
) -> Response:
    """Render a single tile from the stored original and return it as JPEG."""
    state = request.app.state
    jpeg, tile_row, row = await state.tiles.get(doc_id, tile_num, verify)

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Doc-Id": doc_id,
            "X-Tile-Num": str(tile_num),
            "X-Image-Hash": tile_row.image_hash,
            "X-Render-Version": row.render_version,
            "X-Tile-Height": str(tile_row.height),
            "X-Tile-Width": str(state.settings.tile_width),
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
