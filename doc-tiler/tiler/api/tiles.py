"""Regenerate tiles from the stored original: one of them, or a document's whole stream."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse

from .deps import ERROR_RESPONSES, check_doc_id, ndjson_response, service

router = APIRouter()


@router.get(
    "/get_tile",
    responses={200: {"content": {"image/jpeg": {}}}, **ERROR_RESPONSES},
    response_class=Response,
    summary="Regenerate a single tile from the stored original.",
    operation_id="getTile",
)
async def get_tile(
    request: Request,
    doc_id: Annotated[str, Query()],
    tile_num: Annotated[int, Query(ge=0)],
    verify: Annotated[bool, Query()] = True,
) -> Response:
    """Regenerate a single tile from the stored original and return it as JPEG."""
    svc = service(request)
    jpeg, tile_row, row = await svc.get_tile(doc_id, tile_num, verify)

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Doc-Id": doc_id,
            "X-Tile-Num": str(tile_num),
            "X-Image-Hash": tile_row.image_hash,
            "X-Render-Version": row.render_version,
            "X-Tile-Height": str(tile_row.height),
            "X-Tile-Width": str(svc.settings.tile_width),
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.get(
    "/doc/{doc_id}/tiles",
    responses={200: {"content": {"application/x-ndjson": {}}}, **ERROR_RESPONSES},
    response_class=StreamingResponse,
    summary="Stream a stored document's tiles as NDJSON, as the ingest of its bytes would.",
    operation_id="streamDocumentTiles",
)
async def stream_document_tiles(request: Request, doc_id: str) -> StreamingResponse:
    """The event stream `/tile/stream` answers to bytes the store already holds —
    `begin`, the tiles regenerated, `committed` — asked for by doc_id.

    A doc that is unknown, still being ingested or quarantined is an HTTP error; from
    `begin` on errors surface as an in-stream `error` event.
    """
    svc = service(request)
    check_doc_id(doc_id)

    return await ndjson_response(svc.doc_stream(doc_id))
