"""Regenerate a single tile from the stored original."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

from .deps import ERROR_RESPONSES, service

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
    jpeg, tile_row = await svc.get_tile(doc_id, tile_num, verify)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Doc-Id": doc_id,
            "X-Tile-Num": str(tile_num),
            "X-Image-Hash": tile_row.image_hash,
            "X-Tile-Height": str(tile_row.height),
            "X-Tile-Width": str(svc.settings.tile_width),
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
