from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile

from virag.clients.tiler_client import TileResult
from virag.errors import ApiError


log = logging.getLogger(__name__)

router = APIRouter()

_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "virag.pixelrag")


def point_id(doc_id: str, tile_num: int) -> str:
    """Deterministic point id so re-enrolling a document overwrites its own tiles."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{tile_num}"))


@router.post(
    "/enroll",
    summary="Tile a document, embed its tiles, and upsert them into the vector store",
    operation_id="enrollDocument",
)
async def enroll(
    request: Request,
    doc_type: str = Form(...),
    document: UploadFile | None = File(None),
    url: str | None = Form(None),
    metadata: str | None = Form(None),
    force: bool = False,
    allow_missing_name: bool = False,
):

    store = request.app.state.store
    embeddings = request.app.state.embeddings
    tiler = request.app.state.tiler

    meta_obj = _parse_metadata(metadata)
    has_name = _has_name(meta_obj)

    # Service policy: every enrolled document is named. The opt-out is explicit so
    # the caller knowingly diverges — it names the document by its content hash.
    if not has_name and not allow_missing_name:
        raise ApiError(
            status_code=422,
            message="Each enrolled document must include metadata.name. "
                    "Set allow_missing_name=true to fall back to the document content hash.",
        )

    doc_bytes = await document.read() if document is not None else None

    filename = document.filename if document is not None else None
    log.info("enroll request: doc_type=%s filename=%s force=%s", doc_type, filename, force)

    result = await tiler.tile(
        doc_type,
        document=doc_bytes,
        url=url,
        metadata=meta_obj,
        store=True,
        include_bytes=True,
        force=force,
    )

    log.info("tiled: doc_id=%s tile_count=%d", result.doc_id, result.tile_count)

    if has_name:
        payload_meta = meta_obj
    else:
        payload_meta = {**(meta_obj or {}), "name": result.doc_id}
        log.warning("no metadata.name; naming doc_id=%s by its content hash", result.doc_id)

    if result.tile_count == 0:
        log.warning("no tiles produced for doc_id=%s; nothing enrolled", result.doc_id)

        return _summary(result, enrolled=0)

    vectors = await embeddings.embed_images([tile.data for tile in result.tiles])

    log.info("embedded: doc_id=%s n=%d dim=%d", result.doc_id, len(vectors), len(vectors[0]))

    payloads = [
        {
            "doc_id": result.doc_id,
            "tile_num": tile.tile_num,
            "image_hash": tile.image_hash,
            "doc_type": result.doc_type,
            "source_url": result.source_url,
            "render_version": result.render_version,
            "metadata": payload_meta,
        }
        for tile in result.tiles
    ]
    ids = [point_id(result.doc_id, tile.tile_num) for tile in result.tiles]

    await store.upsert(vectors, payloads, ids=ids)

    log.info("upserted: doc_id=%s points=%d", result.doc_id, len(ids))

    return _summary(result, enrolled=len(ids))


def _has_name(meta_obj: dict | None) -> bool:
    """True iff metadata carries a non-empty string `name`."""
    name = meta_obj.get("name") if meta_obj else None

    return isinstance(name, str) and name.strip() != ""


def _parse_metadata(raw: str | None) -> dict | None:
    """Parse the metadata form field (a JSON object) into a dict, or None if absent."""
    if raw is None:
        return None

    try:
        obj = json.loads(raw)

    except (ValueError, TypeError) as e:
        raise ApiError(
            status_code=422,
            message="`metadata` must be a valid JSON object",
            details={"base_message": str(e)},
        ) from e

    if not isinstance(obj, dict):
        raise ApiError(
            status_code=422,
            message="`metadata` must be a JSON object",
        )

    return obj


def _summary(result: TileResult, *, enrolled: int) -> dict:
    return {
        "doc_id": result.doc_id,
        "doc_type": result.doc_type,
        "render_version": result.render_version,
        "tile_count": result.tile_count,
        "enrolled": enrolled,
    }
