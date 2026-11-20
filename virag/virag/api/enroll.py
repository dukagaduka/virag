from __future__ import annotations

import logging

import orjson
from fastapi import APIRouter, File, Form, Request, UploadFile

from virag.errors import ApiError


log = logging.getLogger(__name__)

router = APIRouter()


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
    allow_missing_name: bool = False,
):

    enroller = request.app.state.enroller
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
    log.info("enroll request: doc_type=%s filename=%s", doc_type, filename)

    events = tiler.tile_stream(doc_type, document=doc_bytes, url=url, metadata=meta_obj)
    summary = await enroller.enroll(events, metadata=meta_obj)

    if not has_name:
        log.warning("no metadata.name; doc_id=%s goes by its content hash", summary["doc_id"])

    return summary


def _has_name(meta_obj: dict | None) -> bool:
    """True iff metadata carries a non-empty string `name`."""
    name = meta_obj.get("name") if meta_obj else None

    return isinstance(name, str) and name.strip() != ""


def _parse_metadata(raw: str | None) -> dict | None:
    """Parse the metadata form field (a JSON object) into a dict, or None if absent."""
    if raw is None:
        return None

    try:
        obj = orjson.loads(raw)

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
