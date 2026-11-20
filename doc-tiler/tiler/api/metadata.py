"""Read, replace and merge a document's metadata object."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Request

from ..errors import not_found, unprocessable
from ..store.meta import DocRow
from .deps import ERROR_RESPONSES, check_doc_id

router = APIRouter()


async def _require_doc(request: Request, doc_id: str) -> DocRow:
    """Return the doc's row, raising 404 if the doc is unknown."""
    check_doc_id(doc_id)
    row = await request.app.state.store.get_doc(doc_id)

    if row is None:
        raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)

    return row


@router.get(
    "/doc/{doc_id}/metadata",
    responses=ERROR_RESPONSES,
    summary="Read the metadata object for a document.",
    operation_id="getMetadata",
)
async def get_metadata(request: Request, doc_id: str) -> dict[str, Any]:
    """Read the metadata object for a document."""
    row = await _require_doc(request, doc_id)

    return row.metadata


@router.put(
    "/doc/{doc_id}/metadata",
    responses=ERROR_RESPONSES,
    summary="Replace the whole metadata object.",
    operation_id="replaceMetadata",
)
async def replace_metadata(
    request: Request, doc_id: str, body: Annotated[dict[str, Any], Body()]
) -> dict[str, Any]:
    """Replace the whole metadata object with the validated payload."""
    await _require_doc(request, doc_id)

    state = request.app.state
    validated = state.schema.validated(body)
    await state.store.set_metadata(doc_id, validated)

    return validated


@router.patch(
    "/doc/{doc_id}/metadata",
    responses=ERROR_RESPONSES,
    summary="Shallow-merge metadata (a null value deletes the key).",
    operation_id="mergeMetadata",
)
async def merge_metadata(
    request: Request, doc_id: str, body: Annotated[dict[str, Any], Body()]
) -> dict[str, Any]:
    """Shallow-merge the payload into existing metadata; a null value deletes a key."""
    row = await _require_doc(request, doc_id)

    if not isinstance(body, dict):  # pragma: no cover - FastAPI enforces the type
        raise unprocessable("metadata must be a JSON object", "INVALID_METADATA_JSON")

    merged = dict(row.metadata)

    for key, value in body.items():
        if value is None:
            merged.pop(key, None)  # a null value deletes the key
        else:
            merged[key] = value

    state = request.app.state
    validated = state.schema.validated(merged)
    await state.store.set_metadata(doc_id, validated)

    return validated
