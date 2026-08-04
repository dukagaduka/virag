"""Read, replace and merge a document's metadata object."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Request

from ..errors import not_found, unprocessable
from .deps import ERROR_RESPONSES, check_doc_id, service

router = APIRouter()


async def _require_doc(request: Request, doc_id: str):
    """Return the service and doc row, raising 404 if the doc is unknown."""
    svc = service(request)
    check_doc_id(doc_id)
    row = await svc.store.get_doc(doc_id)
    if row is None:
        raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)
    return svc, row


@router.get(
    "/doc/{doc_id}/metadata",
    responses=ERROR_RESPONSES,
    summary="Read the metadata object for a document.",
    operation_id="getMetadata",
)
async def get_metadata(request: Request, doc_id: str) -> dict[str, Any]:
    """Read the metadata object for a document."""
    _, row = await _require_doc(request, doc_id)
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
    svc, _ = await _require_doc(request, doc_id)
    validated = svc.validate_metadata_payload(body)
    await svc.store.set_metadata(doc_id, validated)
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
    svc, row = await _require_doc(request, doc_id)
    if not isinstance(body, dict):  # pragma: no cover - FastAPI enforces the type
        raise unprocessable("metadata must be a JSON object", "INVALID_METADATA_JSON")
    merged = dict(row.metadata)
    for key, value in body.items():
        if value is None:
            merged.pop(key, None)  # a null value deletes the key
        else:
            merged[key] = value
    validated = svc.validate_metadata_payload(merged)
    await svc.store.set_metadata(doc_id, validated)
    return validated
