"""Read, replace and merge a document's metadata object."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Request

from .deps import ERROR_RESPONSES

router = APIRouter()


@router.get(
    "/doc/{doc_id}/metadata",
    responses=ERROR_RESPONSES,
    summary="Read the metadata object for a document.",
    operation_id="getMetadata",
)
async def get_metadata(request: Request, doc_id: str) -> dict[str, Any]:
    """Read the metadata object for a document."""
    row = await request.app.state.store.require_doc(doc_id)

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
    state = request.app.state

    await state.store.require_doc(doc_id)

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
    state = request.app.state

    row = await state.store.require_doc(doc_id)
    merged = dict(row.metadata)

    for key, value in body.items():
        if value is None:
            merged.pop(key, None)  # a null value deletes the key
        else:
            merged[key] = value

    validated = state.schema.validated(merged)
    await state.store.set_metadata(doc_id, validated)

    return validated
