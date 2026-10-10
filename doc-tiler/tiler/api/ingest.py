"""Ingest a document, tile it, and optionally store it."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..errors import bad_request
from ..models import DocType, IngestManifest
from .deps import ERROR_RESPONSES, check_declared_size, parse_metadata_field, read_capped, service

router = APIRouter()


def _validate_ingest_inputs(doc_type: str, document: UploadFile | None) -> DocType:
    """Shared pre-flight for both ingest routes; returns the normalized doc_type."""
    raw = (doc_type or "").strip().upper()

    try:
        doc_type = DocType(raw)

    except ValueError:
        raise bad_request(
            f"unknown doc_type {raw!r}; expected PDF, URL, MHTML or IMG", "UNKNOWN_DOC_TYPE"
        ) from None

    if doc_type == DocType.URL and document is not None:
        # A URL ingest fetches its own bytes; an attached file would be ambiguous.
        raise bad_request(
            "doc_type=URL takes 'url' only; do not attach a document", "AMBIGUOUS_INPUT"
        )
    
    return doc_type


@router.post(
    "/tile",
    response_model=IngestManifest,
    responses=ERROR_RESPONSES,
    summary="Ingest a document, tile it, optionally store it, and return the tiles.",
    operation_id="ingestDocument",
)
async def ingest_document(
    request: Request,
    doc_type: Annotated[str, Form()],
    document: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    store: Annotated[bool, Query()] = True,
    include_bytes: Annotated[bool, Query()] = True,
    force: Annotated[bool, Query()] = False,
) -> IngestManifest:
    """Ingest a document, tile it, optionally store it, and return the tiles."""
    svc = service(request)
    doc_type = _validate_ingest_inputs(doc_type, document)

    check_declared_size(request, svc.settings.max_doc_bytes)

    data = None
    if document is not None and doc_type != DocType.URL:
        data = await read_capped(document, svc.settings.max_doc_bytes)

    manifest = await svc.ingest(
        doc_type=doc_type,
        data=data,
        url=url,
        metadata=parse_metadata_field(metadata),
        store=store,
        include_bytes=include_bytes,
        force=force,
    )
    
    return IngestManifest(**manifest)


@router.post(
    "/tile/stream",
    responses=ERROR_RESPONSES,
    summary="Ingest a document and stream its tiles as NDJSON, then a `committed` event.",
    operation_id="ingestDocumentStream",
)
async def ingest_document_stream(
    request: Request,
    doc_type: Annotated[str, Form()],
    document: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    force: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    """Stream a stored ingest tile-by-tile (NDJSON), terminated by a `committed` event.

    The doc is persisted incrementally and stays incomplete until the final tile, so a
    broken stream leaves a swept-able partial. Errors surface as an in-stream `error`
    event; the absence of `committed` means the ingest did not complete.
    """
    svc = service(request)
    doc_type = _validate_ingest_inputs(doc_type, document)

    check_declared_size(request, svc.settings.max_doc_bytes)
    data = None
    if document is not None and doc_type != DocType.URL:
        data = await read_capped(document, svc.settings.max_doc_bytes)

    meta_obj = parse_metadata_field(metadata)

    async def ndjson():
        async for event in svc.ingest_stream(
            doc_type=doc_type, data=data, url=url, metadata=meta_obj, force=force
        ):
            yield json.dumps(event).encode() + b"\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@router.post(
    "/revision",
    responses=ERROR_RESPONSES,
    summary="Re-render stored docs whose render_version is stale (or all, with force).",
    operation_id="reviseDocuments",
)
async def revise_documents(
    request: Request,
    force: Annotated[bool, Query()] = False,
) -> dict:
    """Re-render stored documents in place so tiles reproduce at the current render_version.

    Refreshes docs whose `render_version` differs from the tiler's current one (or every
    doc when `force`), plus any left incomplete. Returns `{revised, skipped, failed, ...}`.
    Qdrant is untouched — run virag's `/revision` afterwards to re-embed.
    """
    svc = service(request)
    return await svc.revise(force=force)
