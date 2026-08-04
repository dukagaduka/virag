"""Ingest a document, tile it, and optionally store it."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile

from ..errors import bad_request
from ..models import IngestManifest
from .deps import ERROR_RESPONSES, check_declared_size, parse_metadata_field, read_capped, service

router = APIRouter()


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
    doc_type = (doc_type or "").strip().upper()
    if doc_type not in ("PDF", "URL", "MHTML", "IMG"):
        raise bad_request(
            f"unknown doc_type {doc_type!r}; expected PDF, URL, MHTML or IMG", "UNKNOWN_DOC_TYPE"
        )

    check_declared_size(request, svc.settings.max_doc_bytes)
    data = None
    if document is not None and doc_type != "URL":
        data = await read_capped(document, svc.settings.max_doc_bytes)
    if doc_type == "URL" and document is not None:
        # A URL ingest fetches its own bytes; an attached file would be ambiguous.
        raise bad_request(
            "doc_type=URL takes 'url' only; do not attach a document", "AMBIGUOUS_INPUT"
        )

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
