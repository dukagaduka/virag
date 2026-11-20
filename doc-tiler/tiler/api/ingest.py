"""Ingest a document into the store; the streaming variant also emits its tiles."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..errors import bad_request
from ..models import DocType, IngestResult
from ..service import TilerService
from ..store.blobs import Incoming
from .deps import (
    ERROR_RESPONSES,
    capped_chunks,
    check_declared_size,
    ndjson_response,
    parse_metadata_field,
    service,
)

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


async def _spool(svc: TilerService, doc_type: DocType, document: UploadFile | None) -> Incoming | None:
    """Save the uploaded document to disk before any work starts on it; the service
    takes the spooled upload over. A URL ingest brings no document."""
    if document is None or doc_type == DocType.URL:
        return None

    return await svc.blobs.spool(capped_chunks(document, svc.settings.max_doc_bytes))


@router.post(
    "/tile",
    response_model=IngestResult,
    responses=ERROR_RESPONSES,
    summary="Ingest a document into the store; the tiles are served by /get_tile.",
    operation_id="ingestDocument",
)
async def ingest_document(
    request: Request,
    doc_type: Annotated[str, Form()],
    document: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
) -> IngestResult:
    """Ingest a document into the store and return its identity and tile count."""
    svc = service(request)
    doc_type = _validate_ingest_inputs(doc_type, document)

    check_declared_size(request, svc.settings.max_doc_bytes)

    meta_obj = parse_metadata_field(metadata)
    upload = await _spool(svc, doc_type, document)

    result = await svc.ingest(doc_type=doc_type, upload=upload, url=url, metadata=meta_obj)

    return IngestResult(**result)


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
) -> StreamingResponse:
    """The `/tile` ingest, streamed tile by tile (NDJSON) and terminated by `committed`.

    What fails before `begin` is an HTTP error; from `begin` on errors surface as an
    in-stream `error` event, and the absence of `committed` means the ingest did not
    complete.
    """
    svc = service(request)
    doc_type = _validate_ingest_inputs(doc_type, document)

    check_declared_size(request, svc.settings.max_doc_bytes)

    meta_obj = parse_metadata_field(metadata)
    upload = await _spool(svc, doc_type, document)

    return await ndjson_response(
        svc.ingest_stream(doc_type=doc_type, upload=upload, url=url, metadata=meta_obj)
    )
