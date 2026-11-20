"""Ingest a document into the store; on request the response streams its tiles."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..errors import bad_request
from ..models import DocType, IngestResult
from ..store.blobs import Incoming
from .deps import (
    ERROR_RESPONSES,
    capped_chunks,
    check_declared_size,
    ndjson_response,
    parse_metadata_field,
)

router = APIRouter()


@router.post(
    "/doc",
    response_model=IngestResult,
    responses={
        200: {
            "description": "The stored doc's summary; with `tiles=true`, an NDJSON stream "
                           "of `begin`, a `tile` per tile and `committed`.",
            "content": {"application/x-ndjson": {}},
        },
        **ERROR_RESPONSES,
    },
    summary="Ingest a document into the store; `tiles=true` streams its tiles back.",
    operation_id="ingestDocument",
)
async def ingest_document(
    request: Request,
    doc_type: Annotated[str, Form()],
    document: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form()] = None,
    tiles: Annotated[bool, Query()] = False,
) -> IngestResult | StreamingResponse:
    """Ingest a document into the store and return its identity and tile count.

    With `tiles=true` the same ingest answers as an NDJSON stream — `begin`, a `tile`
    per tile with its bytes, then `committed`. What fails before `begin` is an HTTP
    error; from `begin` on errors surface as an in-stream `error` event, and the
    absence of `committed` means the ingest did not complete.
    """
    state = request.app.state
    doc_type = _validate_ingest_inputs(doc_type, document)

    check_declared_size(request, state.settings.max_doc_bytes)

    meta_obj = parse_metadata_field(metadata)
    upload = await _spool(request, doc_type, document)

    if tiles:
        return await ndjson_response(
            state.ingester.ingest_stream(doc_type=doc_type, upload=upload, url=url, metadata=meta_obj),
            tile_width=state.settings.tile_width,
        )

    result = await state.ingester.ingest(doc_type=doc_type, upload=upload, url=url, metadata=meta_obj)

    return IngestResult(**result)


def _validate_ingest_inputs(doc_type: str, document: UploadFile | None) -> DocType:
    """Pre-flight of an ingest; returns the normalized doc_type."""
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


async def _spool(request: Request, doc_type: DocType, document: UploadFile | None) -> Incoming | None:
    """Save the uploaded document to disk before any work starts on it; the ingest
    takes the spooled upload over. A URL ingest brings no document."""
    if document is None or doc_type == DocType.URL:
        return None

    state = request.app.state

    return await state.blobs.spool(capped_chunks(document, state.settings.max_doc_bytes))
