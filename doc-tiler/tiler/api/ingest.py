"""Ingest a document into the store; on request the response streams its tiles."""

from __future__ import annotations

from typing import Annotated, Any, AsyncIterator

import orjson
from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..errors import bad_request, too_large, unprocessable
from ..models import DocType, IngestResult
from ..store.blobs import Incoming
from .deps import ERROR_RESPONSES, ndjson_response

router = APIRouter()

_CHUNK = 1 << 20  # bytes per read of an upload


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
    limit = state.settings.store_settings.max_doc_bytes

    doc_type = _doc_type(doc_type, document)
    _check_declared_size(request, limit)
    parsed = _metadata(metadata)
    upload = await _spool(request, doc_type, document, limit)

    if tiles:
        return await ndjson_response(
            state.ingester.stream(doc_type=doc_type, upload=upload, url=url, metadata=parsed)
        )

    committed = await state.ingester.ingest(doc_type=doc_type, upload=upload, url=url, metadata=parsed)

    return IngestResult(**committed.payload())


# --------------------------------------------------------------------------- #
# the form
# --------------------------------------------------------------------------- #
def _doc_type(doc_type: str, document: UploadFile | None) -> DocType:
    """The declared doc_type, normalized; 400 when it is unknown or contradicts the form."""
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


def _check_declared_size(request: Request, limit: int) -> None:
    """Reject an oversized upload by its Content-Length header, before reading it."""
    raw = request.headers.get("content-length")

    if raw and raw.isdigit() and int(raw) > limit:
        raise too_large(
            f"request body of {raw} bytes exceeds TILER_MAX_DOC_BYTES={limit}",
            declared_bytes=int(raw),
            limit=limit,
        )


def _metadata(raw: str | None) -> dict[str, Any] | None:
    """The optional `metadata` field as a JSON object, or None."""
    if raw is None or raw == "":
        return None

    try:
        value = orjson.loads(raw)

    except orjson.JSONDecodeError as exc:
        raise unprocessable(f"metadata is not valid JSON: {exc}", "INVALID_METADATA_JSON") from exc

    if not isinstance(value, dict):
        raise unprocessable("metadata must be a JSON object", "INVALID_METADATA_JSON")

    return value


async def _spool(
    request: Request, doc_type: DocType, document: UploadFile | None, limit: int
) -> Incoming | None:
    """Save the uploaded document to disk before any work starts on it; the ingest
    takes the spooled upload over. A URL ingest brings no document."""
    if document is None or doc_type == DocType.URL:
        return None

    return await request.app.state.blobs.spool(_capped_chunks(document, limit))


async def _capped_chunks(upload: UploadFile, limit: int) -> AsyncIterator[bytes]:
    """An upload chunk by chunk, raising 413 once it exceeds the byte limit."""
    total = 0

    while chunk := await upload.read(_CHUNK):
        total += len(chunk)

        if total > limit:
            raise too_large(
                f"document exceeds TILER_MAX_DOC_BYTES={limit}", read_bytes=total, limit=limit
            )

        yield chunk
