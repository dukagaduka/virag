"""Shared request helpers for the API modules."""

from __future__ import annotations

import base64
import hmac
import logging
from contextlib import aclosing
from typing import Any, AsyncIterator

import orjson
from fastapi import Request, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..errors import ApiError, not_found, not_ready, too_large, unauthorized, unprocessable
from ..hashing import is_doc_id

log = logging.getLogger("tiler.api")

_CHUNK = 1 << 20

# auto_error=False: a missing header answers our own 401 envelope, not FastAPI's 403.
_BEARER = HTTPBearer(auto_error=False, description="TILER_API_KEY")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Bad request."},
    401: {"description": "Missing or invalid API key (TILER_API_KEY)."},
    403: {"description": "Blocked (e.g. SSRF)."},
    404: {"description": "Not found."},
    409: {"description": "Render non-determinism, a revision already running, or a doc being ingested."},
    410: {"description": "Document quarantined by a revision."},
    413: {"description": "Payload too large."},
    415: {"description": "Content-type does not match declared doc_type."},
    422: {"description": "Corrupt document or invalid metadata."},
    502: {"description": "Upstream fetch/render failure."},
    503: {"description": "Not ready: a revision is pending or running."},
    504: {"description": "Render timeout."},
}


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """Router dependency: a request must carry `Authorization: Bearer <TILER_API_KEY>`;
    an empty TILER_API_KEY leaves the routes open."""
    expected = request.app.state.settings.api_key

    if not expected:
        return

    if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise unauthorized("missing or invalid API key")


def require_serving(request: Request) -> None:
    """Router dependency: data routes answer 503 while the revision gate is closed."""
    revision = request.app.state.revision

    if not revision.serving:
        raise not_ready(
            "not serving: stored docs are being brought to the current render_version "
            "(see GET /revision)",
            revision=revision.status,
        )


def check_doc_id(doc_id: str) -> str:
    """Validate a doc_id's shape, raising 404 if it is malformed."""
    if not is_doc_id(doc_id):
        # A malformed id can never exist in the store; 404 keeps the contract simple.
        raise not_found(f"unknown doc_id {doc_id!r}", "UNKNOWN_DOC", doc_id=doc_id)

    return doc_id


def check_declared_size(request: Request, limit: int) -> None:
    """Reject oversized uploads from the Content-Length header before reading them."""
    raw = request.headers.get("content-length")

    if raw and raw.isdigit() and int(raw) > limit:
        raise too_large(
            f"request body of {raw} bytes exceeds TILER_MAX_DOC_BYTES={limit}",
            declared_bytes=int(raw),
            limit=limit,
        )


async def capped_chunks(upload: UploadFile, limit: int) -> AsyncIterator[bytes]:
    """An upload chunk by chunk, raising 413 once it exceeds the byte limit."""
    total = 0

    while chunk := await upload.read(_CHUNK):
        total += len(chunk)

        if total > limit:
            raise too_large(
                f"document exceeds TILER_MAX_DOC_BYTES={limit}", read_bytes=total, limit=limit
            )

        yield chunk


async def ndjson_response(events: AsyncIterator[dict[str, Any]], *, tile_width: int) -> StreamingResponse:
    """A document's events (`begin`, a `tile` per tile, `committed`) as an NDJSON
    response, each tile with its bytes.

    The first event is awaited before the response starts, so whatever fails up to it
    keeps its HTTP status. From then on the response is already 200 and a failure goes
    in-band, as an `error` event; the absence of `committed` is the client's signal to
    roll back.
    """
    first = await anext(events)

    def line(event: dict[str, Any]) -> bytes:
        if event["event"] == "tile":
            tile = event["tile"]
            event = {
                "event": "tile",
                "tile_num": tile.tile_num,
                "width": tile_width,  # always TILE_WIDTH
                "height": tile.height,
                "image_hash": tile.image_hash,
                "data": base64.b64encode(tile.jpeg).decode("ascii"),
            }

        return orjson.dumps(event) + b"\n"

    async def lines() -> AsyncIterator[bytes]:
        try:
            # Closed with the response: a client that goes away ends the work at once.
            async with aclosing(events):
                yield line(first)

                async for event in events:
                    yield line(event)

        except ApiError as exc:
            yield line({"event": "error", "code": exc.code, "message": exc.message})

        except Exception as exc:  # noqa: BLE001 - the response is already 200; report in-band
            log.exception("tile stream crashed")
            yield line({"event": "error", "code": "INTERNAL_ERROR", "message": repr(exc)})

    return StreamingResponse(lines(), media_type="application/x-ndjson")


def parse_metadata_field(raw: str | None) -> dict[str, Any] | None:
    """Parse the optional metadata form field into a JSON object, or None."""
    if raw is None or raw == "":
        return None

    try:
        value = orjson.loads(raw)

    except orjson.JSONDecodeError as exc:
        raise unprocessable(f"metadata is not valid JSON: {exc}", "INVALID_METADATA_JSON") from exc

    if not isinstance(value, dict):
        raise unprocessable("metadata must be a JSON object", "INVALID_METADATA_JSON")

    return value
