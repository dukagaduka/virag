"""Shared request helpers for the API modules."""

from __future__ import annotations

import hmac
from typing import Any

import orjson
from fastapi import Request, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..errors import not_found, not_ready, too_large, unauthorized, unprocessable
from ..hashing import is_doc_id
from ..service import TilerService

_CHUNK = 1 << 20

# auto_error=False: a missing header answers our own 401 envelope, not FastAPI's 403.
_BEARER = HTTPBearer(auto_error=False, description="TILER_API_KEY")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Bad request."},
    401: {"description": "Missing or invalid API key (TILER_API_KEY)."},
    403: {"description": "Blocked (e.g. SSRF)."},
    404: {"description": "Not found."},
    409: {"description": "Render non-determinism, or a revision already running."},
    410: {"description": "Document quarantined by a revision."},
    413: {"description": "Payload too large."},
    415: {"description": "Content-type does not match declared doc_type."},
    422: {"description": "Corrupt document or invalid metadata."},
    502: {"description": "Upstream fetch/render failure."},
    503: {"description": "Not ready: a revision is pending or running."},
    504: {"description": "Render timeout."},
}


def service(request: Request) -> TilerService:
    """Return the TilerService held on the app state."""
    return request.app.state.service


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """Router dependency: a request must carry `Authorization: Bearer <TILER_API_KEY>`;
    an empty TILER_API_KEY leaves the routes open."""
    expected = service(request).settings.api_key

    if not expected:
        return

    if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise unauthorized("missing or invalid API key")


def require_serving(request: Request) -> None:
    """Router dependency: data routes answer 503 while the revision gate is closed."""
    revision = service(request).revision

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


async def read_capped(upload: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, raising 413 once it exceeds the byte limit."""
    chunks: list[bytes] = []
    total = 0

    while chunk := await upload.read(_CHUNK):
        total += len(chunk)

        if total > limit:
            raise too_large(
                f"document exceeds TILER_MAX_DOC_BYTES={limit}", read_bytes=total, limit=limit
            )

        chunks.append(chunk)

    return b"".join(chunks)


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
