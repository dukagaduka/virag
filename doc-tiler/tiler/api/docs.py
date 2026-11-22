"""Download and delete a stored document's original blob (with HTTP Range support)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from ..errors import ApiError
from ..media import extension_for
from .deps import ERROR_RESPONSES

router = APIRouter()

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


@router.get(
    "/doc/{doc_id}",
    responses={200: {"content": {"application/octet-stream": {}}}, **ERROR_RESPONSES},
    summary="Download the original stored blob.",
    operation_id="downloadOriginal",
)
async def download_original(request: Request, doc_id: str) -> Response:
    """Stream the original stored blob, honouring a single Range header."""
    state = request.app.state

    # Its blob is not in the store before the ingest commits.
    row = await state.store.require_doc(doc_id, complete=True)
    state.blobs.require(doc_id)

    size = await state.blobs.size(doc_id)
    filename = f"{doc_id}{extension_for(row.mime_type)}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Accept-Ranges": "bytes",
        "X-Doc-Id": doc_id,
    }

    rng = request.headers.get("range")

    if rng:
        window = _parse_range(rng, size)

        if window is not None:
            start, end = window
            headers |= {
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(end - start + 1),
            }

            return StreamingResponse(
                state.blobs.stream(doc_id, start, end),
                status_code=206,
                media_type=row.mime_type,
                headers=headers,
            )

    headers["Content-Length"] = str(size)

    return StreamingResponse(
        state.blobs.stream(doc_id, 0, size - 1), media_type=row.mime_type, headers=headers
    )


@router.delete(
    "/doc/{doc_id}",
    status_code=204,
    responses=ERROR_RESPONSES,
    summary="Delete a document, its tile rows and its blob (flag-guarded).",
    operation_id="deleteDocument",
)
async def delete_document(request: Request, doc_id: str) -> Response:
    """Delete a document's rows and blob when deletion is enabled."""
    state = request.app.state

    if not state.settings.api_settings.allow_delete:
        raise ApiError(
            405, "DELETE_DISABLED", "deletion is disabled; start with TILER_ALLOW_DELETE=true"
        )

    await state.ingester.delete(doc_id)

    return Response(status_code=204)


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Single-range parser. Returns an inclusive `(start, end)` or None for the whole file."""
    match = _RANGE_RE.match(header.strip())

    if not match:
        return None

    raw_start, raw_end = match.groups()

    if raw_start == "" and raw_end == "":
        return None

    if raw_start == "":  # suffix range: last N bytes
        length = int(raw_end)

        if length <= 0:
            raise ApiError(416, "RANGE_NOT_SATISFIABLE", "invalid suffix range")

        start, end = max(0, size - length), size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
        end = min(end, size - 1)

    if start > end or start >= size:
        raise ApiError(416, "RANGE_NOT_SATISFIABLE", f"range not satisfiable for size {size}")

    return start, end
