"""Observe and start the revision that brings stored docs to the current render_version."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from ..responses import OrjsonResponse
from .deps import ERROR_RESPONSES, service

router = APIRouter()


@router.get(
    "/revision",
    summary="Status of the serving gate and of the current or last revision.",
    operation_id="revisionStatus",
)
async def revision_status(request: Request) -> dict[str, Any]:
    """Report the gate (`serving`) and the job's state and progress."""
    return service(request).revision.status


@router.post(
    "/revision",
    status_code=202,
    responses=ERROR_RESPONSES,
    summary="Start a background revision; the service is not-ready until it finishes.",
    operation_id="startRevision",
)
async def start_revision(
    request: Request,
    force: Annotated[bool, Query()] = False,
) -> OrjsonResponse:
    """Re-render, from their stored blobs, the docs rendered by another render_version
    (every doc with `force`). Data routes answer 503 NOT_READY until it finishes; a doc
    that cannot be re-rendered is quarantined. 409 while a revision is running."""
    return OrjsonResponse(status_code=202, content=service(request).revision.start(force))
