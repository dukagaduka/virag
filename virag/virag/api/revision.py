from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from virag.responses import OrjsonResponse


log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/revision",
    summary="Start a background re-embed of the whole corpus into a fresh collection, then blue-green swap the alias",
    operation_id="reviseCollection",
    status_code=202,
)
async def revision(request: Request):
    log.info("revision requested")

    return OrjsonResponse(status_code=202, content=await request.app.state.revision.start())


@router.get(
    "/revision",
    summary="Status of the current or last revision / rollback job",
    operation_id="revisionStatus",
)
async def revision_status(request: Request):
    return request.app.state.revision.status


@router.post(
    "/revision/rollback",
    summary="Start a background rollback of the serving alias to the collection the last revision replaced",
    operation_id="rollbackRevision",
    status_code=202,
)
async def revision_rollback(request: Request):
    log.warning("revision rollback requested")

    return OrjsonResponse(status_code=202, content=await request.app.state.revision.start_rollback())
