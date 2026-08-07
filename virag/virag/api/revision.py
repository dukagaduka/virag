from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from virag.reconcile import revise


log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/revision",
    summary="Re-embed the whole corpus into a fresh collection and blue-green swap the alias",
    operation_id="reviseCollection",
)
async def revision(request: Request):
    store = request.app.state.store
    embeddings = request.app.state.embeddings
    tiler = request.app.state.tiler

    log.info("revision requested")
    return await revise(store, tiler, embeddings)
