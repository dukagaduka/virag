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
    retriever = request.app.state.retriever
    tiler = request.app.state.tiler
    enroll_settings = request.app.state.settings.enroll_settings

    log.info("revision requested")
    return await revise(store, tiler, retriever, enroll_settings)
