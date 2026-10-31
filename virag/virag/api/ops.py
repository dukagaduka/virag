from __future__ import annotations

from fastapi import APIRouter, Request

from virag.responses import OrjsonResponse


router = APIRouter()

_DEPENDENCIES = ("store", "retriever", "tiler", "vlm")


@router.get("/health", summary="Liveness")
async def health():
    return {"status": "ok"}


@router.get("/ready", summary="Readiness: dependencies wired and the serving gate open")
async def ready(request: Request):
    state = request.app.state
    missing = [name for name in _DEPENDENCIES if not hasattr(state, name)]

    if missing:
        return OrjsonResponse(status_code=503, content={"status": "not ready", "missing": missing})

    gate = getattr(state, "gate", None)

    if gate is not None and not gate.serving:
        return OrjsonResponse(status_code=503, content={"status": "not ready", "reason": gate.reason})

    return {"status": "ready"}
