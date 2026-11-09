from __future__ import annotations

from fastapi import APIRouter, Request


router = APIRouter()


@router.get(
    "/v1/models",
    summary="OpenAI-compatible model list: the one VLM model virag answers with",
    operation_id="listModels",
)
async def list_models(request: Request):
    return {"object": "list", "data": [await request.app.state.vlm.model_card()]}
