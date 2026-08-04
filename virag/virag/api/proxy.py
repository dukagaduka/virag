from __future__ import annotations

import httpx

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from virag.errors import ApiError


router = APIRouter()

# Hop-by-hop headers are connection-specific and must not be forwarded (RFC 7230).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

_QDRANT_TIMEOUT = 30


@router.api_route("/tiler/{path:path}", methods=_METHODS)
async def tiler_proxy(path: str, request: Request):
    ts = request.app.state.settings.tiler_settings

    return await _reverse_proxy(request, f"{ts.base_url}/{path}", timeout=ts.timeout)


@router.api_route("/embed/{path:path}", methods=_METHODS)
async def embed_proxy(path: str, request: Request):
    es = request.app.state.settings.embeddings_settings

    return await _reverse_proxy(request, f"{es.base_url}/{path}", timeout=es.timeout, auth_key=es.api_key)


@router.api_route("/qdrant/{path:path}", methods=_METHODS)
async def qdrant_proxy(path: str, request: Request):
    qs = request.app.state.settings.qdrant_settings

    return await _reverse_proxy(request, f"{qs.base_url}/{path}", timeout=_QDRANT_TIMEOUT)


@router.api_route("/v1/{path:path}", methods=_METHODS)
async def vlm_proxy(path: str, request: Request):
    """Every /v1/* except /v1/chat/completions (handled by its own router, registered
    first) is proxied straight to the upstream VLM."""
    vs = request.app.state.settings.vlm_settings

    return await _reverse_proxy(request, f"{vs.base_url}/{path}", timeout=vs.timeout, auth_key=vs.api_key)


async def _reverse_proxy(
    request: Request,
    target_url: str,
    *,
    timeout: float | httpx.Timeout,
    auth_key: str | None = None,
) -> StreamingResponse:

    query = request.url.query
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()
    headers = _forward_request_headers(request, auth_key)

    client = httpx.AsyncClient(timeout=timeout)
    upstream = client.build_request(request.method, target_url, headers=headers, content=body)

    try:
        response = await client.send(upstream, stream=True)

    except httpx.RequestError as e:
        await client.aclose()
        raise ApiError(
            status_code=502,
            message=f"Cannot reach upstream {target_url}",
            details={"error": repr(e)},
        ) from e

    async def body_stream():
        try:
            async for chunk in response.aiter_raw():
                yield chunk

        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        body_stream(),
        status_code=response.status_code,
        headers=_forward_response_headers(response),
    )


def _forward_request_headers(request: Request, auth_key: str | None) -> dict:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() not in ("host", "content-length")
    }

    if auth_key is not None:
        headers["Authorization"] = f"Bearer {auth_key}"

    return headers


def _forward_response_headers(response: httpx.Response) -> dict:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() != "content-length"
    }
