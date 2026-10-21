from __future__ import annotations

import httpx2
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from virag.clients.http_service import HttpService
from virag.errors import ApiError
from virag.settings import HttpSettings, QdrantStoreSettings


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

# The Qdrant passthrough is a debug/ops surface, not a hot path: a small fixed pool.
_QDRANT_TIMEOUT = 30
_QDRANT_CONNECT_TIMEOUT = 10
_QDRANT_MAX_CONNECTIONS = 16


class QdrantProxyService(HttpService):
    """Raw REST access to Qdrant for the /qdrant passthrough; QdrantStore keeps its
    own client for the data path."""

    @classmethod
    def from_settings(cls, qs: QdrantStoreSettings, hs: HttpSettings) -> QdrantProxyService:
        return cls(
            base_url=qs.base_url,
            timeout=_QDRANT_TIMEOUT,
            connect_timeout=_QDRANT_CONNECT_TIMEOUT,
            max_connections=_QDRANT_MAX_CONNECTIONS,
            max_retries=0,
            backoff_base=hs.backoff_base,
            backoff_cap=hs.backoff_cap,
        )


    async def check_connection(self) -> None:
        await self._request("GET", f"{self.base_url}/readyz", action="the check")


@router.api_route("/tiler/{path:path}", methods=_METHODS)
async def tiler_proxy(path: str, request: Request):
    tiler = request.app.state.tiler

    return await _reverse_proxy(request, tiler, f"{tiler.base_url}/{path}")


@router.api_route("/embed/{path:path}", methods=_METHODS)
async def embed_proxy(path: str, request: Request):
    embeddings = request.app.state.embeddings

    return await _reverse_proxy(request, embeddings, f"{embeddings.base_url}/{path}", auth_key=embeddings.api_key)


@router.api_route("/qdrant/{path:path}", methods=_METHODS)
async def qdrant_proxy(path: str, request: Request):
    qdrant_http = request.app.state.qdrant_http

    return await _reverse_proxy(request, qdrant_http, f"{qdrant_http.base_url}/{path}")


@router.api_route("/v1/{path:path}", methods=_METHODS)
async def vlm_proxy(path: str, request: Request):
    """Every /v1/* except /v1/chat/completions (handled by its own router, registered
    first) is proxied straight to the upstream VLM."""
    vlm = request.app.state.vlm

    return await _reverse_proxy(request, vlm, f"{vlm.base_url}/{path}", auth_key=vlm.api_key)


async def _reverse_proxy(
    request: Request,
    service: HttpService,
    target_url: str,
    *,
    auth_key: str | None = None,
) -> StreamingResponse:

    query = request.url.query
    if query:
        target_url = f"{target_url}?{query}"

    body = await request.body()
    headers = _forward_request_headers(request, auth_key)

    # A transparent passthrough never replays: the caller owns retry semantics.
    try:
        response = await service.send(
            request.method,
            target_url,
            retry=False,
            stream=True,
            headers=headers,
            content=body,
        )

    except httpx2.RequestError as e:
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


def _forward_response_headers(response: httpx2.Response) -> dict:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() != "content-length"
    }
