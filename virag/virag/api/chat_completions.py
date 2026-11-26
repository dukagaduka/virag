"""POST /v1/chat/completions: the request goes to the VLM as it came, except for its
`rag` field — with it, the Rag puts the tiles the question is answered from into the last
user message, and the answer may carry their `sources`."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

import httpx2
import orjson
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from virag.errors import unprocessable
from virag.responses import OrjsonResponse


log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat completions; enables pixel-RAG when a `rag` field is present",
    operation_id="chatCompletions",
)
async def chat_completions(request: Request):

    body = await _read_body(request)
    rag = request.app.state.rag

    rag_config = rag.config(body.pop("rag", None))

    sources = None
    if rag_config is not None:
        sources = await rag.augment(body.get("messages", []), rag_config)

    vlm = request.app.state.vlm

    if body.get("stream"):
        return await _stream_response(vlm, body, sources)

    return await _complete_response(vlm, body, sources)


# ---------------------------------------------------------------------------
# Upstream passthrough
# ---------------------------------------------------------------------------

async def _complete_response(vlm, body: dict, sources: list[dict] | None) -> Response:
    response = await vlm.complete(body)
    media_type = response.headers.get("content-type", "application/json")

    if response.status_code == 200 and sources is not None:
        try:
            data = orjson.loads(response.content)
            data["sources"] = sources

            return OrjsonResponse(status_code=200, content=data)

        except ValueError:
            pass

    return Response(content=response.content, status_code=response.status_code, media_type=media_type)


async def _stream_response(vlm, body: dict, sources: list[dict] | None) -> Response:
    stack = AsyncExitStack()
    response = await stack.enter_async_context(vlm.stream(body))

    media_type = response.headers.get("content-type", "text/event-stream")

    if response.status_code >= 400:
        content = await response.aread()
        await stack.aclose()

        return Response(content=content, status_code=response.status_code, media_type=media_type)

    async def generate():
        try:
            async for chunk in _relay(response, sources):
                yield chunk

        except httpx2.RequestError as e:
            # The upstream VLM stalled past its read deadline or dropped the stream
            # mid-response. Whatever streamed so far has already reached the client;
            # log and end the SSE cleanly rather than letting the error surface as an
            # unhandled ASGI exception (which hard-closes the connection and cascades
            # into a client-side abort of the request).
            log.warning("upstream vlm stream interrupted mid-relay: %r", e)

        finally:
            await stack.aclose()

    return StreamingResponse(generate(), status_code=200, media_type=media_type)


async def _relay(response, sources: list[dict] | None):
    """Relay upstream SSE chunks; when sources are requested, splice a final chunk
    carrying them just before the `[DONE]` sentinel (as the `usage` chunk is)."""
    if sources is None:
        async for chunk in response.aiter_bytes():
            yield chunk

        return

    sources_event = (
        b"data: "
        + orjson.dumps({"object": "chat.completion.chunk", "choices": [], "sources": sources})
        + b"\n\n"
    )

    done_marker = b"data: [DONE]"
    spliced = False

    async for chunk in response.aiter_bytes():
        if not spliced and done_marker in chunk:
            head, _, tail = chunk.partition(done_marker)

            if head:
                yield head

            yield sources_event
            yield done_marker + tail
            spliced = True

        else:
            yield chunk

    if not spliced:
        yield sources_event


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

async def _read_body(request: Request) -> dict:
    raw = await request.body()

    try:
        body = orjson.loads(raw)

    except ValueError as e:
        raise unprocessable("Request body must be valid JSON", base_message=str(e)) from e

    if not isinstance(body, dict):
        raise unprocessable("Request body must be a JSON object")

    return body
