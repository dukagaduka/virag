from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from contextlib import AsyncExitStack

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from virag.errors import ApiError


log = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    summary="OpenAI-compatible chat completions; enables pixel-RAG when a `rag` field is present",
    operation_id="chatCompletions",
)
async def chat_completions(request: Request):

    body = await _read_body(request)
    settings = request.app.state.settings

    rag_config = _rag_config(body.pop("rag", None), settings.rag_settings)

    sources = None
    if rag_config is not None:
        sources = await _run_rag(request, body, rag_config)

    vlm = request.app.state.vlm

    if body.get("stream"):
        return await _stream_response(vlm, body, sources)

    return await _complete_response(vlm, body, sources)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

async def _run_rag(request: Request, body: dict, config: dict) -> list[dict] | None:
    """Retrieve tiles for the query, inject them into the last user message, and
    return the sources list when the caller asked for it (else None)."""
    embeddings = request.app.state.embeddings
    store = request.app.state.store
    tiler = request.app.state.tiler

    messages = body.get("messages", [])
    query = _extract_query(messages)

    if not query:
        raise ApiError(
            status_code=422,
            message="RAG is enabled but no user text was found to build the query",
        )

    log.info("rag query: top_n=%d threshold=%s preview=%r", config["top_n"], config["score_threshold"], query[:120])

    vector = (await embeddings.embed_texts([query]))[0]
    hits = await store.query(vector, top_n=config["top_n"])

    if config["score_threshold"] is not None:
        hits = [hit for hit in hits if hit.score >= config["score_threshold"]]

    log.info("rag retrieved: %d hits", len(hits))

    tiles = await _fetch_tiles(tiler, hits)

    if tiles:
        _inject_tiles(messages, [b64 for _, b64 in tiles], config["framing_text"])
        log.info("rag injected: %d tiles", len(tiles))
    else:
        log.info("rag injected: 0 tiles")

    if config["include_sources"]:
        return [_source(hit) for hit, _ in tiles]

    return None


async def _fetch_tiles(tiler, hits) -> list[tuple]:
    """Regenerate each hit's tile as base64 JPEG, skipping any that fail to render."""
    async def fetch(hit):
        doc_id = hit.payload.get("doc_id")
        tile_num = hit.payload.get("tile_num")

        try:
            data = await tiler.get_tile(doc_id, tile_num)

        except ApiError as e:
            log.warning("skipping tile (doc_id=%s tile_num=%s): %s", doc_id, tile_num, e.message)

            return None

        # Integrity: the regenerated tile must be the exact bytes enrolled in Qdrant.
        expected_hash = hit.payload.get("image_hash")
        if expected_hash is not None and hashlib.sha256(data).hexdigest() != expected_hash:
            log.warning(
                "integrity: tile (doc_id=%s tile_num=%s) hash mismatch (qdrant=%s); skipping",
                doc_id, tile_num, expected_hash,
            )

            return None

        return hit, base64.b64encode(data).decode()

    results = await asyncio.gather(*(fetch(hit) for hit in hits))

    return [result for result in results if result is not None]


def _extract_query(messages: list) -> str:
    """Text of the last user message (concatenated text parts of multimodal content)."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            texts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]

            return " ".join(text for text in texts if text).strip()

        return ""

    return ""


def _inject_tiles(messages: list, images_b64: list[str], framing_text: str) -> None:
    """Prepend framing text + retrieved tile images to the last user message."""
    index = _last_user_index(messages)

    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in images_b64
    ]

    original = messages[index].get("content")
    new_content = [{"type": "text", "text": framing_text}, *image_parts]

    if isinstance(original, str):
        new_content.append({"type": "text", "text": original})
    elif isinstance(original, list):
        new_content.extend(original)

    messages[index] = {**messages[index], "content": new_content}


def _last_user_index(messages: list) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index

    return len(messages) - 1


def _source(hit) -> dict:
    payload = hit.payload or {}
    metadata = payload.get("metadata") or {}

    return {
        "name": metadata.get("name"),
        "doc_id": payload.get("doc_id"),
        "tile_num": payload.get("tile_num"),
        "score": hit.score,
    }


# ---------------------------------------------------------------------------
# Upstream passthrough
# ---------------------------------------------------------------------------

async def _complete_response(vlm, body: dict, sources: list[dict] | None) -> Response:
    response = await vlm.complete(body)
    media_type = response.headers.get("content-type", "application/json")

    if response.status_code == 200 and sources is not None:
        try:
            data = response.json()
            data["sources"] = sources

            return JSONResponse(status_code=200, content=data)

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

        except httpx.RequestError as e:
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
        "data: "
        + json.dumps({"object": "chat.completion.chunk", "choices": [], "sources": sources})
        + "\n\n"
    ).encode()

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
        body = json.loads(raw)

    except ValueError as e:
        raise ApiError(
            status_code=422,
            message="Request body must be valid JSON",
            details={"base_message": str(e)},
        ) from e

    if not isinstance(body, dict):
        raise ApiError(
            status_code=422,
            message="Request body must be a JSON object",
        )

    return body


def _rag_config(raw, defaults) -> dict | None:
    """Normalize the `rag` trigger into a config dict, or None when RAG is off."""
    if raw is None or raw is False:
        return None

    if raw is True:
        return {
            "top_n": defaults.top_n,
            "score_threshold": defaults.score_threshold,
            "framing_text": defaults.framing_text,
            "include_sources": False,
        }

    if not isinstance(raw, dict):
        raise ApiError(
            status_code=422,
            message="`rag` must be an object or a boolean",
        )

    top_n = raw.get("top_n", defaults.top_n)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ApiError(
            status_code=422,
            message="`rag.top_n` must be a positive integer",
        )

    threshold = raw.get("score_threshold", defaults.score_threshold)
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))):
        raise ApiError(
            status_code=422,
            message="`rag.score_threshold` must be a number or null",
        )

    framing_text = raw.get("framing_text", defaults.framing_text)
    if not isinstance(framing_text, str):
        raise ApiError(
            status_code=422,
            message="`rag.framing_text` must be a string",
        )

    return {
        "top_n": top_n,
        "score_threshold": threshold,
        "framing_text": framing_text,
        "include_sources": bool(raw.get("include_sources", False)),
    }
