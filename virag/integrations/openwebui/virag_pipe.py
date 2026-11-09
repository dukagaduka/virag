"""
title: virag pixel-RAG
author: virag
version: 0.2.0
requirements: httpx
description: >
  Routes chat through virag's OpenAI-compatible gateway with pixel-RAG enabled
  and renders the retrieved document tiles as native OpenWebUI citations. Exposes
  two selectable models: one with RAG (+sources), one plain passthrough.
"""

import asyncio
import contextlib
import json
from typing import AsyncGenerator

import httpx
from pydantic import BaseModel, Field


class Pipe:

    class Valves(BaseModel):
        VIRAG_BASE_URL: str = Field(
            default="http://virag:8080",
            description="Base URL of the virag gateway (no trailing slash).",
        )
        VIRAG_API_KEY: str = Field(
            default="",
            description="virag's VIRAG_API_KEY; empty only if virag runs with it empty.",
        )
        MODEL: str = Field(
            default="my-vlm",
            description="Upstream VLM model name virag should call.",
        )
        TOP_K: int = Field(
            default=0,
            description="Candidate tiles to take from Qdrant (0 = server default).",
        )
        RETRIEVER_THRESHOLD: float = Field(
            default=0.25,
            description="Minimum Qdrant score of a candidate (negative = server default).",
        )
        TOP_N: int = Field(
            default=0,
            description="Tiles the reranker passes to the model, at most TOP_K (0 = server default).",
        )
        RERANKER_THRESHOLD: float = Field(
            default=-1.0,
            description="Minimum reranker relevance, 0..1 (negative = server default).",
        )
        FRAMING_TEXT: str = Field(
            default="",
            description="Text prepended to injected tiles (blank = server default).",
        )
        SHOW_STATUS: bool = Field(
            default=True,
            description="Emit a 'retrieving' status line while RAG runs.",
        )
        HEARTBEAT_SEC: float = Field(
            default=10.0,
            description="While waiting for the model's first token, emit a keep-alive "
            "status this often (seconds) so a long prefill doesn't trip an idle "
            "disconnect. 0 disables.",
        )


    def __init__(self):
        self.valves = self.Valves()


    def pipes(self):
        # Two entries in the model selector, backed by one function.
        return [
            {"id": "rag", "name": "virag (pixel-RAG)"},
            {"id": "plain", "name": "virag (no RAG)"},
        ]


    async def pipe(self, body: dict, __event_emitter__=None) -> AsyncGenerator[str, None]:
        # body["model"] arrives as "<function_id>.rag" / "<function_id>.plain".
        variant = str(body.get("model", "")).split(".")[-1]
        use_rag = variant != "plain"

        payload = {
            "model": self.valves.MODEL,
            "messages": body["messages"],
            "stream": True,
        }

        if use_rag:
            rag = {"include_sources": True}

            # Unset valves leave the field to virag's own defaults.
            if self.valves.TOP_K > 0:
                rag["top_k"] = self.valves.TOP_K

            if self.valves.RETRIEVER_THRESHOLD >= 0:
                rag["retriever_threshold"] = self.valves.RETRIEVER_THRESHOLD

            if self.valves.TOP_N > 0:
                rag["top_n"] = self.valves.TOP_N

            if self.valves.RERANKER_THRESHOLD >= 0:
                rag["reranker_threshold"] = self.valves.RERANKER_THRESHOLD

            if self.valves.FRAMING_TEXT:
                rag["framing_text"] = self.valves.FRAMING_TEXT

            payload["rag"] = rag

        headers = {}
        if self.valves.VIRAG_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.VIRAG_API_KEY}"

        emit = __event_emitter__ if (use_rag and self.valves.SHOW_STATUS) else None

        if emit:
            await self._status(emit, "Retrieving document tiles…")

        url = f"{self.valves.VIRAG_BASE_URL.rstrip('/')}/v1/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "replace")
                        yield f"\n\n> **virag error {resp.status_code}**: {detail}"
                        return

                    async for piece in self._relay_with_heartbeat(resp, emit):
                        yield piece
        except httpx.RequestError as e:
            yield f"\n\n> **virag unreachable**: {e!r}"
        finally:
            if emit:
                await self._status(emit, "", done=True)


    async def _relay_with_heartbeat(self, resp, emit) -> AsyncGenerator[str, None]:
        """Yield the answer text, but read the SSE in a background task so we can emit
        keep-alive status events while the model is still doing its (possibly long)
        prefill. A fully silent stream can trip an idle disconnect in the OWUI/browser
        chain — the heartbeat keeps that channel warm until the first token arrives."""
        queue: asyncio.Queue = asyncio.Queue()

        async def reader():
            try:
                async for line in resp.aiter_lines():
                    await queue.put(("line", line))

            except httpx.RequestError as e:
                await queue.put(("error", e))

            finally:
                await queue.put(("end", None))

        reader_task = asyncio.create_task(reader())
        streaming = False
        beat = self.valves.HEARTBEAT_SEC

        try:
            while True:
                try:
                    kind, item = (
                        await asyncio.wait_for(queue.get(), timeout=beat)
                        if beat and beat > 0
                        else await queue.get()
                    )

                except asyncio.TimeoutError:
                    # No data yet — only heartbeat before the first token; once content
                    # is flowing it keeps the channel warm on its own.
                    if not streaming and emit:
                        await self._status(emit, "Waiting for the model…")
                    continue

                if kind == "end":
                    break

                if kind == "error":
                    yield "\n\n> _virag: upstream stream ended early — answer may be truncated._"
                    break

                line = item

                if not line or not line.startswith("data: "):
                    continue

                data = line[len("data: "):].strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # virag splices sources as a chunk with empty `choices`.
                sources = chunk.get("sources")
                if sources and emit:
                    await self._emit_sources(sources, emit)
                    continue

                for choice in chunk.get("choices", []):
                    piece = choice.get("delta", {}).get("content")
                    if piece:
                        streaming = True
                        yield piece
        finally:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader_task


    async def _status(self, emit, description: str, done: bool = False):
        await emit({
            "type": "status",
            "data": {"description": description, "done": done},
        })


    async def _emit_sources(self, sources, emit):
        for s in sources:
            doc_id = s.get("doc_id")
            tile = s.get("tile_num")
            score = s.get("score")
            name = s.get("name") or doc_id or "source"

            label = name if tile is None else f"{name} · tile {tile}"

            rerank_score = s.get("rerank_score")

            summary = f"doc_id={doc_id} · tile={tile}"

            if isinstance(score, (int, float)):
                summary += f" · score={score:.3f}"

            if isinstance(rerank_score, (int, float)):
                summary += f" · rerank_score={rerank_score:.3f}"

            # Newer OpenWebUI builds use "source"; older ones use "citation".
            # "citation" is still accepted by current builds, so we target it.
            await emit({
                "type": "citation",
                "data": {
                    "document": [summary],
                    "metadata": [{"source": label, "doc_id": doc_id}],
                    "source": {"name": label},
                },
            })
