from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext

import httpx2

from virag.clients.http_service import OpenAIService
from virag.errors import ApiError
from virag.settings import HttpSettings, VLMSettings
from virag.utils import error_body


# A 1x1 PNG — the smallest valid image to probe whether the model accepts image inputs.
_PROBE_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class VLMClient(OpenAIService):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        *,
        max_concurrency: int = 0,
        **http_options,
    ):

        super().__init__(base_url, model_name, api_key, **http_options)

        # 0 (or less) = unlimited; otherwise cap concurrent upstream calls so a fragile
        # or remote VLM isn't buried under simultaneous multimodal requests.
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None


    @property
    def route(self) -> str:
        return f"{self.base_url}/chat/completions"


    def _guard(self):
        """Limit concurrent upstream calls when a ceiling is configured; a no-op
        otherwise. Held for the whole call, including the entire streaming lifetime."""
        return self._semaphore if self._semaphore is not None else nullcontext()


    @classmethod
    def from_settings(cls, vs: VLMSettings, hs: HttpSettings) -> VLMClient:
        return cls(
            base_url=vs.base_url,
            model_name=vs.model_name,
            api_key=vs.api_key,
            max_concurrency=vs.max_concurrency,
            timeout=vs.timeout,
            connect_timeout=vs.connect_timeout,
            max_connections=vs.max_connections,
            max_retries=vs.max_retries,
            backoff_base=hs.backoff_base,
            backoff_cap=hs.backoff_cap,
        )


    async def check_connection(self) -> None:
        await super().check_connection()
        await self._probe_vision()


    async def _probe_vision(self) -> None:
        """Send a 1x1 image and require a non-error reply — the only portable way to
        tell whether an OpenAI-compatible model accepts image inputs at all."""
        body = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with the single word: ok"},
                        {"type": "image_url", "image_url": {"url": _PROBE_IMAGE}},
                    ],
                }
            ],
            "max_tokens": 1,
            "temperature": 0,
        }

        try:
            response = await self.send("POST", self.route, json=body)

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {self.route} during the vision probe",
                details={"error": repr(e)},
            ) from e

        if response.status_code >= 400:
            raise ApiError(
                status_code=503,
                message=f"Model {self.model_name} does not accept image inputs (not vision-capable)",
                details={"upstream_status": response.status_code, "upstream_error": error_body(response)},
            )


    async def complete(self, body: dict) -> httpx2.Response:
        """Forward a non-streaming chat-completion request and return the upstream
        response verbatim (any status) so the route can passthrough. Transient failures
        (connection errors, 429/502/503/504) are retried up to `max_retries`; only a
        persistently unreachable upstream is turned into an ApiError."""
        async with self._guard():
            try:
                return await self.send("POST", self.route, json=body)

            except httpx2.RequestError as e:
                raise ApiError(
                    status_code=502,
                    message=f"Cannot reach {self.route} during chat completion",
                    details={"error": repr(e), "attempts": self.max_retries + 1},
                ) from e


    @asynccontextmanager
    async def stream(self, body: dict):
        """Open a streaming chat-completion request and yield the live upstream
        response. The caller inspects `status_code` before relaying, then iterates
        `aiter_bytes()`. The upstream slot (semaphore) is held for the entire stream.

        A transient failure while *opening* the stream — before any byte reaches the
        caller — is retried up to `max_retries`. Once bytes are flowing, mid-stream
        errors surface in the caller (not here), so a retry can never replay a
        partially-consumed response."""
        async with self._guard():
            try:
                response = await self.send("POST", self.route, stream=True, json=body)

            except httpx2.RequestError as e:
                raise ApiError(
                    status_code=502,
                    message=f"Cannot reach {self.route} during streaming chat completion",
                    details={"error": repr(e), "attempts": self.max_retries + 1},
                ) from e

            try:
                yield response

            finally:
                await response.aclose()
