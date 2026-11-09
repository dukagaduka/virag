from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx2

from virag.clients.http_service import OpenAIService
from virag.errors import ApiError
from virag.settings import HttpSettings, VLMSettings
from virag.utils import error_body


log = logging.getLogger(__name__)


class VLMClient(OpenAIService):
    """Upstream vision chat model. `max_concurrency` (via `_guard`) shields a fragile
    or remote VLM from simultaneous multimodal requests; a slot is held for the whole
    call, including the entire streaming lifetime."""

    @property
    def route(self) -> str:
        return f"{self.base_url}/chat/completions"


    @classmethod
    def from_settings(cls, vs: VLMSettings, hs: HttpSettings) -> VLMClient:
        return cls(
            base_url=vs.base_url,
            model_name=vs.model_name,
            api_key=vs.api_key,
            max_concurrency=vs.max_concurrency,
            timeout=vs.timeout_sec,
            connect_timeout=vs.connect_timeout_sec,
            max_connections=vs.max_connections,
            max_retries=vs.max_retries,
            backoff_base=hs.backoff_base_sec,
            backoff_cap=hs.backoff_cap_sec,
        )


    async def check_connection(self) -> None:
        await super().check_connection()
        await self._probe_vision()


    async def model_card(self) -> dict:
        """The upstream's `/models` entry of `model_name` — the model virag is set up
        for. A minimal card stands in when the upstream lists no such entry or does not
        answer."""
        try:
            response = await self._request("GET", self.models_route, action="the model list")
            models = self._json(response, action="the model list", parse=lambda data: data["data"])

        except ApiError as e:
            log.warning("model list: upstream failed, answering with %s alone: %s", self.model_name, e.message)
            models = []

        for model in models:
            if isinstance(model, dict) and model.get("id") == self.model_name:
                return model

        return {"id": self.model_name, "object": "model", "created": 0, "owned_by": "virag"}


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
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.PROBE_IMAGE}"}},
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
