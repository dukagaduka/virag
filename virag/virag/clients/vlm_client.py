from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, nullcontext

import httpx

from virag.errors import ApiError
from virag.settings import VLMSettings
from virag.utils import error_body


# A 1x1 PNG — the smallest valid image to probe whether the model accepts image inputs.
_PROBE_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# Fixed linear backoff between transient-connection retries (seconds * attempt).
_RETRY_BACKOFF_SEC = 0.5


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float | httpx.Timeout,
        max_concurrency: int = 0,
        max_retries: int = 3,
    ):

        self.base_url    = base_url
        self.model_name  = model_name
        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries

        # 0 (or less) = unlimited; otherwise cap concurrent upstream calls so a fragile
        # or remote VLM isn't buried under simultaneous multimodal requests.
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None


    @property
    def route(self) -> str:
        return f"{self.base_url}/chat/completions"


    @property
    def models_route(self) -> str:
        return f"{self.base_url}/models"


    @property
    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}


    def _guard(self):
        """Limit concurrent upstream calls when a ceiling is configured; a no-op
        otherwise. Held for the whole call, including the entire streaming lifetime."""
        return self._semaphore if self._semaphore is not None else nullcontext()


    @classmethod
    def from_settings(cls, vs: VLMSettings) -> VLMClient:
        return cls(
            base_url=vs.base_url,
            model_name=vs.model_name,
            api_key=vs.api_key,
            timeout=vs.timeout,
            max_concurrency=vs.max_concurrency,
            max_retries=vs.max_retries,
        )


    async def check_connection(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    self.models_route,
                    headers=self.auth_header,
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot establish connection to the {self.models_route} during the check",
                    details={"http_status_error": {"code": e.response.status_code}, "text": e.response.text},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.models_route} during the check",
                    details={"error": repr(e)},
                ) from e

            try:
                data = response.json()
                models = [model["id"] for model in data["data"]]

            except (KeyError, TypeError, ValueError) as e:
                raise ApiError(
                    status_code=422,
                    message=f"Cannot decode {self.models_route} response or unexpected response structure during the check",
                    details={"base_message": str(e)},
                ) from e

            if self.model_name not in models:
                raise ApiError(
                    status_code=404,
                    message=f"Model {self.model_name} is not established at {self.base_url}",
                    details={"models": models},
                )

            await self._probe_vision(client)


    async def _probe_vision(self, client: httpx.AsyncClient) -> None:
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
            response = await client.post(
                self.route,
                json=body,
                headers=self.auth_header,
            )

        except httpx.RequestError as e:
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


    async def complete(self, body: dict) -> httpx.Response:
        """Forward a non-streaming chat-completion request and return the upstream
        response verbatim (any status) so the route can passthrough. Transient
        connection errors are retried up to `max_retries`; only a persistently
        unreachable upstream is turned into an ApiError."""
        async with self._guard():
            last_exc: httpx.RequestError | None = None

            for attempt in range(self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        return await client.post(
                            self.route,
                            json=body,
                            headers=self.auth_header,
                        )

                except httpx.RequestError as e:
                    last_exc = e

                    if attempt < self.max_retries:
                        await asyncio.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))

            raise ApiError(
                status_code=502,
                message=f"Cannot reach {self.route} during chat completion",
                details={"error": repr(last_exc), "attempts": self.max_retries + 1},
            ) from last_exc


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
            last_exc: httpx.RequestError | None = None

            for attempt in range(self.max_retries + 1):
                yielded = False

                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        async with client.stream(
                            "POST",
                            self.route,
                            json=body,
                            headers=self.auth_header,
                        ) as response:
                            yielded = True
                            yield response
                            return

                except httpx.RequestError as e:
                    # Reached only on a pre-yield open failure. If we already handed the
                    # response to the caller, never retry — re-raise so teardown is honest.
                    if yielded:
                        raise

                    last_exc = e

                    if attempt < self.max_retries:
                        await asyncio.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))

            raise ApiError(
                status_code=502,
                message=f"Cannot reach {self.route} during streaming chat completion",
                details={"error": repr(last_exc), "attempts": self.max_retries + 1},
            ) from last_exc
