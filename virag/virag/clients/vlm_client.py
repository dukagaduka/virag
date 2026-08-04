from __future__ import annotations

from contextlib import asynccontextmanager

import httpx

from virag.errors import ApiError
from virag.settings import VLMSettings
from virag.utils import error_body


# A 1x1 PNG — the smallest valid image to probe whether the model accepts image inputs.
_PROBE_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class VLMClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float | httpx.Timeout,
    ):

        self.base_url   = base_url
        self.model_name = model_name
        self.api_key    = api_key
        self.timeout    = timeout


    @property
    def route(self) -> str:
        return f"{self.base_url}/chat/completions"


    @property
    def models_route(self) -> str:
        return f"{self.base_url}/models"


    @property
    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}


    @classmethod
    def from_settings(cls, vs: VLMSettings) -> VLMClient:
        return cls(
            base_url=vs.base_url,
            model_name=vs.model_name,
            api_key=vs.api_key,
            timeout=vs.timeout,
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
        response verbatim (any status) so the route can passthrough. Only an
        unreachable upstream is turned into an ApiError."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                return await client.post(
                    self.route,
                    json=body,
                    headers=self.auth_header,
                )

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=502,
                    message=f"Cannot reach {self.route} during chat completion",
                    details={"error": repr(e)},
                ) from e


    @asynccontextmanager
    async def stream(self, body: dict):
        """Open a streaming chat-completion request and yield the live upstream
        response. The caller inspects `status_code` before relaying, then iterates
        `aiter_bytes()`. Only an unreachable upstream is turned into an ApiError."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self.route,
                    json=body,
                    headers=self.auth_header,
                ) as response:
                    yield response

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=502,
                    message=f"Cannot reach {self.route} during streaming chat completion",
                    details={"error": repr(e)},
                ) from e
