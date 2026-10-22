"""Shared HTTP transport for every upstream service virag talks to.

One long-lived pooled client per upstream, retries with jittered backoff for
transient failures, and a single mapping of transport/status errors to ApiError.
Concrete clients never touch the HTTP library's exceptions directly.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Collection, Coroutine, Iterable
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Callable, TypeVar

import httpx2

from virag.errors import ApiError
from virag.utils import error_body


# Upstream replies that signal a transient overload/outage worth replaying.
RETRY_STATUSES = frozenset({429, 502, 503, 504})

# Transport failures where the request can be safely replayed. ReadTimeout and
# PoolTimeout are deliberately absent: a slow upstream or our own exhausted pool
# only gets worse under extra load.
RETRY_ERRORS = (
    httpx2.ConnectError,
    httpx2.ConnectTimeout,
    httpx2.RemoteProtocolError,
    httpx2.ReadError,
    httpx2.WriteError,
)

T = TypeVar("T")


class HttpService(ABC):
    """An upstream reached over one pooled, keep-alive HTTP client."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float,
        connect_timeout: float,
        max_connections: int,
        max_retries: int,
        backoff_base: float,
        backoff_cap: float,
        max_concurrency: int = 0,
        headers: dict | None = None,
    ):

        self.base_url     = base_url
        self.max_retries  = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap  = backoff_cap

        # 0 = unlimited; otherwise cap concurrent upstream calls made under `_guard`
        # so this process cannot bury the upstream under its own load.
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

        # `timeout` bounds read/write/pool waits (a read deadline is the max silence
        # between bytes, not the whole request); connecting is held to its own,
        # shorter deadline. Keep-alive is sized to the pool so connections are reused.
        self.http = httpx2.AsyncClient(
            timeout=httpx2.Timeout(timeout, connect=connect_timeout),
            limits=httpx2.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            headers=headers,
        )


    @abstractmethod
    async def check_connection(self) -> None:
        """Verify the upstream is reachable and usable; raise ApiError otherwise."""


    async def close(self) -> None:
        await self.http.aclose()


    def _guard(self):
        """Hold one concurrency slot when a ceiling is configured; a no-op otherwise.
        The slot is shared by every caller of this client, retries included."""
        return self._semaphore if self._semaphore is not None else nullcontext()


    async def send(
        self,
        method: str,
        url: str,
        *,
        retry: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> httpx2.Response:
        """Send one request, replaying it on a transient failure (RETRY_ERRORS or a
        RETRY_STATUSES reply) with jittered exponential backoff.

        The final response is returned whatever its status, so passthrough callers
        can relay it verbatim. A `stream=True` response must be closed by the caller.
        """
        attempts = self.max_retries + 1 if retry else 1
        attempt = 0

        while True:
            last = attempt >= attempts - 1

            try:
                # Rebuilt on every attempt: a multipart body cannot be replayed.
                request = self.http.build_request(method, url, **kwargs)
                response = await self.http.send(request, stream=stream)

            except RETRY_ERRORS:
                if last:
                    raise

                await asyncio.sleep(self._backoff(attempt))

                attempt += 1
                continue

            if last or response.status_code not in RETRY_STATUSES:
                return response

            delay = self._retry_after(response) or self._backoff(attempt)

            await response.aclose()
            await asyncio.sleep(delay)

            attempt += 1


    async def _request(
        self,
        method: str,
        url: str,
        *,
        action: str,
        retry: bool = True,
        passthrough_statuses: Collection[int] = (),
        **kwargs,
    ) -> httpx2.Response:
        """`send` that requires a 2xx: an unreachable upstream or an error status
        becomes an ApiError(503) naming the `action`. Statuses listed in
        `passthrough_statuses` are returned instead, for the caller to handle."""
        try:
            response = await self.send(method, url, retry=retry, **kwargs)

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {url} during {action}",
                details={"error": repr(e)},
            ) from e

        if not response.is_success and response.status_code not in passthrough_statuses:
            raise ApiError(
                status_code=503,
                message=f"Upstream {url} failed during {action}",
                details={"upstream_status": response.status_code, "upstream_error": error_body(response)},
            )

        return response


    @asynccontextmanager
    async def _stream(
        self,
        method: str,
        url: str,
        *,
        action: str,
        retry: bool = True,
        **kwargs,
    ):
        """Open a streaming request that must answer 2xx and yield the live response;
        it is always closed on exit. Errors while *reading* the body surface in the
        caller as the HTTP library's RequestError."""
        try:
            response = await self.send(method, url, retry=retry, stream=True, **kwargs)

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {url} during {action}",
                details={"error": repr(e)},
            ) from e

        try:
            if not response.is_success:
                await response.aread()

                raise ApiError(
                    status_code=503,
                    message=f"Upstream {url} failed during {action}",
                    details={"upstream_status": response.status_code, "upstream_error": error_body(response)},
                )

            yield response

        finally:
            await response.aclose()


    def _json(
        self,
        response: httpx2.Response,
        *,
        action: str,
        parse: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Decode a JSON body, optionally reshaping it with `parse`; a malformed body or
        an unexpected structure becomes an ApiError(422)."""
        try:
            data = response.json()

            return parse(data) if parse is not None else data

        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ApiError(
                status_code=422,
                message=f"Cannot decode {response.url} response or unexpected response structure during {action}",
                details={"base_message": str(e)},
            ) from e


    @staticmethod
    async def _gather(coros: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
        """Run coroutines concurrently and return their results in input order.

        Unlike asyncio.gather, the first failure cancels the rest — no orphaned
        requests keep loading the upstream — and is re-raised as-is rather than
        wrapped in an ExceptionGroup, so callers still see a plain ApiError.
        """
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(coro) for coro in coros]

        except BaseExceptionGroup as eg:
            raise eg.exceptions[0]

        return [task.result() for task in tasks]


    def _backoff(self, attempt: int) -> float:
        """Full-jitter exponential backoff: uniform over [0, min(cap, base * 2^attempt)]."""
        return random.uniform(0, min(self.backoff_cap, self.backoff_base * 2 ** attempt))


    def _retry_after(self, response: httpx2.Response) -> float | None:
        """A numeric `Retry-After` (seconds), capped at the backoff ceiling."""
        try:
            seconds = float(response.headers.get("retry-after", ""))

        except ValueError:
            return None

        return min(max(seconds, 0.0), self.backoff_cap)


class OpenAIService(HttpService):
    """An OpenAI-compatible upstream (vLLM or hosted): Bearer auth on every request
    and a `/models` check that the configured model is actually served."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        **http_options,
    ):

        super().__init__(
            base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            **http_options,
        )

        self.model_name = model_name
        self.api_key    = api_key


    @property
    def models_route(self) -> str:
        return f"{self.base_url}/models"


    async def check_connection(self) -> None:
        response = await self._request("GET", self.models_route, action="the check")
        models = self._json(response, action="the check", parse=self._model_ids)

        if self.model_name not in models:
            raise ApiError(
                status_code=404,
                message=f"Model {self.model_name} is not established at {self.base_url}",
                details={"models": models},
            )


    @staticmethod
    def _model_ids(data: dict) -> list[str]:
        return [model["id"] for model in data["data"]]
