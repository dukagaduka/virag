from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine, Iterable, Iterator
from itertools import islice
from typing import Any, TypeVar

import httpx2
import orjson

from virag.errors import ApiError


T = TypeVar("T")


def error_body(response: httpx2.Response):
    """Return an httpx response body as parsed JSON if possible, else the raw text."""
    try:
        return orjson.loads(response.content)

    except ValueError:
        return response.text


async def gather(coros: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Run coroutines concurrently and return their results in input order.

    Unlike asyncio.gather, the first failure cancels the rest — no orphaned requests
    keep loading an upstream — and is re-raised as-is rather than wrapped in an
    ExceptionGroup, so callers still see a plain ApiError.
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coro) for coro in coros]

    except BaseExceptionGroup as eg:
        raise eg.exceptions[0]

    return [task.result() for task in tasks]


async def cancel(task: asyncio.Task) -> None:
    """Cancel a background task and wait for it to unwind."""
    task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        pass


def chunked(items: Iterator[T], size: int) -> Iterator[list[T]]:
    """Lazily cut an iterator into lists of `size` (the last may be shorter)."""
    while chunk := list(islice(items, size)):
        yield chunk


def unavailable_as_503(
    service: str,
    *errors: type[Exception],
    when: Callable[[Exception], bool] | None = None,
    details: Callable[[Exception], dict] | None = None,
):
    """Decorator for async calls to a backend: `errors` (only those `when` accepts) mean
    the `service` is unreachable right now and are raised as ApiError(503), with
    `details(e)` or the error's repr."""
    def decorator(method):
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            try:
                return await method(*args, **kwargs)

            except errors as e:
                if when is not None and not when(e):
                    raise

                raise ApiError(
                    status_code=503,
                    message=f"{service} is unavailable during {method.__name__}",
                    details=details(e) if details is not None else {"error": repr(e)},
                ) from e

        return wrapper

    return decorator
