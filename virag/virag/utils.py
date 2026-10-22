from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable
from typing import Any, TypeVar

import httpx2


T = TypeVar("T")


def error_body(response: httpx2.Response):
    """Return an httpx response body as parsed JSON if possible, else the raw text."""
    try:
        return response.json()

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
