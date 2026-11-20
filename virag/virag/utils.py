from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator
from itertools import islice
from typing import Any, TypeVar

import httpx2
import orjson
from python_multipart.multipart import MultipartParser, parse_options_header

from virag.errors import ApiError


log = logging.getLogger("virag.utils")

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


async def every(interval_sec: float, job: Callable[[], Awaitable[None]], *, name: str) -> None:
    """Run `job` every `interval_sec` until cancelled (0 = never). A run that fails is
    logged and the loop goes on."""
    if interval_sec <= 0:
        return

    while True:
        await asyncio.sleep(interval_sec)

        try:
            await job()

        except Exception as e:  # noqa: BLE001 - a periodic job must never end its loop
            log.error("%s failed: %r", name, e)


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


class MultipartFields:
    """The text fields of a multipart/form-data body, picked out as the body streams by:
    `feed` it every chunk, read `fields`. File parts are passed over — nothing of them
    is kept. A body that is not multipart, or is malformed, raises ValueError."""

    # A text field beyond this is not one worth reading.
    MAX_FIELD_BYTES = 1 << 20


    def __init__(self, content_type: str):

        kind, options = parse_options_header(content_type or "")

        if kind != b"multipart/form-data" or not options.get(b"boundary"):
            raise ValueError("the body is not multipart/form-data")

        # Complete text fields, by name, in the order they arrived.
        self.fields: dict[str, str] = {}

        # The part being read: its headers so far, and the name and data of a text field
        # (the name is None while passing over a file part).
        self._header = (bytearray(), bytearray())
        self._name: str | None = None
        self._data = bytearray()

        self._parser = MultipartParser(
            options[b"boundary"],
            {
                "on_part_begin": self._part_begin,
                "on_header_field": self._header_field,
                "on_header_value": self._header_value,
                "on_header_end": self._header_end,
                "on_part_data": self._part_data,
                "on_part_end": self._part_end,
            },
        )


    def feed(self, chunk: bytes) -> None:
        """Take the next chunk of the body."""
        self._parser.write(chunk)


    def _part_begin(self) -> None:
        self._name = None
        self._data = bytearray()


    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header[0].extend(data[start:end])


    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._header[1].extend(data[start:end])


    def _header_end(self) -> None:
        field, value = bytes(self._header[0]), bytes(self._header[1])
        self._header = (bytearray(), bytearray())

        if field.lower() != b"content-disposition":
            return

        _, options = parse_options_header(value)

        if b"filename" not in options and options.get(b"name"):
            self._name = options[b"name"].decode("utf-8", "replace")


    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._name is None:
            return

        self._data.extend(data[start:end])

        if len(self._data) > self.MAX_FIELD_BYTES:
            raise ValueError(f"form field {self._name!r} is larger than {self.MAX_FIELD_BYTES} bytes")


    def _part_end(self) -> None:
        if self._name is not None:
            self.fields[self._name] = self._data.decode("utf-8", "replace")
