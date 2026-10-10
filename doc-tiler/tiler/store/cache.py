"""Async LRU cache with in-flight coalescing.

Two consumers: final JPEG bytes keyed by `(doc_id, tile_num)`, and the full-page
PNG keyed by `doc_id` so sequential strip requests reuse one render.
Concurrent misses on the same key render exactly once.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Hashable


class AsyncLRU:
    """Bounded async LRU that renders each missing key exactly once."""

    def __init__(self, capacity: int, name: str = "cache"):
        self.capacity = max(1, capacity)
        self.name = name
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._inflight: dict[Hashable, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        self.coalesced = 0


    def peek(self, key: Hashable) -> Any | None:
        """Return the cached value without affecting LRU order, or None."""
        return self._data.get(key)


    async def get_or_create(
        self, key: Hashable, factory: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Return the cached value, or run `factory` once to produce it.

        Concurrent callers for the same missing key wait on the first
        producer instead of each running `factory`.
        """
        while True:
            async with self._lock:
                if key in self._data:
                    self._data.move_to_end(key)
                    self.hits += 1
                    return self._data[key]
                
                pending = self._inflight.get(key)
                if pending is None:
                    self.misses += 1
                    fut: asyncio.Future = asyncio.get_running_loop().create_future()
                    self._inflight[key] = fut
                    break

                self.coalesced += 1

            # Someone else is already producing this key: wait for their result.
            try:
                return await asyncio.shield(pending)
            except Exception:
                # The producer failed; loop and try to become the producer ourselves.
                continue

        try:
            value = await factory()

        except BaseException as exc:
            async with self._lock:
                self._inflight.pop(key, None)
            if not fut.done():
                fut.set_exception(exc)
                
            # Nobody may be awaiting the future; swallow the "never retrieved" warning.
            fut.exception()

            raise

        async with self._lock:
            self._inflight.pop(key, None)
            self._data[key] = value
            self._data.move_to_end(key)

            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

        if not fut.done():
            fut.set_result(value)

        return value


    async def invalidate(self, predicate: Callable[[Hashable], bool]) -> None:
        """Drop every cached key for which `predicate` returns true."""
        async with self._lock:
            for key in [k for k in self._data if predicate(k)]:
                self._data.pop(key, None)


    async def clear(self) -> None:
        """Drop all cached values."""
        async with self._lock:
            self._data.clear()


    def stats(self) -> dict[str, int]:
        """Return current size and hit/miss/coalesce counters."""
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "coalesced": self.coalesced,
        }
