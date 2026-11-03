from __future__ import annotations

import logging
from typing import Any, Iterable, TypeAlias

import orjson
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from virag.errors import ApiError
from virag.settings import ValkeySettings
from virag.utils import unavailable_as_503


log = logging.getLogger(__name__)


JsonValue: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None


class ValkeyStore:
    """Async API to Valkey: plain and JSON values, conditional writes, hashes and sets,
    every key under one prefix. It knows nothing of what the keys mean — that is the
    Coordinator's."""

    # SET … IFEQ arrived in Valkey 8.1, DELIFEQ in 9.0: compare-and-set / -delete
    # without Lua.
    MIN_VERSION = (9, 0)

    # Defined first: it decorates methods below while the class body is still executing
    _unavailable_as_503 = staticmethod(unavailable_as_503("Valkey", RedisConnectionError, RedisTimeoutError))


    def __init__(
        self,
        url: str,
        prefix: str,
        *,
        timeout: float,
        max_connections: int,
    ) -> None:

        self.prefix = prefix

        self.pool = redis.ConnectionPool.from_url(
            url,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            health_check_interval=30,
            max_connections=max_connections,
            decode_responses=True,
        )
        self.redis = redis.Redis(connection_pool=self.pool)


    @classmethod
    def from_settings(cls, vs: ValkeySettings) -> ValkeyStore:
        return cls(
            url=vs.url,
            prefix=vs.prefix,
            timeout=vs.timeout_sec,
            max_connections=vs.max_connections,
        )


    def _key(self, name: str) -> str:
        """The fully qualified key of `name`."""
        return f"{self.prefix}:{name}"


    @_unavailable_as_503
    async def check_connection(self) -> None:
        """Reachable, and new enough for the conditional commands."""
        await self.redis.ping()

        version = (await self.redis.info("server")).get("valkey_version")
        parsed = tuple(int(part) for part in str(version).split(".")[:2]) if version else None

        if parsed is None or parsed < self.MIN_VERSION:
            raise ApiError(
                status_code=503,
                message=f"Valkey {version or '(not Valkey)'} is too old: virag needs "
                        f"{'.'.join(map(str, self.MIN_VERSION))}+ (SET IFEQ, DELIFEQ)",
            )

        log.info("valkey: version=%s prefix=%s", version, self.prefix)


    # ------------------------------------------------------------------ #
    # values
    # ------------------------------------------------------------------ #
    @_unavailable_as_503
    async def get(self, name: str) -> str | None:
        return await self.redis.get(self._key(name))


    @_unavailable_as_503
    async def get_json(self, name: str) -> JsonValue:
        raw = await self.redis.get(self._key(name))

        return None if raw is None else orjson.loads(raw)


    @_unavailable_as_503
    async def set_json(self, name: str, value: JsonValue) -> None:
        await self.redis.set(self._key(name), orjson.dumps(value))


    @_unavailable_as_503
    async def set_nx(self, name: str, value: str) -> bool:
        """Set `name` only if it does not exist yet. True if it was set."""
        return bool(await self.redis.set(self._key(name), value, nx=True))


    @_unavailable_as_503
    async def set_ifeq(self, name: str, value: str, expected: str) -> bool:
        """Set `name` only if it holds `expected` now. True if it was set."""
        # Sent as is: redis-py flags its own `ifeq` argument experimental (Redis' variant).
        return bool(await self.redis.execute_command("SET", self._key(name), value, "IFEQ", expected))


    @_unavailable_as_503
    async def set_ttl(
        self,
        name: str,
        value: JsonValue,
        ttl_sec: float,
        *,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set `name` to expire in `ttl_sec`; `nx` only if it does not exist, `xx` only if
        it still does. True if it was set."""
        return bool(
            await self.redis.set(self._key(name), orjson.dumps(value), px=int(ttl_sec * 1000), nx=nx, xx=xx)
        )


    @_unavailable_as_503
    async def exists(self, name: str) -> bool:
        return bool(await self.redis.exists(self._key(name)))


    @_unavailable_as_503
    async def exists_each(self, names: Iterable[str]) -> list[bool]:
        """Whether each of `names` exists, in one round trip."""
        names = list(names)
        if not names:
            return []

        pipe = self.redis.pipeline(transaction=False)
        for name in names:
            pipe.exists(self._key(name))

        return [bool(found) for found in await pipe.execute()]


    @_unavailable_as_503
    async def delete(self, *names: str) -> None:
        if names:
            await self.redis.delete(*(self._key(name) for name in names))


    @_unavailable_as_503
    async def delete_ifeq(self, name: str, expected: str) -> bool:
        """Delete `name` only if it holds `expected` now. True if it was deleted."""
        return bool(await self.redis.execute_command("DELIFEQ", self._key(name), expected))


    # ------------------------------------------------------------------ #
    # hashes (JSON fields)
    # ------------------------------------------------------------------ #
    @_unavailable_as_503
    async def hset_json(self, name: str, fields: dict[str, JsonValue]) -> None:
        if fields:
            await self.redis.hset(
                self._key(name),
                mapping={field: orjson.dumps(value) for field, value in fields.items()},
            )


    @_unavailable_as_503
    async def hgetall_json(self, name: str) -> dict[str, JsonValue]:
        raw = await self.redis.hgetall(self._key(name))

        return {field: orjson.loads(value) for field, value in raw.items()}


    @_unavailable_as_503
    async def hdel(self, name: str, *fields: str) -> None:
        if fields:
            await self.redis.hdel(self._key(name), *fields)


    # ------------------------------------------------------------------ #
    # sets
    # ------------------------------------------------------------------ #
    @_unavailable_as_503
    async def sadd(self, name: str, *members: str) -> None:
        if members:
            await self.redis.sadd(self._key(name), *members)


    @_unavailable_as_503
    async def smembers(self, name: str) -> set[str]:
        return set(await self.redis.smembers(self._key(name)))


    async def close(self) -> None:
        await self.redis.aclose()
