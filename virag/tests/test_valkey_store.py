"""ValkeyStore against a real Valkey (VALKEY_TEST_URL, e.g. redis://localhost:6379/15):
the conditional commands the Coordinator's rules rest on, and the 503 when unreachable.
Skipped without it."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from virag.clients.valkey_store import ValkeyStore
from virag.errors import ApiError


URL = os.environ.get("VALKEY_TEST_URL")

needs_valkey = pytest.mark.skipif(not URL, reason="VALKEY_TEST_URL is not set")


def _run(check):
    """Run `check(store)` against a fresh key prefix, cleaned up afterwards."""
    async def go():
        store = ValkeyStore(URL, f"virag-test-{uuid.uuid4().hex}", timeout=2, max_connections=4)

        try:
            await store.check_connection()
            return await check(store)

        finally:
            keys = [key async for key in store.redis.scan_iter(match=f"{store.prefix}:*")]
            if keys:
                await store.redis.delete(*keys)
            await store.close()

    return asyncio.run(go())


@needs_valkey
def test_set_nx_and_ifeq_are_compare_and_set():
    async def check(store):
        return (
            await store.set_nx("k", "a"),
            await store.set_nx("k", "b"),                               # taken
            await store.set_ifeq("k", "c", "b"),                        # not holding b
            await store.set_ifeq("k", "c", "a"),
            await store.get("k"),
        )

    assert _run(check) == (True, False, False, True, "c")


@needs_valkey
def test_delete_ifeq_spares_another_holder():
    async def check(store):
        await store.set_nx("k", "mine")
        spared = await store.delete_ifeq("k", "theirs")
        deleted = await store.delete_ifeq("k", "mine")

        return spared, deleted, await store.exists("k")

    assert _run(check) == (False, True, False)


@needs_valkey
def test_an_expired_lease_is_never_renewed_back():
    async def check(store):
        await store.set_ttl("lease", {"pid": 1}, 0.05, nx=True)
        renewed = await store.set_ttl("lease", {"pid": 1}, 30, xx=True)
        await asyncio.sleep(0.1)                                        # 0.05 s TTL, renewed to 30 s
        still = await store.exists("lease")

        await store.set_ttl("gone", {"pid": 2}, 0.05, nx=True)
        await asyncio.sleep(0.1)
        revived = await store.set_ttl("gone", {"pid": 2}, 30, xx=True)

        return renewed, still, revived, await store.exists("gone")

    assert _run(check) == (True, True, False, False)


@needs_valkey
def test_json_hashes_and_sets_round_trip():
    async def check(store):
        await store.set_json("status", {"state": "running", "n": 1})
        await store.hset_json("writers", {"w1": {"doc_id": "a"}, "w2": {"doc_id": "b"}})
        await store.hdel("writers", "w1")
        await store.sadd("journal", "a", "b", "a")

        return (
            await store.get_json("status"),
            await store.hgetall_json("writers"),
            await store.smembers("journal"),
            await store.exists_each(["status", "missing"]),
        )

    assert _run(check) == (
        {"state": "running", "n": 1},
        {"w2": {"doc_id": "b"}},
        {"a", "b"},
        [True, False],
    )


def test_unreachable_valkey_is_503():
    async def go():
        store = ValkeyStore("redis://127.0.0.1:1/0", "virag-test", timeout=0.2, max_connections=1)

        try:
            await store.get("k")

        finally:
            await store.close()

    with pytest.raises(ApiError) as exc:
        asyncio.run(go())

    assert exc.value.status_code == 503 and exc.value.code == "NOT_READY"
