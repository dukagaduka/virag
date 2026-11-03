"""QdrantStore: gRPC transport (retry policy shape, unavailable → 503) and the alias
resolved from Qdrant rather than cached."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import grpc
import pytest

from virag.clients.qdrant_store import QdrantStore
from virag.errors import ApiError

from conftest import make_store


class _RpcError(grpc.RpcError):

    def __init__(self, code: grpc.StatusCode):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "boom"


def _store() -> QdrantStore:
    return make_store()


def test_service_config_retries_points_and_safe_collection_calls():
    config = json.loads(QdrantStore._service_config(3, 0.25, 8))
    method = config["methodConfig"][0]

    names = {(n["service"], n.get("method")) for n in method["name"]}
    assert ("qdrant.Points", None) in names                             # every point op
    assert ("qdrant.Collections", "Get") in names
    assert ("qdrant.Collections", "Update") in names
    assert not {("qdrant.Collections", m) for m in ("Create", "Delete", "UpdateAliases")} & names

    policy = method["retryPolicy"]
    assert policy["maxAttempts"] == 4
    assert policy["initialBackoff"] == "0.250s" and policy["maxBackoff"] == "8.000s"
    assert policy["retryableStatusCodes"] == ["UNAVAILABLE"]


def test_service_config_without_retries_has_no_policy():
    assert json.loads(QdrantStore._service_config(0, 0.25, 8)) == {}


def test_zero_backoff_is_made_positive_for_grpc():
    policy = json.loads(QdrantStore._service_config(1, 0, 0))["methodConfig"][0]["retryPolicy"]

    assert policy["initialBackoff"] == "0.010s" and policy["maxBackoff"] == "0.010s"


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED])
def test_unreachable_qdrant_becomes_503(monkeypatch, code):
    store = _store()

    async def fail(**kwargs):
        raise _RpcError(code)

    monkeypatch.setattr(store.qdrant, "query_points", fail)

    with pytest.raises(ApiError) as exc:
        asyncio.run(store.query([0.0] * 4))

    assert exc.value.status_code == 503
    assert exc.value.details["grpc_status"] == code.name


def test_other_grpc_errors_pass_through(monkeypatch):
    store = _store()

    async def fail(**kwargs):
        raise _RpcError(grpc.StatusCode.INVALID_ARGUMENT)

    monkeypatch.setattr(store.qdrant, "delete", fail)

    with pytest.raises(grpc.RpcError):
        asyncio.run(store.delete_by_doc("doc"))


def _aliases(store, mapping: dict):
    """Stub get_aliases with the alias → collection `mapping` (mutable, read per call)."""
    async def get_aliases():
        return SimpleNamespace(aliases=[
            SimpleNamespace(alias_name=alias, collection_name=collection) for alias, collection in mapping.items()
        ])

    store.qdrant.get_aliases = get_aliases


def test_resolve_reads_the_alias_each_time():
    store = _store()
    mapping = {"c": "c__v1"}
    _aliases(store, mapping)

    async def get_collections():
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in ("c__v1", "c__v2", "other")])

    store.qdrant.get_collections = get_collections

    async def go():
        before = await store.resolve()
        mapping["c"] = "c__v2"                                          # another process swapped it
        return before, await store.resolve()

    assert asyncio.run(go()) == (("c__v1", None), ("c__v2", "c__v1"))


def test_first_start_race_adopts_the_collection_another_process_created():
    store = _store()
    mapping: dict = {}
    _aliases(store, mapping)

    async def collection_exists(name):
        return False

    async def create_collection(**kwargs):
        mapping["c"] = "c__v1"                                          # the other process won
        raise _RpcError(grpc.StatusCode.ALREADY_EXISTS)

    store.qdrant.collection_exists = collection_exists
    store.qdrant.create_collection = create_collection

    assert asyncio.run(store._resolve_or_create()) == "c__v1"
