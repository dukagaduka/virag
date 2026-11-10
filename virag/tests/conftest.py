"""Shared fixtures: in-memory fakes for the backends and a lifespan-free app.

The real `lifespan` builds network clients and runs dependency + binding checks, so
the tests build the app directly and populate `app.state` with fakes instead. Each
handler only ever reaches its backends through `request.app.state`, so this exercises
the real routing, RAG pipeline and error handling with no network. The Coordinator is
the real one, over an in-memory Valkey."""

from __future__ import annotations

import hashlib
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client.models import Distance

from virag.clients.reranker_client import RerankerClient
from virag.clients.tiler_client import TilerService
from virag.coordination import Coordinator
from virag.errors import ApiError, install_error_handlers
from virag.main import include_routers
from virag.settings import (
    RetrieverSettings,
    EnrollSettings,
    HttpSettings,
    QdrantStoreSettings,
    RagSettings,
    ReconcileSettings,
    RerankerSettings,
    Settings,
    TilerSettings,
    ValkeySettings,
    VLMSettings,
)


def tile_bytes(doc_id: str, tile_num: int) -> bytes:
    """Deterministic JPEG-stand-in bytes for a tile, so hashes are reproducible."""
    return f"tile:{doc_id}:{tile_num}".encode()


def tile_hash(doc_id: str, tile_num: int) -> str:
    return hashlib.sha256(tile_bytes(doc_id, tile_num)).hexdigest()


@dataclass
class FakeHit:
    score: float
    payload: dict


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------

class FakeRetriever:

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[str]] = []

    async def embed_texts(self, texts, *, bulk=False):
        self.text_calls.append(list(texts))
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]

    async def embed_images(self, images, *, bulk=False):
        self.image_calls.append(list(images))
        return [[0.5] * self.dim for _ in images]


class FakeReranker:
    """Scores the i-th image of a call with `scores[i]`; `error` makes every call fail."""

    rejected = staticmethod(RerankerClient.rejected)

    def __init__(self, scores: list[float] | None = None, error: ApiError | None = None):
        self._scores = scores or []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def score(self, query, base64_images, *, image_mime="image/jpeg"):
        self.calls.append((query, len(base64_images)))
        if self._error is not None:
            raise self._error
        return self._scores[:len(base64_images)]


class FakeStore:

    def __init__(self, hits: list[FakeHit] | None = None):
        self._hits = hits or []
        self.upserts: list[dict] = []
        self.metadata: dict | None = None                     # the serving collection's
        self.collection_metadata: dict[str, dict] = {}        # every other collection's
        self._doc_ids: set[str] = set()
        self.deletes: list[str] = []
        self.delete_collections: list[str | None] = []
        self.physical_name = "c__v1"                          # where the fake alias points
        self.previous_name: str | None = None
        self.dropped: list[str] = []
        self.rollback_retired_at = 0
        self.on_build = None                                  # async hook, runs as a revision builds
        self.queries: list[dict] = []

    async def query(self, vector, *, top_k=5, score_threshold=None):
        self.queries.append({"top_k": top_k, "score_threshold": score_threshold})
        hits = [hit for hit in self._hits if score_threshold is None or hit.score >= score_threshold]

        return hits[:top_k]

    async def upsert(self, vectors, payloads, ids=None, wait=True, collection=None):
        self.upserts.append({"vectors": vectors, "payloads": payloads, "ids": ids, "collection": collection})

    async def create_collection(self):
        pass

    async def serving_physical(self) -> str:
        return self.physical_name

    async def resolve(self):
        return self.physical_name, self.previous_name

    async def create_revision_collection(self, physical: str) -> str:
        self.revision_collection = f"c__v{int(physical.rsplit('__v', 1)[1]) + 1}"
        self.collection_metadata[self.revision_collection] = dict(self.metadata or {})  # binding carried
        if self.on_build is not None:
            await self.on_build()
        return self.revision_collection

    async def drop_collection(self, name: str) -> None:
        self.dropped.append(name)

    def _serve(self, physical: str) -> None:
        """Repoint the fake alias, moving the serving metadata along with it."""
        self.collection_metadata[self.physical_name] = self.metadata or {}
        self.metadata = self.collection_metadata.pop(physical, {})
        self.physical_name = physical

    async def swap_alias(self, new_physical: str, old_physical: str, stale_previous) -> None:
        assert old_physical == self.physical_name
        self.swapped_to = new_physical
        self.previous_name = self.physical_name
        self._serve(new_physical)

    async def rollback_alias(self, restored: str) -> int:
        self._serve(restored)
        self.previous_name = None
        return self.rollback_retired_at

    async def get_collection_metadata(self, *, collection=None) -> dict:
        if collection not in (None, self.physical_name):
            return self.collection_metadata.get(collection, {})
        return self.metadata or {}

    async def set_collection_metadata(self, metadata: dict, *, collection=None) -> None:
        if collection not in (None, self.physical_name):
            self.collection_metadata[collection] = {**self.collection_metadata.get(collection, {}), **metadata}
        else:
            self.metadata = {**(self.metadata or {}), **metadata}             # Qdrant merges

    async def delete_by_doc(self, doc_id: str, *, wait: bool = True, collection=None) -> None:
        self.deletes.append(doc_id)
        self.delete_collections.append(collection)
        self._doc_ids.discard(doc_id)

    async def delete_by_docs(self, doc_ids, *, collection=None) -> None:
        for doc_id in doc_ids:
            await self.delete_by_doc(doc_id, collection=collection)

    async def iter_doc_ids(self, *, collection=None) -> set[str]:
        return set(self._doc_ids)

    async def close(self):
        pass


class FakeValkey:
    """In-memory ValkeyStore: the same primitives with the same semantics (TTL by the
    clock, NX / XX / IFEQ, DELIFEQ), so the real Coordinator runs on it. `down` makes
    every call fail as an unreachable Valkey does."""

    def __init__(self):
        self.values: dict[str, tuple[Any, float | None]] = {}  # name -> (value, expires_at)
        self.hashes: dict[str, dict[str, Any]] = {}
        self.sets: dict[str, set[str]] = {}
        self.down = False

    def _check(self) -> None:
        if self.down:
            raise ApiError(status_code=503, message="Valkey is unavailable")

    def _value(self, name: str):
        value, expires_at = self.values.get(name, (None, None))
        if expires_at is not None and time.monotonic() >= expires_at:
            del self.values[name]
            return None
        return value

    def expire(self, name: str) -> None:
        """Test helper: the key runs out now (a lease nobody renewed)."""
        self.values.pop(name, None)

    async def check_connection(self) -> None:
        self._check()

    async def get(self, name):
        self._check()
        return self._value(name)

    async def get_json(self, name):
        self._check()
        return self._value(name)

    async def set_json(self, name, value) -> None:
        self._check()
        self.values[name] = (value, None)

    async def set_nx(self, name, value) -> bool:
        self._check()
        if self._value(name) is not None:
            return False
        self.values[name] = (value, None)
        return True

    async def set_ifeq(self, name, value, expected) -> bool:
        self._check()
        if self._value(name) != expected:
            return False
        self.values[name] = (value, None)
        return True

    async def set_ttl(self, name, value, ttl_sec, *, nx=False, xx=False) -> bool:
        self._check()
        exists = self._value(name) is not None
        if (nx and exists) or (xx and not exists):
            return False
        self.values[name] = (value, time.monotonic() + ttl_sec)
        return True

    async def exists(self, name) -> bool:
        self._check()
        return self._value(name) is not None or bool(self.hashes.get(name)) or bool(self.sets.get(name))

    async def exists_each(self, names) -> list[bool]:
        return [await self.exists(name) for name in names]

    async def delete(self, *names) -> None:
        self._check()
        for name in names:
            self.values.pop(name, None)
            self.hashes.pop(name, None)
            self.sets.pop(name, None)

    async def delete_ifeq(self, name, expected) -> bool:
        self._check()
        if self._value(name) != expected:
            return False
        del self.values[name]
        return True

    async def hset_json(self, name, fields) -> None:
        self._check()
        self.hashes.setdefault(name, {}).update({k: dict(v) for k, v in fields.items()})

    async def hgetall_json(self, name) -> dict:
        self._check()
        return {k: dict(v) for k, v in self.hashes.get(name, {}).items()}

    async def hdel(self, name, *fields) -> None:
        self._check()
        for field_ in fields:
            self.hashes.get(name, {}).pop(field_, None)

    async def sadd(self, name, *members) -> None:
        self._check()
        if members:
            self.sets.setdefault(name, set()).update(members)

    async def smembers(self, name) -> set[str]:
        self._check()
        return set(self.sets.get(name, ()))

    async def close(self) -> None:
        pass


def make_coordinator(store=None, valkey=None, ttl: int = 30) -> Coordinator:
    """A real Coordinator over FakeValkey, its lease already taken. The lease never
    runs out by itself: a test ends it with `valkey.expire(lease_key(coordinator))`."""
    valkey = valkey if valkey is not None else FakeValkey()
    coordinator = Coordinator(valkey, store if store is not None else FakeStore(), process_ttl_sec=ttl)
    valkey.values[lease_key(coordinator)] = (coordinator.identity, None)
    return coordinator


def lease_key(coordinator: Coordinator) -> str:
    return f"procs:{coordinator.token}"


@dataclass(slots=True)
class Tile:
    tile_num: int
    width: int
    height: int
    image_hash: str
    data: str


@dataclass(slots=True)
class TileResult:
    """What FakeTiler.tile_stream emits: one doc's identity and its tiles."""

    doc_id: str
    doc_type: str
    source_url: str | None
    render_version: str
    page_height: int
    tile_count: int
    tiles: list[Tile]


class FakeTiler:

    RENDER_VERSION_MISMATCH = TilerService.RENDER_VERSION_MISMATCH
    unavailable = staticmethod(TilerService.unavailable)

    def __init__(self, result: TileResult | None = None, info: dict | None = None):
        self._result = result
        self._info = info or {"volume_id": "vol-test", "render_version": "rv-test"}
        self.get_tile_calls: list[tuple] = []
        self.existing: set[str] = set()
        self.metadata_patches: list[tuple] = []
        self.schema: dict = {
            "version": 1,
            "fields": [
                {"key": "name", "type": "text"},
                {"key": "virag_started_at", "type": "number"},
                {"key": "virag_enrolled_at", "type": "number"},
            ],
        }
        self.docs: list[dict] = []
        self.pending_docs: list[dict] = []
        self.max_concurrency = 0
        self.fail_tiles: set[tuple] = set()

    async def tile_stream(self, doc_type, *, document=None, url=None, metadata=None, force=False):
        res = self._result
        yield {"event": "begin", "doc_id": res.doc_id, "doc_type": res.doc_type,
               "source_url": res.source_url, "render_version": res.render_version}
        for t in res.tiles:
            yield {"event": "tile", "tile_num": t.tile_num, "width": t.width,
                   "height": t.height, "image_hash": t.image_hash, "data": t.data}
        yield {"event": "committed", "doc_id": res.doc_id, "doc_type": res.doc_type,
               "source_url": res.source_url, "render_version": res.render_version,
               "page_height": 0, "tile_count": res.tile_count, "result": "rendered"}

    async def get_tile(self, doc_id, tile_num, *, render_version=None, bulk=False) -> bytes:
        self.get_tile_calls.append((doc_id, tile_num))
        if (doc_id, tile_num) in self.fail_tiles:
            raise RuntimeError(f"tile {doc_id}:{tile_num} unavailable")
        if render_version is not None and render_version != self._info["render_version"]:
            raise ApiError(status_code=409, message="render_version mismatch", code=self.RENDER_VERSION_MISMATCH)
        return tile_bytes(doc_id, tile_num)

    async def get_info(self) -> dict:
        return self._info

    async def doc_exists(self, doc_id: str) -> bool:
        return doc_id in self.existing

    async def patch_metadata(self, doc_id: str, patch: dict) -> None:
        self.metadata_patches.append((doc_id, patch))

    async def get_schema(self) -> dict:
        return self.schema

    async def iter_docs(self, *, filters=None) -> list[dict]:
        # No filters = the Qdrant ⊆ Tiler listing; filters = the pending search.
        return list(self.pending_docs if filters else self.docs)


class FakeVLMResponse:
    """Minimal stand-in for the buffered upstream response used by _complete_response."""

    def __init__(self, status_code=200, json_body=None, content=b"", content_type="application/json"):
        self.status_code = status_code
        self._json = json_body
        self.content = content if json_body is None else _json_bytes(json_body)
        self.headers = {"content-type": content_type}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeStreamResponse:

    def __init__(self, chunks: list[bytes], status_code=200, content_type="text/event-stream"):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self) -> bytes:
        return b"".join(self._chunks)


class FakeVLM:

    def __init__(self, complete_response=None, stream_chunks=None):
        self._complete = complete_response or FakeVLMResponse(json_body={"choices": [{"index": 0}]})
        self._stream_chunks = stream_chunks or [b"data: {\"choices\":[]}\n\n", b"data: [DONE]\n\n"]
        self.complete_bodies: list[dict] = []
        self.stream_bodies: list[dict] = []

    async def complete(self, body: dict):
        self.complete_bodies.append(body)
        return self._complete

    @asynccontextmanager
    async def stream(self, body: dict):
        self.stream_bodies.append(body)
        yield FakeStreamResponse(self._stream_chunks)


def _json_bytes(obj) -> bytes:
    import json
    return json.dumps(obj).encode()


# ---------------------------------------------------------------------------
# Settings + app fixtures
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    rag = overrides.pop("rag_settings", RagSettings(framing_text="FRAME"))
    return Settings(
        api_key=overrides.pop("api_key", ""),
        admin_api_key=overrides.pop("admin_api_key", ""),
        rag_settings=rag,
        http_settings=HttpSettings(backoff_base_sec=0, backoff_cap_sec=0),
        qdrant_settings=QdrantStoreSettings(
            base_url="http://qdrant", grpc_port=6334, timeout_sec=60, max_retries=0,
            collection_name="c", vector_size=4, distance_type=Distance.COSINE,
            chunk_size=24, max_batch_chunks=1024,
        ),
        valkey_settings=ValkeySettings(url="redis://valkey:6379/0"),
        retriever_settings=RetrieverSettings(
            base_url="http://retriever/v1", model_name="retriever", api_key="k-retriever", batch_size=16,
            max_concurrency=0, bulk_priority=0, image_batching="auto", timeout_sec=60,
            connect_timeout_sec=10, max_connections=100, max_retries=0, top_k=5, threshold=None,
        ),
        reranker_settings=overrides.pop("reranker_settings", RerankerSettings(base_url=None)),
        tiler_settings=TilerSettings(
            base_url="http://tiler", max_concurrency=0, timeout_sec=120, connect_timeout_sec=10, max_connections=100, max_retries=0,
        ),
        vlm_settings=VLMSettings(
            base_url="http://vlm/v1", model_name="vlm", api_key="k-vlm", timeout_sec=120,
            connect_timeout_sec=10, max_connections=100, max_concurrency=0, max_retries=0,
        ),
        enroll_settings=overrides.pop("enroll_settings", EnrollSettings(batch_size=16)),
        reconcile_settings=ReconcileSettings(self_heal=True, interval_sec=0, pending_ttl_sec=3600),
    )


def make_store(settings: Settings | None = None):
    """A real QdrantStore (no network until used) configured like production."""
    from virag.clients.qdrant_store import QdrantStore

    settings = settings or make_settings()

    return QdrantStore.from_settings(settings.qdrant_settings, settings.http_settings)


def build_app(*, settings=None, store=None, coordinator=None, retriever=None, reranker=None, tiler=None, vlm=None) -> FastAPI:
    """The same routers as create_app(), but no lifespan — state is injected. A store
    without a coordinator gets one of its own; no reranker means it is off."""
    app = FastAPI()
    install_error_handlers(app)
    include_routers(app)

    app.state.settings = settings if settings is not None else make_settings()
    if store is not None:
        app.state.store = store
        app.state.coordinator = coordinator if coordinator is not None else make_coordinator(store)
    if retriever is not None:
        app.state.retriever = retriever
    app.state.reranker = reranker
    if tiler is not None:
        app.state.tiler = tiler
    if vlm is not None:
        app.state.vlm = vlm
    return app


@pytest.fixture
def make_client():
    """Factory: build a TestClient over an app with the given fakes/state."""
    def _make(**kwargs) -> TestClient:
        app = build_app(**kwargs)
        return TestClient(app, raise_server_exceptions=False)
    return _make


def sample_tile_result(doc_id="doc-a", n_tiles=2, doc_type="PDF") -> TileResult:
    return TileResult(
        doc_id=doc_id,
        doc_type=doc_type,
        source_url=None,
        render_version="rv-test",
        page_height=0,
        tile_count=n_tiles,
        tiles=[
            Tile(tile_num=i, width=875, height=1024, image_hash=tile_hash(doc_id, i), data=f"b64-{i}")
            for i in range(n_tiles)
        ],
    )


def hit_for(doc_id, tile_num, score=0.9, name="Doc A") -> FakeHit:
    """A Qdrant hit whose image_hash matches what FakeTiler.get_tile will produce."""
    return FakeHit(
        score=score,
        payload={
            "doc_id": doc_id,
            "tile_num": tile_num,
            "image_hash": tile_hash(doc_id, tile_num),
            "metadata": {"name": name},
        },
    )
