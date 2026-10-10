"""Shared fixtures: in-memory fakes for the four backends and a lifespan-free app.

The real `lifespan` builds network clients and runs dependency + binding checks, so
the tests build the app directly and populate `app.state` with fakes instead. Each
handler only ever reaches its backends through `request.app.state`, so this exercises
the real routing, RAG pipeline and error handling with no network."""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client.models import Distance

from virag.api import chat_completions as chat_api
from virag.api import enroll as enroll_api
from virag.api import ops as ops_api
from virag.api import proxy as proxy_api
from virag.clients.tiler_client import Tile, TileResult
from virag.errors import install_error_handlers
from virag.settings import (
    EmbeddingsSettings,
    QdrantStoreSettings,
    RagSettings,
    ReconcileSettings,
    Settings,
    TilerSettings,
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

class FakeEmbeddings:

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.chunk_size = 16
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[str]] = []

    async def embed_texts(self, texts):
        self.text_calls.append(list(texts))
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]

    async def embed_images(self, images):
        self.image_calls.append(list(images))
        return [[0.5] * self.dim for _ in images]


class FakeStore:

    def __init__(self, hits: list[FakeHit] | None = None):
        self._hits = hits or []
        self.upserts: list[dict] = []
        self.metadata: dict | None = None
        self._doc_ids: set[str] = set()
        self.deletes: list[str] = []

    async def query(self, vector, top_n):
        return self._hits[:top_n]

    async def upsert(self, vectors, payloads, ids=None, wait=True, collection=None):
        self.upserts.append({"vectors": vectors, "payloads": payloads, "ids": ids, "collection": collection})

    async def create_collection(self):
        pass

    async def create_revision_collection(self) -> str:
        self.revision_collection = "c__v2"
        return self.revision_collection

    async def swap_alias(self, new_physical: str) -> None:
        self.swapped_to = new_physical

    async def get_collection_metadata(self) -> dict:
        return self.metadata or {}

    async def set_collection_metadata(self, metadata: dict) -> None:
        self.metadata = dict(metadata)

    async def delete_by_doc(self, doc_id: str, *, wait: bool = True) -> None:
        self.deletes.append(doc_id)
        self._doc_ids.discard(doc_id)

    async def iter_doc_ids(self) -> set[str]:
        return set(self._doc_ids)

    async def close(self):
        pass


class FakeTiler:

    def __init__(self, result: TileResult | None = None, info: dict | None = None):
        self._result = result
        self._info = info or {"secret_hash": "", "render_version": "rv-test"}
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

    async def tile(self, doc_type, *, document=None, url=None, metadata=None,
                   store=True, include_bytes=True, force=False):
        return self._result

    async def tile_stream(self, doc_type, *, document=None, url=None, metadata=None, force=False):
        res = self._result
        yield {"event": "begin", "doc_id": res.doc_id, "doc_type": res.doc_type,
               "source_url": res.source_url, "render_version": res.render_version}
        for t in res.tiles:
            yield {"event": "tile", "tile_num": t.tile_num, "width": t.width,
                   "height": t.height, "image_hash": t.image_hash, "data": t.data}
        yield {"event": "committed", "doc_id": res.doc_id, "doc_type": res.doc_type,
               "source_url": res.source_url, "render_version": res.render_version,
               "page_height": 0, "tile_count": res.tile_count}

    async def get_tile(self, doc_id, tile_num) -> bytes:
        self.get_tile_calls.append((doc_id, tile_num))
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
    rag = overrides.pop("rag_settings", RagSettings(top_n=5, score_threshold=None, framing_text="FRAME"))
    return Settings(
        host="127.0.0.1",
        port=8080,
        db_secret=overrides.pop("db_secret", ""),
        rag_settings=rag,
        qdrant_settings=QdrantStoreSettings(
            base_url="http://qdrant", collection_name="c", vector_size=4, distance_type=Distance.COSINE,
            chunk_byte_budget=24 * 1024 * 1024, max_batch_chunks=1024,
        ),
        embeddings_settings=EmbeddingsSettings(
            base_url="http://emb/v1", model_name="emb", api_key="k-emb", chunk_size=16, timeout=60,
        ),
        tiler_settings=TilerSettings(base_url="http://tiler", timeout=120),
        vlm_settings=VLMSettings(
            base_url="http://vlm/v1", model_name="vlm", api_key="k-vlm", timeout=120,
            max_concurrency=0, max_retries=0,
        ),
        reconcile_settings=ReconcileSettings(self_heal=True, interval_sec=0, pending_ttl_sec=3600),
    )


def build_app(*, settings=None, store=None, embeddings=None, tiler=None, vlm=None) -> FastAPI:
    """The same routers as create_app(), but no lifespan — state is injected."""
    app = FastAPI()
    install_error_handlers(app)
    app.include_router(ops_api.router)
    app.include_router(enroll_api.router)
    app.include_router(chat_api.router)
    app.include_router(proxy_api.router)

    app.state.settings = settings if settings is not None else make_settings()
    if store is not None:
        app.state.store = store
    if embeddings is not None:
        app.state.embeddings = embeddings
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
        stored=True,
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
