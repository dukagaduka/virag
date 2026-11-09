from __future__ import annotations

import logging
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Sequence

import grpc
import numpy as np
import orjson

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Batch,
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    VectorParams
)

from virag.errors import ApiError
from virag.settings import HttpSettings, QdrantStoreSettings
from virag.utils import unavailable_as_503


log = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryResult:
    id: str
    score: float
    payload: dict


class QdrantStore:
    """Points are addressed through a stable serving alias; the physical collection it
    points to is versioned so a /revision can blue-green swap it atomically. The
    collection it replaced is retained for a rollback until the next revision.

    Nothing about the alias is cached: another virag process may swap it at any time,
    so `resolve()` reads it from Qdrant whenever a physical name matters.
    """

    # Cost of one vector float on the wire: 4 B packed in protobuf, rounded up for
    # framing. (The REST/JSON body measured ~21.5 B/float — see the old enroll 400s.)
    BYTES_PER_FLOAT = 6

    # gRPC methods replayed on UNAVAILABLE: every point op (upserts carry deterministic
    # ids, deletes are by filter) and the collection calls that are reads or rewrite the
    # same values. Create / Delete / UpdateAliases are not: a replay after an
    # unacknowledged success would report a false "already exists" / "alias not found".
    RETRY_SERVICES = ("qdrant.Points",)
    RETRY_COLLECTION_METHODS = ("Get", "List", "CollectionExists", "ListAliases", "ListCollectionAliases", "Update")

    # Statuses meaning "Qdrant is unusable right now" — surfaced as a 503. A rejected
    # QDRANT_API_KEY is one: the details name UNAUTHENTICATED.
    UNAVAILABLE_CODES = (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAUTHENTICATED,
    )

    # doc_ids per filtered delete request — one MatchAny filter instead of N requests.
    DELETE_DOCS_CHUNK = 256

    # Points per scroll page when listing doc_ids (payload only, no vectors).
    SCROLL_PAGE = 1024


    # Defined first: it decorates methods below while the class body is still executing.
    # Unreachable means: still so after the channel's own retries.
    _unavailable_as_503 = staticmethod(unavailable_as_503(
        "Qdrant",
        grpc.RpcError,
        when=lambda e: e.code() in QdrantStore.UNAVAILABLE_CODES,              # type: ignore[attr-defined]
        details=lambda e: {"grpc_status": e.code().name, "error": e.details()},  # type: ignore[attr-defined]
    ))


    def __init__(
        self,
        base_url: str,
        collection_name: str,
        vector_size: int,
        *,
        grpc_port: int,
        api_key: str,
        timeout: int,
        max_retries: int,
        backoff_base: float,
        backoff_cap: float,
        distance_type: Distance,
        chunk_byte_budget: int,
        max_batch_chunks: int,
    ) -> None:

        # General connection params
        self.base_url = base_url
        # `serving_name` is the alias clients query. Point ops use it; collection-level
        # ops (create/index/metadata) use the physical collection it resolves to.
        self.serving_name = collection_name

        # Internal collection params
        self.vector_size = vector_size
        self.distance_type = distance_type

        # Upsert batching limits
        self.chunk_byte_budget = chunk_byte_budget
        self.max_batch_chunks = max_batch_chunks

        # gRPC: one multiplexed HTTP/2 channel — no pool of idle HTTP/1.1 connections to
        # race the server's keep-alive — and retries done by the channel itself. The key
        # travels in plain text inside the compose network by design; the client's
        # warning about it would only repeat that on every start.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

            self.qdrant = AsyncQdrantClient(
                url=self.base_url,
                prefer_grpc=True,
                grpc_port=grpc_port,
                api_key=api_key or None,
                timeout=timeout,
                grpc_options={
                    "grpc.enable_retries": 1,
                    "grpc.service_config": self._service_config(max_retries, backoff_base, backoff_cap),
                },
            )


    @classmethod
    def from_settings(cls, qs: QdrantStoreSettings, hs: HttpSettings) -> QdrantStore:
        return cls(
            base_url=qs.base_url,
            collection_name=qs.collection_name,
            vector_size=qs.vector_size,
            grpc_port=qs.grpc_port,
            api_key=qs.api_key,
            timeout=qs.timeout_sec,
            max_retries=qs.max_retries,
            backoff_base=hs.backoff_base_sec,
            backoff_cap=hs.backoff_cap_sec,
            distance_type=qs.distance_type,
            chunk_byte_budget=qs.chunk_byte_budget,
            max_batch_chunks=qs.max_batch_chunks,
        )


    @_unavailable_as_503
    async def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        *,
        ids: Sequence[str] | None = None,
        wait: bool = True,
        collection: str | None = None,
    ):

        assert len(vectors) == len(payloads), "Sizes of `vectors` and `payloads` must be equal"
        assert ids is None or len(ids) == len(vectors), "Sizes of `ids` and `vectors` must be equal"

        target = collection or self.serving_name

        do_size_check = True
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            if vectors.shape[1] == self.vector_size:
                do_size_check = False
            else:
                raise ValueError(f"Every vector's size must equal {self.vector_size}")

        chunk_size = self._chunk_size(payloads)

        for i in range(0, len(vectors), chunk_size):
            v_batch = vectors[i:i+chunk_size]
            p_batch = payloads[i:i+chunk_size]

            id_batch = (
                [str(uuid.uuid4()) for _ in range(len(v_batch))]
                if ids is None
                else list(ids[i:i+chunk_size])
            )

            if do_size_check:
                for v in v_batch:  # validate per-batch, not all up front
                    if len(v) != self.vector_size:
                        raise ValueError(f"Every vector's size must equal {self.vector_size}")

            await self.qdrant.upsert(
                collection_name=target,
                wait=wait,
                points=Batch(
                    ids=id_batch,
                    vectors=v_batch,       # type: ignore
                    payloads=list(p_batch),  # type: ignore
                ),
            )


    def _chunk_size(self, payloads: Sequence[dict]) -> int:
        # Estimate the serialized cost of one point and derive how many fit
        # into the byte budget, capped by `max_batch_chunks`. Payloads are
        # homogeneous, so a single sample is enough to size the batch.
        payload_bytes = len(orjson.dumps(payloads[0], default=str)) if payloads else 0

        est_point_bytes = self.vector_size * self.BYTES_PER_FLOAT + payload_bytes

        by_bytes = max(1, self.chunk_byte_budget // est_point_bytes)

        return min(self.max_batch_chunks, by_bytes)


    @_unavailable_as_503
    async def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None
    ) -> list[QueryResult]:
        """The `top_k` points nearest to `vector`. Qdrant applies `score_threshold` by the
        collection's distance: for cosine and dot it is a floor, for euclid and manhattan a
        ceiling."""

        assert len(vector) == self.vector_size, (
            f"Vector size mismatch: got {len(vector)} but expected {self.vector_size}"
        )

        result = await self.qdrant.query_points(
            collection_name=self.serving_name,
            query=vector, # type: ignore
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )

        query_results = []

        for point in result.points:
            query_results.append(
                QueryResult(id=point.id, score=point.score, payload=point.payload) # type: ignore
            )

        return query_results


    # ------------------------------------------------------------------ #
    # collection lifecycle (alias-aware)
    # ------------------------------------------------------------------ #
    async def create_collection(self):
        physical = await self._resolve_or_create()

        await self._validate(physical)
        await self._ensure_indexes(physical)


    async def _alias_target(self) -> str | None:
        """The physical collection the serving alias points to, or None without the alias."""
        aliases = await self.qdrant.get_aliases()
        alias_map = {a.alias_name: a.collection_name for a in aliases.aliases}

        return alias_map.get(self.serving_name)


    async def serving_physical(self) -> str:
        """The physical collection the serving alias points to now."""
        physical = await self._alias_target()
        if physical is None:
            raise RuntimeError(f"Qdrant alias `{self.serving_name}` is missing")

        return physical


    async def resolve(self) -> tuple[str, str | None]:
        """(serving physical collection, rollback target) as they are in Qdrant now."""
        physical = await self.serving_physical()

        return physical, await self._find_previous(physical)


    async def _resolve_or_create(self) -> str:
        """Resolve the serving alias to its physical collection, migrating or creating it."""
        if (physical := await self._alias_target()) is not None:
            return physical

        if await self.qdrant.collection_exists(self.serving_name):
            # A plain collection under the serving name predates the alias scheme; the
            # alias cannot claim a taken name, so refuse rather than fail obscurely.
            message = (
                f"Qdrant collection `{self.serving_name}` exists as a plain collection "
                f"(pre-alias layout); drop it, or choose another QDRANT_COLLECTION_NAME"
            )
            log.critical("FATAL: %s", message)

            raise RuntimeError(message)

        physical = f"{self.serving_name}__v1"

        try:
            await self._create_physical(physical)
            await self._alias_to(physical)

        except Exception:
            # Another virag process starting on the same empty Qdrant created it first.
            if (resolved := await self._alias_target()) is None:
                raise

            return resolved

        return physical


    async def _create_physical(self, name: str) -> None:
        if not await self.qdrant.collection_exists(name):
            await self.qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance_type),
            )


    async def _validate(self, name: str) -> None:
        vectors_info = (await self.qdrant.get_collection(name)).config.params.vectors

        size = vectors_info.size # type: ignore
        distance_type = vectors_info.distance # type: ignore

        assert size == self.vector_size, (
            f"Vector size mismatch: collection `{name}` has size {size}, "
            f"but expected {self.vector_size}"
        )

        assert distance_type == self.distance_type, (
            f"Distance type mismatch: collection `{name}` uses {distance_type}, "
            f"but expected {self.distance_type}"
        )


    async def _ensure_indexes(self, name: str) -> None:
        # `doc_id` is filtered on for per-document delete and the Qdrant ⊆ Tiler
        # reconcile; the index makes those filtered ops cheap. Idempotent.
        await self.qdrant.create_payload_index(
            collection_name=name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )


    async def _alias_to(self, physical: str) -> None:
        await self.qdrant.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=physical, alias_name=self.serving_name)
                )
            ]
        )


    # ------------------------------------------------------------------ #
    # revision (blue-green re-embed target)
    # ------------------------------------------------------------------ #
    def _version(self, name: str | None) -> int | None:
        """The `N` of a `<serving>__vN` physical name, or None for any other name."""
        match = re.match(rf"^{re.escape(self.serving_name)}__v(\d+)$", name or "")

        return int(match.group(1)) if match else None


    def _next_physical(self, physical: str) -> str:
        """The next versioned physical name, e.g. `<serving>__v2` after `__v1`."""
        version = self._version(physical) or 1

        return f"{self.serving_name}__v{version + 1}"


    async def _find_previous(self, physical: str) -> str | None:
        """The newest retained version older than `physical` — the rollback target."""
        current = self._version(physical)
        if current is None:
            return None

        collections = (await self.qdrant.get_collections()).collections
        older = [
            (version, c.name)
            for c in collections
            if (version := self._version(c.name)) is not None and version < current
        ]

        return max(older)[1] if older else None


    async def create_revision_collection(self, physical: str) -> str:
        """Create the physical collection after `physical` (indexed, binding metadata
        carried) to re-embed into. A leftover of an aborted or crashed revision is
        dropped first, so a build never resumes on top of stale points."""
        new_name = self._next_physical(physical)

        if await self.qdrant.collection_exists(new_name):
            log.warning("revision: dropping leftover collection %s from an earlier attempt", new_name)
            await self.qdrant.delete_collection(collection_name=new_name)

        await self._create_physical(new_name)
        await self._ensure_indexes(new_name)

        meta = await self._collection_metadata(physical)
        if meta:
            await self.qdrant.update_collection(collection_name=new_name, metadata=meta)

        return new_name


    async def drop_collection(self, name: str) -> None:
        """Drop a non-serving physical collection (an aborted build, an abandoned one)."""
        if name == await self._alias_target():
            raise ValueError(f"refusing to drop the serving collection {name}")

        await self.qdrant.delete_collection(collection_name=name)


    async def swap_alias(self, new_physical: str, old_physical: str, stale_previous: str | None) -> None:
        """Atomically repoint the serving alias from `old_physical` to `new_physical`. The
        replaced one becomes the rollback target, stamped with `retired_at` so a rollback
        knows which enrolls it missed; `stale_previous`, the one it supersedes, is dropped."""
        meta = await self._collection_metadata(old_physical)
        await self.qdrant.update_collection(
            collection_name=old_physical,
            metadata={**meta, "retired_at": int(time.time())},
        )

        await self._repoint_alias(new_physical)

        if stale_previous and stale_previous not in (new_physical, old_physical):
            await self.qdrant.delete_collection(collection_name=stale_previous)


    async def rollback_alias(self, restored: str) -> int:
        """Point the serving alias back at `restored`, the retained collection. Returns its
        `retired_at`: documents enrolled since then live only in the collection it
        replaces, which the caller drops once that one's writers are done."""
        info = (await self.qdrant.get_collection(restored)).config.params.vectors
        if info.size != self.vector_size:  # type: ignore
            raise ApiError(
                status_code=409,
                message=f"Previous collection {restored} holds {info.size}-d vectors "  # type: ignore
                        f"but QDRANT_VECTOR_SIZE is {self.vector_size}; revert the retriever config first",
            )

        meta = await self._collection_metadata(restored)
        retired_at = int(meta.get("retired_at") or 0)

        # Explicit null: Qdrant merges metadata updates, so an omitted key would survive.
        await self.qdrant.update_collection(collection_name=restored, metadata={**meta, "retired_at": None})
        await self._repoint_alias(restored)

        return retired_at


    async def _repoint_alias(self, physical: str) -> None:
        await self.qdrant.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=self.serving_name)),
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=physical, alias_name=self.serving_name)
                ),
            ]
        )


    # ------------------------------------------------------------------ #
    # metadata / points
    # ------------------------------------------------------------------ #
    async def _collection_metadata(self, name: str) -> dict:
        info = await self.qdrant.get_collection(name)
        return getattr(info.config, "metadata", None) or {}


    async def get_collection_metadata(self, *, collection: str | None = None) -> dict:
        """Metadata of the serving collection, or of `collection` (a revision build or
        the rollback target)."""
        return await self._collection_metadata(collection or await self.serving_physical())


    async def set_collection_metadata(self, metadata: dict, *, collection: str | None = None) -> None:
        """Merge `metadata` into the serving collection's (or `collection`'s) metadata."""
        await self.qdrant.update_collection(
            collection_name=collection or await self.serving_physical(),
            metadata=metadata,
        )


    @_unavailable_as_503
    async def delete_by_doc(
        self,
        doc_id: str,
        *,
        wait: bool = True,
        collection: str | None = None,
    ) -> None:
        """Delete every point belonging to one document — the enroll rollback + reconcile primitive."""
        await self.qdrant.delete(
            collection_name=collection or self.serving_name,
            wait=wait,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )


    @_unavailable_as_503
    async def delete_by_docs(
        self,
        doc_ids: Sequence[str],
        *,
        collection: str | None = None,
    ) -> None:
        """Delete every point of many documents, `DELETE_DOCS_CHUNK` doc_ids per request."""
        chunk = self.DELETE_DOCS_CHUNK
        doc_ids = list(doc_ids)

        for i in range(0, len(doc_ids), chunk):
            await self.qdrant.delete(
                collection_name=collection or self.serving_name,
                wait=True,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids[i:i + chunk]))]
                ),
            )


    @_unavailable_as_503
    async def iter_doc_ids(self, *, collection: str | None = None) -> set[str]:
        """Scroll the whole collection and return the distinct doc_ids it holds."""
        doc_ids: set[str] = set()
        offset = None

        while True:
            points, offset = await self.qdrant.scroll(
                collection_name=collection or self.serving_name,
                limit=self.SCROLL_PAGE,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )

            for point in points:
                doc_id = (point.payload or {}).get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)

            if offset is None:
                break

        return doc_ids


    async def close(self):
        await self.qdrant.close()


    @classmethod
    def _service_config(cls, max_retries: int, backoff_base: float, backoff_cap: float) -> str:
        """gRPC service config: replay `RETRY_*` methods on UNAVAILABLE with the channel's
        own jittered exponential backoff (the same base/cap as the HTTP upstreams)."""
        if max_retries <= 0:
            return "{}"

        names = [{"service": service} for service in cls.RETRY_SERVICES]
        names += [{"service": "qdrant.Collections", "method": method} for method in cls.RETRY_COLLECTION_METHODS]

        # gRPC requires strictly positive backoffs.
        initial = max(backoff_base, 0.01)
        ceiling = max(backoff_cap, initial)

        return orjson.dumps({
            "methodConfig": [
                {
                    "name": names,
                    "retryPolicy": {
                        "maxAttempts": max_retries + 1,
                        "initialBackoff": f"{initial:.3f}s",
                        "maxBackoff": f"{ceiling:.3f}s",
                        "backoffMultiplier": 2,
                        "retryableStatusCodes": ["UNAVAILABLE"],
                    },
                }
            ]
        }).decode()
