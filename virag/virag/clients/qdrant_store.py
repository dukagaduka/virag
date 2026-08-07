from __future__ import annotations

from dataclasses import dataclass
import json
import re
import uuid
from typing import Sequence

import numpy as np

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
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams
)

from virag.settings import QdrantStoreSettings


# Empirical cost of one JSON-serialized float (repr + separator); see enroll
# 400s where 920x2048 vectors produced a ~40 MB body (~21.5 B/float measured).
BYTES_PER_FLOAT_JSON = 24


@dataclass(slots=True)
class QueryResult:
    id: str
    score: float
    payload: dict


class QdrantStore:
    """Points are addressed through a stable serving alias; the physical collection it
    points to is versioned so a /revision can blue-green swap it atomically.
    """

    def __init__(
        self,
        base_url: str,
        collection_name: str,
        vector_size: int,
        *,
        distance_type: Distance = Distance.COSINE,
        chunk_byte_budget: int = 24 * 1024 * 1024,
        max_batch_chunks: int = 1024
    ) -> None:

        # General connection params
        self.base_url = base_url
        # `serving_name` is the alias clients query; `physical_name` is the collection it
        # resolves to (set by create_collection). Point ops use the alias, collection-level
        # ops (create/index/metadata) use the physical name.
        self.serving_name = collection_name
        self.physical_name = collection_name

        # Internal collection params
        self.vector_size = vector_size
        self.distance_type = distance_type

        # Upsert batching limits
        self.chunk_byte_budget = chunk_byte_budget
        self.max_batch_chunks = max_batch_chunks

        self.qdrant = AsyncQdrantClient(
            url=self.base_url
        )


    @classmethod
    def from_settings(cls, qs: QdrantStoreSettings) -> QdrantStore:
        return cls(
            base_url=qs.base_url,
            collection_name=qs.collection_name,
            vector_size=qs.vector_size,
            distance_type=qs.distance_type,
            chunk_byte_budget=qs.chunk_byte_budget,
            max_batch_chunks=qs.max_batch_chunks,
        )


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
        payload_bytes = (
            len(json.dumps(payloads[0], default=str).encode("utf-8"))
            if payloads
            else 0
        )

        est_point_bytes = self.vector_size * BYTES_PER_FLOAT_JSON + payload_bytes

        by_bytes = max(1, self.chunk_byte_budget // est_point_bytes)

        return min(self.max_batch_chunks, by_bytes)


    async def query(
        self,
        vector: Sequence[float],
        *,
        top_n: int = 5
    ) -> list[QueryResult]:

        assert len(vector) == self.vector_size, (
            f"Vector size mismatch: got {len(vector)} but expected {self.vector_size}"
        )

        result = await self.qdrant.query_points(
            collection_name=self.serving_name,
            query=vector, # type: ignore
            limit=top_n,
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
        self.physical_name = await self._resolve_or_create()
        await self._validate(self.physical_name)
        await self._ensure_indexes(self.physical_name)


    async def _resolve_or_create(self) -> str:
        """Resolve the serving alias to its physical collection, migrating or creating it."""
        aliases = await self.qdrant.get_aliases()
        alias_map = {a.alias_name: a.collection_name for a in aliases.aliases}

        if self.serving_name in alias_map:
            return alias_map[self.serving_name]

        if await self.qdrant.collection_exists(self.serving_name):
            # Legacy plain collection (pre-alias) — migrate it under the alias scheme.
            return await self._migrate_legacy()

        physical = f"{self.serving_name}__v1"
        await self._create_physical(physical)
        await self._alias_to(physical)

        return physical


    async def _migrate_legacy(self) -> str:
        """One-time: copy a plain `serving_name` collection into `__v1` and alias it.

        The plain name must be freed before the alias can claim it; this runs at startup
        before any traffic, so the brief window with neither is invisible.
        """
        physical = f"{self.serving_name}__v1"

        if not await self.qdrant.collection_exists(physical):
            await self._create_physical(physical)
            meta = await self._collection_metadata(self.serving_name)
            if meta:
                await self.qdrant.update_collection(collection_name=physical, metadata=meta)
            await self._copy_points(self.serving_name, physical)

        # Never drop the source until the copy is provably complete.
        src_count = (await self.qdrant.count(collection_name=self.serving_name, exact=True)).count
        dst_count = (await self.qdrant.count(collection_name=physical, exact=True)).count
        if dst_count != src_count:
            raise RuntimeError(
                f"legacy migration count mismatch: {self.serving_name}={src_count} "
                f"but {physical}={dst_count}; refusing to drop the source"
            )

        await self.qdrant.delete_collection(collection_name=self.serving_name)
        await self._alias_to(physical)

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


    async def _copy_points(self, src: str, dst: str) -> None:
        offset = None
        while True:
            points, offset = await self.qdrant.scroll(
                collection_name=src,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if points:
                await self.qdrant.upsert(
                    collection_name=dst,
                    wait=True,
                    points=[
                        PointStruct(id=p.id, vector=p.vector, payload=p.payload)  # type: ignore
                        for p in points
                    ],
                )
            if offset is None:
                break


    # ------------------------------------------------------------------ #
    # revision (blue-green re-embed target)
    # ------------------------------------------------------------------ #
    def _next_physical(self) -> str:
        """The next versioned physical name, e.g. `<serving>__v2` after `__v1`."""
        match = re.match(rf"^{re.escape(self.serving_name)}__v(\d+)$", self.physical_name or "")
        version = int(match.group(1)) if match else 1

        return f"{self.serving_name}__v{version + 1}"


    async def create_revision_collection(self) -> str:
        """Create the next physical collection (indexed, binding metadata carried) to re-embed into."""
        new_name = self._next_physical()

        await self._create_physical(new_name)
        await self._ensure_indexes(new_name)

        meta = await self._collection_metadata(self.physical_name)
        if meta:
            await self.qdrant.update_collection(collection_name=new_name, metadata=meta)

        return new_name


    async def swap_alias(self, new_physical: str) -> None:
        """Atomically repoint the serving alias to `new_physical` and drop the old collection."""
        old_physical = self.physical_name

        await self.qdrant.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=self.serving_name)),
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=new_physical, alias_name=self.serving_name)
                ),
            ]
        )

        self.physical_name = new_physical

        if old_physical and old_physical != new_physical:
            await self.qdrant.delete_collection(collection_name=old_physical)


    # ------------------------------------------------------------------ #
    # metadata / points
    # ------------------------------------------------------------------ #
    async def _collection_metadata(self, name: str) -> dict:
        info = await self.qdrant.get_collection(name)
        return getattr(info.config, "metadata", None) or {}


    async def get_collection_metadata(self) -> dict:
        return await self._collection_metadata(self.physical_name)


    async def set_collection_metadata(self, metadata: dict) -> None:
        await self.qdrant.update_collection(
            collection_name=self.physical_name,
            metadata=metadata,
        )


    async def delete_by_doc(self, doc_id: str, *, wait: bool = True) -> None:
        """Delete every point belonging to one document — the enroll rollback + reconcile primitive."""
        await self.qdrant.delete(
            collection_name=self.serving_name,
            wait=wait,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )


    async def iter_doc_ids(self) -> set[str]:
        """Scroll the whole collection and return the distinct doc_ids it holds."""
        doc_ids: set[str] = set()
        offset = None

        while True:
            points, offset = await self.qdrant.scroll(
                collection_name=self.serving_name,
                limit=1024,
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
