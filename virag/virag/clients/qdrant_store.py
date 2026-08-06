from __future__ import annotations

from dataclasses import dataclass
import json
import uuid
from typing import Sequence

import numpy as np

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Batch,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
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
        self.collection_name = collection_name

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
    ):

        assert len(vectors) == len(payloads), "Sizes of `vectors` and `payloads` must be equal"
        assert ids is None or len(ids) == len(vectors), "Sizes of `ids` and `vectors` must be equal"

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
                collection_name=self.collection_name,
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
            collection_name=self.collection_name,
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
    
    
    async def create_collection(self):
        if not await self.qdrant.collection_exists(self.collection_name):
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=self.distance_type),
            )

        else:
            vectors_info = (await self.qdrant.get_collection(
                self.collection_name
            )).config.params.vectors

            size = vectors_info.size # type: ignore
            distance_type = vectors_info.distance # type: ignore

            assert size == self.vector_size, (
                f"Vector size mismatch: collection `{self.collection_name}` has size {size}, "
                f"but expected {self.vector_size}"
            )

            assert distance_type == self.distance_type, (
                f"Distance type mismatch: collection `{self.collection_name}` uses {distance_type}, "
                f"but expected {self.distance_type}"
            )

        await self._ensure_indexes()


    async def _ensure_indexes(self):
        # `doc_id` is filtered on for per-document delete and the Qdrant ⊆ Tiler
        # reconcile; the index makes those filtered ops cheap. Idempotent.
        await self.qdrant.create_payload_index(
            collection_name=self.collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
            
            
    async def clear_collection(
        self,
        *,
        vector_size: int | None = None,
        distance_type: Distance | None = None
    ):
        
        await self._delete()
        
        if vector_size is not None and vector_size != self.vector_size:
            self.vector_size = vector_size
            
        if distance_type is not None and distance_type != self.distance_type:
            self.distance_type = distance_type

        await self.create_collection()
        
    
    async def get_collection_metadata(self) -> dict:
        info = await self.qdrant.get_collection(self.collection_name)
        return getattr(info.config, "metadata", None) or {}


    async def set_collection_metadata(self, metadata: dict) -> None:
        await self.qdrant.update_collection(
            collection_name=self.collection_name,
            metadata=metadata,
        )


    async def delete_by_doc(self, doc_id: str, *, wait: bool = True) -> None:
        """Delete every point belonging to one document — the enroll rollback + reconcile primitive."""
        await self.qdrant.delete(
            collection_name=self.collection_name,
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
                collection_name=self.collection_name,
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


    async def sample_doc_ids(self, limit: int) -> list[str]:
        points, _ = await self.qdrant.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        doc_ids: list[str] = []
        for point in points:
            doc_id = (point.payload or {}).get("doc_id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)

        return doc_ids


    async def close(self):
        await self.qdrant.close()
        
    
    async def _delete(self):
        await self.qdrant.delete_collection(collection_name=self.collection_name)
                
                
    
    

        
        
        
    