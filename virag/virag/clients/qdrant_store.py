from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Sequence

import numpy as np

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Batch, 
    Distance, 
    VectorParams
)

from virag.settings import QdrantStoreSettings


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
        distance_type: Distance = Distance.COSINE
    ) -> None:
        
        # General connection params
        self.base_url = base_url
        self.collection_name = collection_name
        
        # Internal collection params
        self.vector_size = vector_size
        self.distance_type = distance_type
        
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
        )
            
            
    async def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        *,
        ids: Sequence[str] | None = None,
        wait: bool = True,
        chunk_size: int = 1024,
    ):

        assert isinstance(chunk_size, int) and chunk_size > 0, "`chunk_size` must be a positive integer"
        assert len(vectors) == len(payloads), "Sizes of `vectors` and `payloads` must be equal"
        assert ids is None or len(ids) == len(vectors), "Sizes of `ids` and `vectors` must be equal"

        do_size_check = True
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            if vectors.shape[1] == self.vector_size:
                do_size_check = False
            else:
                raise ValueError(f"Every vector's size must equal {self.vector_size}")

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
                
                
    
    

        
        
        
    