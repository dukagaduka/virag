from __future__ import annotations

import logging
from contextlib import nullcontext

import httpx2

from virag.clients.http_service import OpenAIService
from virag.errors import ApiError
from virag.settings import HttpSettings, RetrieverSettings
from virag.utils import error_body, gather


log = logging.getLogger(__name__)


class RetrieverClient(OpenAIService):
    """The retrieval embedding model behind any OpenAI-compatible `/embeddings` API.

    Texts go through the standard `input` field. Images go through chat-style
    `messages` (a vLLM extension); several images per request only when the upstream
    accepts a batch of conversations (see `image_batching` and `check_connection`).

    Calls marked `bulk` (enroll, reconcile, revision) are capped by `max_concurrency`
    and may carry `bulk_priority`; interactive calls (chat queries) bypass both, so a
    large ingest never queues a user's query behind it.
    """

    # Replies meaning the upstream rejected the *content* of a request (bad item, body
    # too large, unsupported batch shape) rather than failing transiently. Only these
    # split a batch into single-item requests; transient failures are retried as-is by
    # the transport, since splitting would multiply load on an already struggling upstream.
    SPLIT_STATUSES = frozenset({400, 413, 422})


    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        *,
        batch_size: int,
        bulk_priority: int,
        image_batching: str,
        **http_options,
    ):

        super().__init__(base_url, model_name, api_key, **http_options)

        self.batch_size     = batch_size
        self.bulk_priority  = bulk_priority
        self.image_batching = image_batching

        # Resolved by check_connection; until then only an explicit "on" batches images.
        self.batch_images = image_batching == "on"
        self.dimension: int | None = None


    @property
    def route(self) -> str:
        return f"{self.base_url}/embeddings"


    @classmethod
    def from_settings(cls, rs: RetrieverSettings, hs: HttpSettings) -> RetrieverClient:
        return cls(
            base_url=rs.base_url,
            model_name=rs.model_name,
            api_key=rs.api_key,
            batch_size=rs.batch_size,
            bulk_priority=rs.bulk_priority,
            image_batching=rs.image_batching,
            max_concurrency=rs.max_concurrency,
            timeout=rs.timeout_sec,
            connect_timeout=rs.connect_timeout_sec,
            max_connections=rs.max_connections,
            max_retries=rs.max_retries,
            backoff_base=hs.backoff_base_sec,
            backoff_cap=hs.backoff_cap_sec,
        )


    async def check_connection(self) -> None:
        """Beyond `/models`: learn the vector size, how images may be sent, and that the
        bulk priority is accepted — so a misfit upstream fails startup, not an enroll."""
        await super().check_connection()

        text_vectors = await self._probe(["ping"])
        if text_vectors is None:
            raise ApiError(
                status_code=503,
                message=f"Model {self.model_name} rejected a text embedding request",
            )

        self.dimension = len(text_vectors[0])
        self.batch_images = await self._resolve_image_batching()

        if self.bulk_priority and await self._probe(["ping"], priority=self.bulk_priority) is None:
            raise ApiError(
                status_code=503,
                message=f"Retriever rejected priority={self.bulk_priority}: start it with "
                        f"`--scheduling-policy priority` or set RETRIEVER_BULK_PRIORITY=0",
            )

        log.info(
            "retriever: dimension=%d image_batching=%s bulk_priority=%d",
            self.dimension,
            "batched" if self.batch_images else "single",
            self.bulk_priority,
        )


    async def _resolve_image_batching(self) -> bool:
        """Decide whether images go several per request; require image support at all."""
        image = self._image_conversation(self.PROBE_IMAGE, "image/png")

        if self.image_batching != "off":
            batch_vectors = await self._probe([image, image])

            if batch_vectors is not None:
                self._require_dimension(batch_vectors)

                return True

            if self.image_batching == "on":
                raise ApiError(
                    status_code=503,
                    message=f"Model {self.model_name} does not accept batched image requests "
                            f"(RETRIEVER_IMAGE_BATCHING=on)",
                )

            log.warning("retriever: batched image requests unsupported; sending one image per request")

        single_vectors = await self._probe([image])

        if single_vectors is None:
            raise ApiError(
                status_code=503,
                message=f"Model {self.model_name} does not accept image inputs",
            )

        self._require_dimension(single_vectors)

        return False


    def _require_dimension(self, vectors: list[list[float]]) -> None:
        """Image and text vectors must share one space — the same size."""
        if len(vectors[0]) != self.dimension:
            raise ApiError(
                status_code=503,
                message=f"Model {self.model_name} returns {len(vectors[0])}-d image vectors "
                        f"but {self.dimension}-d text vectors",
            )


    async def _probe(self, units: list, **extra) -> list[list[float]] | None:
        """One probe request: its vectors, or None when the upstream rejects it (or
        returns a vector count that does not match the request)."""
        try:
            response = await self.send(
                "POST",
                self.route,
                json={"model": self.model_name, **self._payload(units), **extra},
            )

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {self.route} during the check",
                details={"error": repr(e)},
            ) from e

        if not response.is_success:
            return None

        vectors = self._json(response, action="the check", parse=self._vectors)

        return vectors if len(vectors) == len(units) else None


    async def embed_texts(
        self,
        texts: list[str],
        *,
        bulk: bool = False,
    ) -> list[list[float]]:
        """Embed texts through the standard `input` field, `batch_size` per request."""
        return await self._embed(texts, batch_size=self.batch_size, bulk=bulk)


    async def embed_images(
        self,
        base64_images: list[str],
        *,
        image_mime: str = "image/jpeg",
        bulk: bool = False,
    ) -> list[list[float]]:
        """Embed base64 images as chat-style conversations; `batch_size` per request
        when the upstream accepts batches, otherwise one per request."""
        conversations = [self._image_conversation(b64, image_mime) for b64 in base64_images]
        batch_size = self.batch_size if self.batch_images else 1

        return await self._embed(conversations, batch_size=batch_size, bulk=bulk)


    async def _embed(
        self,
        units: list,
        *,
        batch_size: int,
        bulk: bool,
    ) -> list[list[float]]:
        """Fire every batch concurrently (bulk ones gated by `_guard`); output order
        matches input order."""
        batches = [units[i:i + batch_size] for i in range(0, len(units), batch_size)]
        results = await gather(self._embed_batch(batch, bulk=bulk) for batch in batches)

        return [vector for vectors in results for vector in vectors]


    async def _embed_batch(self, units: list, *, bulk: bool) -> list[list[float]]:
        """POST one batch. A content rejection of a multi-item batch is retried once as
        single-item requests, isolating the bad item and covering upstreams that do not
        take batches; a rejected single item is an error."""
        body = {"model": self.model_name, **self._payload(units)}

        # Sent only when configured: `priority` is a vLLM extension other APIs reject.
        if bulk and self.bulk_priority:
            body["priority"] = self.bulk_priority

        async with (self._guard() if bulk else nullcontext()):
            response = await self._request(
                "POST",
                self.route,
                action="embedding",
                passthrough_statuses=self.SPLIT_STATUSES,
                json=body,
            )

        if not response.is_success:
            if len(units) == 1:
                raise ApiError(
                    status_code=503,
                    message=f"Upstream {self.route} rejected an item during embedding",
                    details={"upstream_status": response.status_code, "upstream_error": error_body(response)},
                )

            log.warning(
                "retriever: batch of %d rejected (%d); retrying as single-item requests",
                len(units),
                response.status_code,
            )

            return await self._embed(units, batch_size=1, bulk=bulk)

        vectors = self._json(response, action="embedding", parse=self._vectors)

        if len(vectors) != len(units):
            raise ApiError(
                status_code=422,
                message=f"Upstream {self.route} returned {len(vectors)} vectors for {len(units)} inputs",
            )

        return vectors


    @staticmethod
    def _payload(units: list) -> dict:
        """Request fields for a batch: texts via `input`; one image conversation via
        `messages`, several as a batch of conversations."""
        if isinstance(units[0], str):
            return {"input": units}

        return {"messages": units[0] if len(units) == 1 else units}


    @staticmethod
    def _image_conversation(b64: str, image_mime: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{b64}"},
                    }
                ],
            }
        ]


    @staticmethod
    def _vectors(data: dict) -> list[list[float]]:
        embeddings = sorted(data["data"], key=lambda x: x["index"])

        return [emb["embedding"] for emb in embeddings]
