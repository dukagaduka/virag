from __future__ import annotations

import logging

from virag.clients.http_service import OpenAIService
from virag.errors import ApiError
from virag.settings import HttpSettings, RerankerSettings
from virag.utils import gather


log = logging.getLogger(__name__)


class RerankerClient(OpenAIService):
    """A reranker scoring each (query, image) pair on its own: a score does not depend on
    the other images of its request, so K candidates go `batch_size` per request, all at
    once, and their scores still compare.

    Two wire formats (`api`): `score` — vLLM's `/score`, scores only, in input order;
    `rerank` — the Cohere/Jina-style `/v1/rerank` that vLLM, SGLang and others share.
    vLLM's `/v1/rerank` echoes every image back whatever `return_documents` says, so
    against vLLM `score` is the one to use. `base_url` is the server root: neither
    route lives under `/v1` alone.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        *,
        api: str,
        batch_size: int,
        **http_options,
    ):

        super().__init__(base_url, model_name, api_key, **http_options)

        self.api        = api
        self.batch_size = batch_size


    @property
    def models_route(self) -> str:
        return f"{self.base_url}/v1/models"


    @property
    def route(self) -> str:
        return f"{self.base_url}/score" if self.api == "score" else f"{self.base_url}/v1/rerank"


    @classmethod
    def from_settings(cls, rs: RerankerSettings, hs: HttpSettings) -> RerankerClient:
        return cls(
            base_url=rs.base_url.rstrip("/"),   # type: ignore[union-attr]
            model_name=rs.model_name,
            api_key=rs.api_key,
            api=rs.api,
            batch_size=rs.batch_size,
            max_concurrency=rs.max_concurrency,
            timeout=rs.timeout_sec,
            connect_timeout=rs.connect_timeout_sec,
            max_connections=rs.max_connections,
            max_retries=rs.max_retries,
            backoff_base=hs.backoff_base_sec,
            backoff_cap=hs.backoff_cap_sec,
        )


    async def check_connection(self) -> None:
        """Beyond `/models`: one image scored through `api`, so a text-only reranker or
        the wrong wire format shows up at startup."""
        await super().check_connection()
        await self.score("ping", [self.PROBE_IMAGE], image_mime="image/png")

        log.info("reranker: api=%s batch_size=%d", self.api, self.batch_size)


    async def score(
        self,
        query: str,
        base64_images: list[str],
        *,
        image_mime: str = "image/jpeg",
    ) -> list[float]:
        """Relevance of each image to `query`, in input order."""
        batches = [base64_images[i:i + self.batch_size] for i in range(0, len(base64_images), self.batch_size)]
        results = await gather(self._score_batch(query, batch, image_mime) for batch in batches)

        return [score for scores in results for score in scores]


    async def _score_batch(self, query: str, base64_images: list[str], image_mime: str) -> list[float]:
        documents = [self._image_document(b64, image_mime) for b64 in base64_images]

        async with self._guard():
            response = await self._request("POST", self.route, action="reranking", json=self._payload(query, documents))

        parse = self._scores_from_score if self.api == "score" else self._scores_from_rerank
        scores = self._json(response, action="reranking", parse=parse)

        if len(scores) != len(base64_images):
            raise ApiError(
                status_code=422,
                message=f"Upstream {self.route} returned {len(scores)} scores for {len(base64_images)} images",
            )

        return scores


    def _payload(self, query: str, documents: list[dict]) -> dict:
        if self.api == "score":
            return {"model": self.model_name, "queries": query, "documents": documents}

        # No `top_n`: every score comes back, the cut is made over all batches at once.
        return {"model": self.model_name, "query": query, "documents": documents, "return_documents": False}


    @staticmethod
    def rejected(e: ApiError) -> bool:
        """The reranker refused the request or answered something unreadable — a
        contract mismatch to fix, as opposed to a reranker that is down or overloaded."""
        status = (e.details or {}).get("upstream_status")

        return e.status_code == 422 or (status is not None and 400 <= status < 500 and status not in (408, 429))


    @staticmethod
    def _image_document(b64: str, image_mime: str) -> dict:
        return {"content": [{"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{b64}"}}]}


    @staticmethod
    def _scores_from_score(data: dict) -> list[float]:
        return [float(item["score"]) for item in sorted(data["data"], key=lambda item: item["index"])]


    @staticmethod
    def _scores_from_rerank(data: dict) -> list[float]:
        return [float(item["relevance_score"]) for item in sorted(data["results"], key=lambda item: item["index"])]
