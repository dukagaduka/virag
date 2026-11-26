"""RAG: the tiles a question is answered from.

A chat request that carries a `rag` field is augmented before it goes to the VLM:

  1. The query — the text of the last user message — is embedded, and the closest
     points are found in Qdrant (`top_k`, `retriever_threshold`).
  2. Each hit's tile is fetched from the tiler and checked against the hash it was
     embedded from. A tile that fails the check, or one the tiler refuses, is skipped;
     a tile rendered by another render_version, or a tiler that is down, fails the
     request.
  3. With a reranker, the tiles are scored against the query and the best `top_n` at or
     above `reranker_threshold` are kept. A failing reranker never fails the request:
     the retriever's order, cut to `top_n`, is used instead.
  4. The tiles are put into the last user message, after the framing text.

`RagConfig` is the `rag` field of one request, validated, with what it leaves out taken
from the settings. `Rag` holds the backends and runs the steps.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, replace

from virag.clients.qdrant_store import Hit, QdrantStore
from virag.clients.reranker_client import RerankerClient
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import TilerClient
from virag.errors import ApiError
from virag.settings import RagSettings, RerankerSettings, RetrieverSettings
from virag.utils import gather


log = logging.getLogger("virag.rag")


@dataclass(frozen=True, slots=True)
class RagConfig:
    """How one request wants its RAG done."""

    top_k: int
    retriever_threshold: float | None
    top_n: int
    reranker_threshold: float | None
    framing_text: str
    include_sources: bool


    @classmethod
    def from_request(
        cls,
        raw,
        rag: RagSettings,
        retriever: RetrieverSettings,
        reranker: RerankerSettings,
    ) -> RagConfig | None:
        """The `rag` field of a request as a config, or None when RAG is off; what the
        request leaves out comes from the `rag`, `retriever` and `reranker` settings. The
        reranker's fields are validated always, but bound to `top_k` only while it is on."""
        if raw is None or raw is False:
            return None

        if raw is True:
            raw = {}

        if not isinstance(raw, dict):
            raise ApiError(
                status_code=422,
                message="`rag` must be an object or a boolean",
            )

        top_k = cls._positive_int(raw, "top_k", retriever.top_k)
        retriever_threshold = cls._number_or_null(raw, "retriever_threshold", retriever.threshold)
        top_n = cls._positive_int(raw, "top_n", reranker.top_n)
        reranker_threshold = cls._number_or_null(raw, "reranker_threshold", reranker.threshold)

        if reranker_threshold is not None and not 0 <= reranker_threshold <= 1:
            raise ApiError(
                status_code=422,
                message="`rag.reranker_threshold` must be within [0, 1]",
            )

        if reranker.enabled and top_n > top_k:
            raise ApiError(
                status_code=422,
                message=f"`rag.top_n` ({top_n}) must not exceed `rag.top_k` ({top_k})",
            )

        framing_text = raw.get("framing_text", rag.framing_text)
        if not isinstance(framing_text, str):
            raise ApiError(
                status_code=422,
                message="`rag.framing_text` must be a string",
            )

        return cls(
            top_k=top_k,
            retriever_threshold=retriever_threshold,
            top_n=top_n,
            reranker_threshold=reranker_threshold,
            framing_text=framing_text,
            include_sources=bool(raw.get("include_sources", False)),
        )


    @staticmethod
    def _positive_int(raw: dict, key: str, default: int) -> int:
        value = raw.get(key, default)

        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ApiError(
                status_code=422,
                message=f"`rag.{key}` must be a positive integer",
            )

        return value


    @staticmethod
    def _number_or_null(raw: dict, key: str, default: float | None) -> float | None:
        value = raw.get(key, default)

        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ApiError(
                status_code=422,
                message=f"`rag.{key}` must be a number or null",
            )

        return value


@dataclass(frozen=True, slots=True)
class RetrievedTile:
    """A tile found for a query: the hit, the tile itself (JPEG in base64), and the
    reranker's score once it has one."""

    hit: Hit
    image_b64: str
    rerank_score: float | None = None


    def source(self, name: str) -> dict:
        """The tile as an entry of the response's `sources`."""
        source = {
            "name": name,
            "doc_id": self.hit.point.doc_id,
            "tile_num": self.hit.point.tile_num,
            "score": self.hit.score,
        }

        # Present only for a tile the reranker scored.
        if self.rerank_score is not None:
            source["rerank_score"] = self.rerank_score

        return source


class Rag:
    """Finds the tiles a question is answered from and puts them into the request."""

    def __init__(
        self,
        store: QdrantStore,
        tiler: TilerClient,
        retriever: RetrieverClient,
        reranker: RerankerClient | None,
        rag_settings: RagSettings,
        retriever_settings: RetrieverSettings,
        reranker_settings: RerankerSettings,
    ):

        self.store     = store
        self.tiler     = tiler
        self.retriever = retriever
        self.reranker  = reranker

        self.rag_settings       = rag_settings
        self.retriever_settings = retriever_settings
        self.reranker_settings  = reranker_settings


    def config(self, raw) -> RagConfig | None:
        """The `rag` field of a request as a config; None when the request has RAG off."""
        return RagConfig.from_request(raw, self.rag_settings, self.retriever_settings, self.reranker_settings)


    async def augment(self, messages: list, config: RagConfig) -> list[dict] | None:
        """Retrieve tiles for the query, rerank them when a reranker is set, inject them
        into the last user message, and return the sources list when the caller asked for
        it (else None)."""
        query = self.query_of(messages)

        if not query:
            raise ApiError(
                status_code=422,
                message="RAG is enabled but no user text was found to build the query",
            )

        log.info(
            "rag query: top_k=%d retriever_threshold=%s preview=%r",
            config.top_k, config.retriever_threshold, query[:120],
        )

        vector = (await self.retriever.embed_texts([query]))[0]
        hits = await self.store.query(vector, top_k=config.top_k, score_threshold=config.retriever_threshold)

        log.info("rag retrieved: %d hits", len(hits))

        tiles = await self._fetch_tiles(hits)

        if self.reranker is not None:
            tiles = await self._rerank(query, tiles, config)

        if tiles:
            self.inject(messages, [tile.image_b64 for tile in tiles], config.framing_text)
            log.info("rag injected: %d tiles", len(tiles))
        else:
            log.info("rag injected: 0 tiles")

        if config.include_sources:
            names = await self._doc_names({tile.hit.point.doc_id for tile in tiles})

            return [tile.source(names[tile.hit.point.doc_id]) for tile in tiles]

        return None


    # ------------------------------------------------------------------ #
    # the request's messages
    # ------------------------------------------------------------------ #
    @staticmethod
    def query_of(messages: list) -> str:
        """Text of the last user message (concatenated text parts of multimodal content)."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content")

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]

                return " ".join(text for text in texts if text).strip()

            return ""

        return ""


    @classmethod
    def inject(cls, messages: list, images_b64: list[str], framing_text: str) -> None:
        """Prepend framing text + retrieved tile images to the last user message."""
        index = cls._last_user_index(messages)

        image_parts = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in images_b64
        ]

        original = messages[index].get("content")
        new_content = [{"type": "text", "text": framing_text}, *image_parts]

        if isinstance(original, str):
            new_content.append({"type": "text", "text": original})
        elif isinstance(original, list):
            new_content.extend(original)

        messages[index] = {**messages[index], "content": new_content}


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    async def _fetch_tiles(self, hits: list[Hit]) -> list[RetrievedTile]:
        """Fetch each hit's tile, skipping any the tiler refuses or that fails its
        integrity check. The first failure of the whole request cancels the other
        fetches."""
        tiles = await gather(self._fetch_tile(hit) for hit in hits)

        return [tile for tile in tiles if tile is not None]


    async def _fetch_tile(self, hit: Hit) -> RetrievedTile | None:
        """One hit's tile, or None for a tile to skip.

        Two failures are the whole request's, never quietly answered without tiles: a tile
        the tiler now renders by another render_version than the one it was embedded from
        (409 RENDER_VERSION_MISMATCH — the collection is stale and must be rebuilt), and a
        tiler that is unreachable or not serving (503 NOT_READY).
        """
        doc_id, tile_num = hit.point.doc_id, hit.point.tile_num
        render_version = hit.point.render_version or None

        try:
            data = await self.tiler.get_tile(doc_id, tile_num, render_version=render_version)

        except ApiError as e:
            if e.code == self.tiler.RENDER_VERSION_MISMATCH:
                raise

            if self.tiler.unavailable(e):
                raise ApiError(status_code=503, message=f"The tiler is not ready: {e.message}", details=e.details) from e

            log.warning("skipping tile (doc_id=%s tile_num=%s): %s", doc_id, tile_num, e.message)

            return None

        # Integrity: the fetched tile must be the exact bytes enrolled in Qdrant; a
        # point without a hash cannot be checked, so it is not trusted either.
        expected_hash = hit.point.image_hash
        if expected_hash is None:
            log.warning("integrity: tile (doc_id=%s tile_num=%s) has no image_hash in qdrant; skipping", doc_id, tile_num)

            return None

        if hashlib.sha256(data).hexdigest() != expected_hash:
            log.warning(
                "integrity: tile (doc_id=%s tile_num=%s) hash mismatch (qdrant=%s); skipping",
                doc_id, tile_num, expected_hash,
            )

            return None

        return RetrievedTile(hit, base64.b64encode(data).decode())


    async def _rerank(self, query: str, tiles: list[RetrievedTile], config: RagConfig) -> list[RetrievedTile]:
        """Score the candidate tiles against the query; keep those at or above the
        threshold, best first, at most `top_n`. A failing reranker never fails the request:
        the retriever's order, cut to `top_n`, is used instead."""
        if not tiles:
            return []

        try:
            scores = await self.reranker.score(query, [tile.image_b64 for tile in tiles])  # type: ignore[union-attr]

        except ApiError as e:
            # A refusal is a contract mismatch to fix; an outage is just waited out.
            level = logging.ERROR if self.reranker.rejected(e) else logging.WARNING  # type: ignore[union-attr]
            log.log(level, "rag rerank failed, using the retriever's order: %s | details=%s", e.message, e.details)

            return tiles[:config.top_n]

        threshold = config.reranker_threshold
        ranked = sorted(
            (replace(tile, rerank_score=score) for tile, score in zip(tiles, scores) if threshold is None or score >= threshold),
            key=lambda tile: tile.rerank_score,
            reverse=True,
        )

        log.info(
            "rag reranked: %d of %d kept (top_n=%d reranker_threshold=%s)",
            min(len(ranked), config.top_n), len(tiles), config.top_n, threshold,
        )

        return ranked[:config.top_n]


    async def _doc_names(self, doc_ids: set[str]) -> dict[str, str]:
        """Each document's `metadata.name` as the tiler holds it now (points carry no
        metadata); a document without one, or one the tiler would not answer for, goes by
        its doc_id."""
        async def name(doc_id: str) -> str:
            try:
                metadata = await self.tiler.get_metadata(doc_id)

            except ApiError as e:
                log.warning("no name for doc_id=%s: %s", doc_id, e.message)

                return doc_id

            return metadata.get("name") or doc_id

        ordered = sorted(doc_ids)

        return dict(zip(ordered, await gather(name(doc_id) for doc_id in ordered)))


    @staticmethod
    def _last_user_index(messages: list) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index

        return len(messages) - 1
