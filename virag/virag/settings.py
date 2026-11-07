"""Service settings: every environment variable, its default and its bounds — declared once.

Each section is a pydantic-settings model read from its own env prefix, and a field's
name is the variable's suffix (`RetrieverSettings.batch_size` ← `RETRIEVER_BATCH_SIZE`),
so the declarations below double as the reference. An empty variable counts as unset.
The few irregular names are spelled out with `validation_alias`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client.models import Distance


class SettingsError(Exception):
    """Raised when service configuration is invalid."""


_DEFAULT_FRAMING_TEXT = (
    "The following image tiles are retrieved document context. "
    "Use them to answer the question."
)


class _Section(BaseSettings):
    """Shared config: immutable, empty variables count as unset, aliased fields can
    still be set by name (tests, code)."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_ignore_empty=True,
        extra="ignore",
        validate_by_name=True,
        protected_namespaces=(),
    )


class _UpstreamSettings(_Section):
    """Transport knobs every HTTP upstream shares: pool, deadlines, retries, concurrency cap."""

    base_url: str
    timeout_sec: int = Field(ge=1)              # read/write/pool deadline: max silence between bytes
    connect_timeout_sec: int = Field(10, ge=1)
    max_connections: int = Field(100, ge=1)     # pooled keep-alive connections
    max_concurrency: int = Field(0, ge=0)       # 0 = unlimited
    max_retries: int = Field(3, ge=0)           # transient failures only

    @model_validator(mode="after")
    def _concurrency_within_pool(self) -> _UpstreamSettings:
        # Above the pool size a concurrency cap never binds — the pool would queue instead.
        if self.max_concurrency > self.max_connections:
            prefix = self.model_config.get("env_prefix", "")

            raise ValueError(
                f"`{prefix}MAX_CONCURRENCY` ({self.max_concurrency}) must not exceed "
                f"`{prefix}MAX_CONNECTIONS` ({self.max_connections})."
            )

        return self


class RagSettings(_Section):
    """How retrieved tiles are framed in the prompt; a request's own `rag` object overrides it.
    How many tiles are retrieved, and how close they must be, is set in `RetrieverSettings`."""

    model_config = SettingsConfigDict(env_prefix="VIRAG_")

    framing_text: str = _DEFAULT_FRAMING_TEXT


class HttpSettings(_Section):
    """Retry backoff shared by every upstream: full jitter, uniform over
    [0, min(cap, base * 2^attempt)]; a numeric Retry-After wins, still capped."""

    model_config = SettingsConfigDict(env_prefix="HTTP_RETRY_")

    backoff_base_sec: float = Field(0.25, ge=0)
    backoff_cap_sec: float = Field(8.0, ge=0)


class QdrantStoreSettings(_Section):
    """Qdrant: the data path speaks gRPC to the host of `base_url`; REST (the URL's port)
    still serves the /qdrant proxy and the client's version check."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    base_url: str = "http://qdrant-service:6333"
    grpc_port: int = Field(6334, ge=1, le=65535)
    timeout_sec: int = Field(60, ge=1)          # per-call deadline
    max_retries: int = Field(3, ge=0, le=4)     # gRPC caps a call at 5 attempts in total
    collection_name: str = "store-collection"   # the serving alias
    vector_size: int = Field(ge=1)
    distance_type: Distance = Distance.COSINE
    chunk_size: int = Field(24, ge=1)           # per-upsert budget, MB
    max_batch_chunks: int = Field(1024, ge=1)   # hard cap on points per upsert

    @property
    def chunk_byte_budget(self) -> int:
        return self.chunk_size * 1024 * 1024


class ValkeySettings(_Section):
    """Valkey: where virag processes coordinate (leases, writers, the revision lock)."""

    model_config = SettingsConfigDict(env_prefix="VALKEY_")

    url: str                                    # redis://[:password@]host:port/db
    prefix: str = "virag"                       # key namespace; one per deployment sharing a Valkey
    timeout_sec: float = Field(2.0, gt=0)       # connect and per-command deadline
    max_connections: int = Field(64, ge=1)


class RetrieverSettings(_UpstreamSettings):
    """The retrieval embedding model behind an OpenAI-compatible `/embeddings` API, and
    what a RAG query takes from Qdrant (a request's `rag` object overrides `top_k` and
    `threshold`)."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVER_")

    base_url: str = "http://retriever-vllm-virag:8000/v1"
    model_name: str = "retriever-model"
    api_key: str = "dummy"
    timeout_sec: int = Field(60, ge=1)
    batch_size: int = Field(32, ge=1)           # items per request
    max_concurrency: int = Field(32, ge=0)      # bulk (ingest) calls only; queries bypass it
    bulk_priority: int = Field(0, ge=0)         # 0 = never send vLLM `priority`
    image_batching: Literal["auto", "on", "off"] = "auto"
    top_k: int = Field(5, ge=1)                 # candidates per RAG query
    threshold: float | None = None              # min Qdrant score of a candidate; unset = no threshold


class RerankerSettings(_UpstreamSettings):
    """The optional reranker that picks the tiles for the VLM out of the retriever's
    candidates; off while `base_url` is unset (a request's `rag` object overrides
    `top_n` and `threshold`)."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    base_url: str | None = None                 # the server root, without /v1; unset = no reranker
    model_name: str = "reranker-model"
    api_key: str = "dummy"
    api: Literal["score", "rerank"] = "score"   # vLLM `/score` | Cohere/Jina-style `/v1/rerank`
    timeout_sec: int = Field(30, ge=1)
    batch_size: int = Field(8, ge=1)            # tiles per request
    top_n: int = Field(5, ge=1)                 # tiles the VLM gets; at most RETRIEVER_TOP_K
    threshold: float | None = Field(None, ge=0, le=1)  # min relevance of a tile; unset = no threshold

    @property
    def enabled(self) -> bool:
        return self.base_url is not None


class TilerSettings(_UpstreamSettings):
    model_config = SettingsConfigDict(env_prefix="TILER_")

    base_url: str = "http://tiler-virag:8080"
    timeout_sec: int = Field(120, ge=1)         # for the tile stream: the max gap between events
    max_concurrency: int = Field(32, ge=0)      # bulk tile regeneration only; chat RAG bypasses it


class VLMSettings(_UpstreamSettings):
    """The upstream OpenAI-compatible vision chat model (external to the deployment)."""

    model_config = SettingsConfigDict(env_prefix="VLM_")

    base_url: str
    model_name: str
    api_key: str = "dummy"
    timeout_sec: int = Field(480, ge=1)         # covers a long prefill on big pixel-RAG prompts


class EnrollSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="VIRAG_ENROLL_")

    batch_size: int = Field(32, ge=1)           # tiles per embed+upsert step


class ReconcileSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="VIRAG_")

    self_heal: bool = True
    interval_sec: int = Field(300, ge=0, validation_alias="VIRAG_RECONCILE_INTERVAL_SEC")  # 0 = startup only
    pending_ttl_sec: int = Field(3600, ge=1)    # must exceed the longest enroll


class RevisionSettings(_Section):
    """What virag does when it starts against a collection embedded from another
    render_version than the tiler's."""

    model_config = SettingsConfigDict(env_prefix="VIRAG_")

    revision_on_render_mismatch: bool = False   # on → revise at startup; off → log, wait for POST /revision


class CoordinationSettings(_Section):
    model_config = SettingsConfigDict(env_prefix="VIRAG_")

    process_ttl_sec: int = Field(30, ge=3)      # a process silent this long counts as dead


class Settings(_Section):
    model_config = SettingsConfigDict(env_prefix="VIRAG_")

    host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    # Single binding secret shared with the tiler (and stamped on the Qdrant collection);
    # read from TILER_DB_SECRET so all three agree on one value.
    db_secret: str = Field("", validation_alias="TILER_DB_SECRET")

    rag_settings: RagSettings = Field(default_factory=RagSettings)
    http_settings: HttpSettings = Field(default_factory=HttpSettings)
    qdrant_settings: QdrantStoreSettings = Field(default_factory=QdrantStoreSettings)  # type: ignore[arg-type]
    valkey_settings: ValkeySettings = Field(default_factory=ValkeySettings)  # type: ignore[arg-type]
    retriever_settings: RetrieverSettings = Field(default_factory=RetrieverSettings)
    reranker_settings: RerankerSettings = Field(default_factory=RerankerSettings)
    tiler_settings: TilerSettings = Field(default_factory=TilerSettings)
    vlm_settings: VLMSettings = Field(default_factory=VLMSettings)  # type: ignore[arg-type]
    enroll_settings: EnrollSettings = Field(default_factory=EnrollSettings)
    reconcile_settings: ReconcileSettings = Field(default_factory=ReconcileSettings)
    revision_settings: RevisionSettings = Field(default_factory=RevisionSettings)
    coordination_settings: CoordinationSettings = Field(default_factory=CoordinationSettings)

    @model_validator(mode="after")
    def _reranker_within_candidates(self) -> Settings:
        # The reranker picks its tiles out of the retriever's candidates.
        reranker, retriever = self.reranker_settings, self.retriever_settings

        if reranker.enabled and reranker.top_n > retriever.top_k:
            raise ValueError(
                f"`RERANKER_TOP_N` ({reranker.top_n}) must not exceed `RETRIEVER_TOP_K` ({retriever.top_k})."
            )

        return self


    @classmethod
    def from_env(cls) -> Settings:
        """Read every section from the environment; any invalid or missing variable
        becomes one SettingsError naming the variables at fault."""
        try:
            return cls()

        except ValidationError as e:
            raise SettingsError(cls._describe(e)) from e


    @staticmethod
    def _describe(error: ValidationError) -> str:
        """Render a validation error in terms of environment variable names."""
        model = _SECTIONS.get(error.title)
        problems = []

        for item in error.errors():
            if item["loc"] and model is not None:
                problems.append(f"`{Settings._env_name(model, str(item['loc'][0]))}`: {item['msg']}")
            else:
                problems.append(item["msg"].removeprefix("Value error, "))

        return "; ".join(problems)


    @staticmethod
    def _env_name(model: type[BaseSettings], loc: str) -> str:
        """The variable behind an error location — a field name, or already an alias."""
        info = model.model_fields.get(loc)
        if info is None:
            return loc

        if isinstance(info.validation_alias, str):
            return info.validation_alias

        return f"{model.model_config.get('env_prefix', '')}{loc}".upper()


_SECTIONS: dict[str, type[BaseSettings]] = {
    section.__name__: section
    for section in (
        Settings,
        RagSettings,
        HttpSettings,
        QdrantStoreSettings,
        ValkeySettings,
        RetrieverSettings,
        RerankerSettings,
        TilerSettings,
        VLMSettings,
        EnrollSettings,
        ReconcileSettings,
        RevisionSettings,
        CoordinationSettings,
    )
}
