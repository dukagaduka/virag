"""Basic Store Service settings"""

from __future__ import annotations

import os
from dataclasses import dataclass

from qdrant_client.models import Distance


class SettingsError(Exception):
    """Raised when service configuration is invalid."""


_IMAGE_BATCHING_MODES = ("auto", "on", "off")

_DEFAULT_FRAMING_TEXT = (
    "The following image tiles are retrieved document context. "
    "Use them to answer the question."
)


@dataclass(slots=True, frozen=True)
class RagSettings:
    top_n: int
    score_threshold: float | None
    framing_text: str


@dataclass(slots=True, frozen=True)
class QdrantStoreSettings:
    base_url: str
    collection_name: str
    vector_size: int
    distance_type: Distance
    chunk_byte_budget: int
    max_batch_chunks: int


@dataclass(slots=True, frozen=True)
class HttpSettings:
    backoff_base: float
    backoff_cap: float


@dataclass(slots=True, frozen=True)
class RetrieverSettings:
    base_url: str
    model_name: str
    api_key: str
    batch_size: int
    max_concurrency: int
    bulk_priority: int
    image_batching: str
    timeout: int
    connect_timeout: int
    max_connections: int
    max_retries: int


@dataclass(slots=True, frozen=True)
class TilerSettings:
    base_url: str
    max_concurrency: int
    timeout: int
    connect_timeout: int
    max_connections: int
    max_retries: int


@dataclass(slots=True, frozen=True)
class VLMSettings:
    base_url: str
    model_name: str
    api_key: str
    timeout: int
    connect_timeout: int
    max_connections: int
    max_concurrency: int
    max_retries: int


@dataclass(slots=True, frozen=True)
class EnrollSettings:
    batch_size: int


@dataclass(slots=True, frozen=True)
class ReconcileSettings:
    self_heal: bool
    interval_sec: int
    pending_ttl_sec: int


@dataclass(slots=True, frozen=True)
class Settings:
    host: str
    port: int
    db_secret: str

    rag_settings: RagSettings
    http_settings: HttpSettings
    qdrant_settings: QdrantStoreSettings
    retriever_settings: RetrieverSettings
    tiler_settings: TilerSettings
    vlm_settings: VLMSettings
    enroll_settings: EnrollSettings
    reconcile_settings: ReconcileSettings

    @classmethod
    def from_env(cls) -> Settings:
        service_host = os.environ.get("VIRAG_HOST", "127.0.0.1")

        service_port = int_coercion(
            os.environ.get("VIRAG_PORT", "8080"),
            "VIRAG_PORT",
            low=1,
            high=65535,
        )

        # Single binding secret shared with the tiler (and stamped on the Qdrant
        # collection); read from TILER_DB_SECRET so all three agree on one value.
        service_db_secret = os.environ.get("TILER_DB_SECRET", "")

        rag_top_n = int_coercion(
            os.environ.get("VIRAG_TOP_N", "5"),
            "VIRAG_TOP_N",
            low=1,
        )

        rag_score_threshold_raw = os.environ.get("VIRAG_SCORE_THRESHOLD")
        if rag_score_threshold_raw in (None, ""):
            rag_score_threshold = None
        else:
            try:
                rag_score_threshold = float(rag_score_threshold_raw)
            except ValueError as e:
                raise SettingsError("`VIRAG_SCORE_THRESHOLD` must be a float.") from e

        rag_framing_text = os.environ.get("VIRAG_FRAMING_TEXT") or _DEFAULT_FRAMING_TEXT

        # -----------------------------
        # HTTP (shared upstream retry policy)
        # -----------------------------

        http_backoff_base = float_coercion(
            os.environ.get("HTTP_RETRY_BACKOFF_BASE_SEC", "0.25"),
            "HTTP_RETRY_BACKOFF_BASE_SEC",
            low=0,
        )

        http_backoff_cap = float_coercion(
            os.environ.get("HTTP_RETRY_BACKOFF_CAP_SEC", "8"),
            "HTTP_RETRY_BACKOFF_CAP_SEC",
            low=0,
        )

        # -----------------------------
        # QDRANT
        # -----------------------------

        qdrant_base_url = os.environ.get(
            "QDRANT_BASE_URL",
            "http://qdrant-service:6333",
        )

        qdrant_collection_name = os.environ.get(
            "QDRANT_COLLECTION_NAME",
            "store-collection",
        )

        qdrant_vector_size = int_coercion(
            require_env("QDRANT_VECTOR_SIZE"),
            "QDRANT_VECTOR_SIZE",
            low=1,
        )

        qdrant_distance_type_value = os.environ.get(
            "QDRANT_DISTANCE_TYPE",
            "Cosine",
        )

        try:
            qdrant_distance_type = Distance(qdrant_distance_type_value)
        except ValueError as e:
            raise SettingsError(
                f"Incorrect `QDRANT_DISTANCE_TYPE`: "
                f"{qdrant_distance_type_value!r}. "
                f"Must be one of {[t.value for t in Distance]}."
            ) from e

        qdrant_chunk_size_mb = int_coercion(
            os.environ.get("QDRANT_CHUNK_SIZE", "24"),
            "QDRANT_CHUNK_SIZE",
            low=1,
        )

        qdrant_chunk_byte_budget = qdrant_chunk_size_mb * 1024 * 1024

        qdrant_max_batch_chunks = int_coercion(
            os.environ.get("QDRANT_MAX_BATCH_CHUNKS", "1024"),
            "QDRANT_MAX_BATCH_CHUNKS",
            low=1,
        )


        # -----------------------------
        # RETRIEVER
        # -----------------------------

        retriever_base_url = os.environ.get(
            "RETRIEVER_BASE_URL",
            "http://retriever-vllm-virag:8000/v1"
        )

        retriever_model_name = os.environ.get(
            "RETRIEVER_MODEL_NAME",
            "retriever-model"
        )

        retriever_api_key = os.environ.get(
            "RETRIEVER_API_KEY",
            "dummy"
        )

        retriever_batch_size = int_coercion(
            os.environ.get("RETRIEVER_BATCH_SIZE", "32"),
            "RETRIEVER_BATCH_SIZE",
            low=1
        )

        # Caps bulk (ingest) calls only; interactive queries bypass it.
        retriever_max_concurrency = int_coercion(
            os.environ.get("RETRIEVER_MAX_CONCURRENCY", "32"),
            "RETRIEVER_MAX_CONCURRENCY",
            low=0
        )

        # 0 = never send `priority` (portable); > 0 needs a vLLM retriever started
        # with `--scheduling-policy priority` (lower value = served earlier).
        retriever_bulk_priority = int_coercion(
            os.environ.get("RETRIEVER_BULK_PRIORITY", "0"),
            "RETRIEVER_BULK_PRIORITY",
            low=0
        )

        retriever_image_batching = os.environ.get("RETRIEVER_IMAGE_BATCHING", "auto").strip().lower()

        if retriever_image_batching not in _IMAGE_BATCHING_MODES:
            raise SettingsError(
                f"Incorrect `RETRIEVER_IMAGE_BATCHING`: {retriever_image_batching!r}. "
                f"Must be one of {list(_IMAGE_BATCHING_MODES)}."
            )

        retriever_timeout = int_coercion(
            os.environ.get("RETRIEVER_TIMEOUT_SEC", "60"),
            "RETRIEVER_TIMEOUT_SEC",
            low=1
        )

        retriever_connect_timeout = int_coercion(
            os.environ.get("RETRIEVER_CONNECT_TIMEOUT_SEC", "10"),
            "RETRIEVER_CONNECT_TIMEOUT_SEC",
            low=1
        )

        retriever_max_connections = int_coercion(
            os.environ.get("RETRIEVER_MAX_CONNECTIONS", "100"),
            "RETRIEVER_MAX_CONNECTIONS",
            low=1
        )

        retriever_max_retries = int_coercion(
            os.environ.get("RETRIEVER_MAX_RETRIES", "3"),
            "RETRIEVER_MAX_RETRIES",
            low=0
        )

        concurrency_within_pool(
            retriever_max_concurrency,
            retriever_max_connections,
            "RETRIEVER",
        )

        # -----------------------------
        # TILER
        # -----------------------------

        tiler_base_url = os.environ.get(
            "TILER_BASE_URL",
            "http://tiler-virag:8080"
        )

        tiler_timeout = int_coercion(
            os.environ.get("TILER_TIMEOUT_SEC", "120"),
            "TILER_TIMEOUT_SEC",
            low=1
        )

        tiler_connect_timeout = int_coercion(
            os.environ.get("TILER_CONNECT_TIMEOUT_SEC", "10"),
            "TILER_CONNECT_TIMEOUT_SEC",
            low=1
        )

        tiler_max_connections = int_coercion(
            os.environ.get("TILER_MAX_CONNECTIONS", "100"),
            "TILER_MAX_CONNECTIONS",
            low=1
        )

        tiler_max_retries = int_coercion(
            os.environ.get("TILER_MAX_RETRIES", "3"),
            "TILER_MAX_RETRIES",
            low=0
        )

        # Caps bulk tile regeneration (self-heal, revision) only; chat RAG bypasses it.
        tiler_max_concurrency = int_coercion(
            os.environ.get("TILER_MAX_CONCURRENCY", "32"),
            "TILER_MAX_CONCURRENCY",
            low=0
        )

        concurrency_within_pool(
            tiler_max_concurrency,
            tiler_max_connections,
            "TILER",
        )

        # -----------------------------
        # VLM (upstream, OpenAI-compatible)
        # -----------------------------

        vlm_base_url = require_env("VLM_BASE_URL")

        vlm_model_name = require_env("VLM_MODEL_NAME")

        vlm_api_key = os.environ.get(
            "VLM_API_KEY",
            "dummy"
        )

        vlm_timeout = int_coercion(
            os.environ.get("VLM_TIMEOUT_SEC", "480"),
            "VLM_TIMEOUT_SEC",
            low=1
        )

        vlm_connect_timeout = int_coercion(
            os.environ.get("VLM_CONNECT_TIMEOUT_SEC", "10"),
            "VLM_CONNECT_TIMEOUT_SEC",
            low=1
        )

        vlm_max_connections = int_coercion(
            os.environ.get("VLM_MAX_CONNECTIONS", "100"),
            "VLM_MAX_CONNECTIONS",
            low=1
        )

        vlm_max_concurrency = int_coercion(
            os.environ.get("VLM_MAX_CONCURRENCY", "0"),
            "VLM_MAX_CONCURRENCY",
            low=0,
        )

        concurrency_within_pool(
            vlm_max_concurrency,
            vlm_max_connections,
            "VLM",
        )

        vlm_max_retries = int_coercion(
            os.environ.get("VLM_MAX_RETRIES", "3"),
            "VLM_MAX_RETRIES",
            low=0,
        )

        # -----------------------------
        # ENROLL
        # -----------------------------

        # Tiles gathered per embed+upsert step (enroll, self-heal, revision).
        enroll_batch_size = int_coercion(
            os.environ.get("VIRAG_ENROLL_BATCH_SIZE", "32"),
            "VIRAG_ENROLL_BATCH_SIZE",
            low=1,
        )

        # -----------------------------
        # RECONCILE (enroll self-repair)
        # -----------------------------

        reconcile_self_heal = bool_coercion(
            os.environ.get("VIRAG_SELF_HEAL", "true"),
            "VIRAG_SELF_HEAL",
        )

        reconcile_interval_sec = int_coercion(
            os.environ.get("VIRAG_RECONCILE_INTERVAL_SEC", "300"),
            "VIRAG_RECONCILE_INTERVAL_SEC",
            low=0,
        )

        reconcile_pending_ttl_sec = int_coercion(
            os.environ.get("VIRAG_PENDING_TTL_SEC", "3600"),
            "VIRAG_PENDING_TTL_SEC",
            low=1,
        )


        return cls(
            host=service_host,
            port=service_port,
            db_secret=service_db_secret,
            rag_settings=RagSettings(
                top_n=rag_top_n,
                score_threshold=rag_score_threshold,
                framing_text=rag_framing_text,
            ),
            http_settings=HttpSettings(
                backoff_base=http_backoff_base,
                backoff_cap=http_backoff_cap,
            ),
            qdrant_settings=QdrantStoreSettings(
                base_url=qdrant_base_url,
                collection_name=qdrant_collection_name,
                vector_size=qdrant_vector_size,
                distance_type=qdrant_distance_type,
                chunk_byte_budget=qdrant_chunk_byte_budget,
                max_batch_chunks=qdrant_max_batch_chunks,
            ),
            retriever_settings=RetrieverSettings(
                base_url=retriever_base_url,
                model_name=retriever_model_name,
                api_key=retriever_api_key,
                batch_size=retriever_batch_size,
                max_concurrency=retriever_max_concurrency,
                bulk_priority=retriever_bulk_priority,
                image_batching=retriever_image_batching,
                timeout=retriever_timeout,
                connect_timeout=retriever_connect_timeout,
                max_connections=retriever_max_connections,
                max_retries=retriever_max_retries,
            ),
            tiler_settings=TilerSettings(
                base_url=tiler_base_url,
                max_concurrency=tiler_max_concurrency,
                timeout=tiler_timeout,
                connect_timeout=tiler_connect_timeout,
                max_connections=tiler_max_connections,
                max_retries=tiler_max_retries,
            ),
            vlm_settings=VLMSettings(
                base_url=vlm_base_url,
                model_name=vlm_model_name,
                api_key=vlm_api_key,
                timeout=vlm_timeout,
                connect_timeout=vlm_connect_timeout,
                max_connections=vlm_max_connections,
                max_concurrency=vlm_max_concurrency,
                max_retries=vlm_max_retries,
            ),
            enroll_settings=EnrollSettings(
                batch_size=enroll_batch_size,
            ),
            reconcile_settings=ReconcileSettings(
                self_heal=reconcile_self_heal,
                interval_sec=reconcile_interval_sec,
                pending_ttl_sec=reconcile_pending_ttl_sec,
            ),
        )


def require_env(name: str) -> str:
    """Return the value of a required environment variable."""
    value = os.environ.get(name)

    if value is None:
        raise SettingsError(f"`{name}` environment variable is required.")

    return value


def concurrency_within_pool(max_concurrency: int, max_connections: int, prefix: str) -> None:
    """Above the pool size a concurrency cap never binds — the pool would queue instead."""
    if max_concurrency > max_connections:
        raise SettingsError(
            f"`{prefix}_MAX_CONCURRENCY` ({max_concurrency}) must not exceed "
            f"`{prefix}_MAX_CONNECTIONS` ({max_connections})."
        )


def bool_coercion(value: str, name: str) -> bool:
    """Convert a string to a boolean, accepting the usual truthy/falsy spellings."""
    normalized = value.strip().lower()

    if normalized in ("1", "true", "yes", "on"):
        return True

    if normalized in ("0", "false", "no", "off"):
        return False

    raise SettingsError(f"`{name}` must be a boolean (got {value!r}).")


def float_coercion(
    value: str,
    name: str,
    low: float | None = None,
) -> float:
    """Convert a string to a float and validate its lower bound."""
    try:
        coerced_value = float(value)
    except ValueError as e:
        raise SettingsError(f"`{name}` must be a float.") from e

    if low is not None and coerced_value < low:
        raise SettingsError(
            f"`{name}` must be greater than or equal to {low}. "
            f"Got {coerced_value}."
        )

    return coerced_value


def int_coercion(
    value: str,
    name: str,
    low: int | None = None,
    high: int | None = None,
) -> int:
    """Convert a string to an integer and validate its bounds."""
    try:
        coerced_value = int(value)
    except ValueError as e:
        raise SettingsError(f"`{name}` must be an integer.") from e

    if low is not None and coerced_value < low:
        raise SettingsError(
            f"`{name}` must be greater than or equal to {low}. "
            f"Got {coerced_value}."
        )

    if high is not None and coerced_value > high:
        raise SettingsError(
            f"`{name}` must be less than or equal to {high}. "
            f"Got {coerced_value}."
        )

    return coerced_value