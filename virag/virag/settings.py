"""Basic Store Service settings"""

from __future__ import annotations

import os
from dataclasses import dataclass

from qdrant_client.models import Distance


class SettingsError(Exception):
    """Raised when service configuration is invalid."""


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
class EmbeddingsSettings:
    base_url: str
    model_name: str
    api_key: str
    chunk_size: int
    timeout: int


@dataclass(slots=True, frozen=True)
class TilerSettings:
    base_url: str
    timeout: int


@dataclass(slots=True, frozen=True)
class VLMSettings:
    base_url: str
    model_name: str
    api_key: str
    timeout: int


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
    qdrant_settings: QdrantStoreSettings
    embeddings_settings: EmbeddingsSettings
    tiler_settings: TilerSettings
    vlm_settings: VLMSettings
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
        # EMBEDDINGS
        # -----------------------------

        embeddings_base_url = os.environ.get(
            "EMBEDDINGS_BASE_URL",
            "http://embeddings-vllm-virag:8000/v1"
        )

        embeddings_model_name = os.environ.get(
            "VLLM_SERVED_MODEL_NAME",
            "embeddings-model"
        )

        embeddings_api_key = os.environ.get(
            "VLLM_API_KEY",
            "dummy"
        )

        embeddings_chunk_size = int_coercion(
            os.environ.get("EMBEDDINGS_CHUNK_SIZE", "16"),
            "EMBEDDINGS_CHUNK_SIZE",
            low=1
        )

        embeddings_timeout = int_coercion(
            os.environ.get("EMBEDDINGS_TIMEOUT_SEC", "60"),
            "EMBEDDINGS_TIMEOUT_SEC",
            low=1
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
            os.environ.get("VLM_TIMEOUT_SEC", "120"),
            "VLM_TIMEOUT_SEC",
            low=1
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
            qdrant_settings=QdrantStoreSettings(
                base_url=qdrant_base_url,
                collection_name=qdrant_collection_name,
                vector_size=qdrant_vector_size,
                distance_type=qdrant_distance_type,
                chunk_byte_budget=qdrant_chunk_byte_budget,
                max_batch_chunks=qdrant_max_batch_chunks,
            ),
            embeddings_settings=EmbeddingsSettings(
                base_url=embeddings_base_url,
                model_name=embeddings_model_name,
                api_key=embeddings_api_key,
                chunk_size=embeddings_chunk_size,
                timeout=embeddings_timeout
            ),
            tiler_settings=TilerSettings(
                base_url=tiler_base_url,
                timeout=tiler_timeout,
            ),
            vlm_settings=VLMSettings(
                base_url=vlm_base_url,
                model_name=vlm_model_name,
                api_key=vlm_api_key,
                timeout=vlm_timeout,
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


def bool_coercion(value: str, name: str) -> bool:
    """Convert a string to a boolean, accepting the usual truthy/falsy spellings."""
    normalized = value.strip().lower()

    if normalized in ("1", "true", "yes", "on"):
        return True

    if normalized in ("0", "false", "no", "off"):
        return False

    raise SettingsError(f"`{name}` must be a boolean (got {value!r}).")


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