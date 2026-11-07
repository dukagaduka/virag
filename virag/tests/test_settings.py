"""Settings.from_env — required vars, coercion, enum and bounds validation."""

from __future__ import annotations

import pytest
from qdrant_client.models import Distance

from virag.settings import Settings, SettingsError, _DEFAULT_FRAMING_TEXT


def _base_env(monkeypatch):
    """The minimum env for a valid Settings: the required vars."""
    for key in list(_ALL_KEYS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "768")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm/v1")
    monkeypatch.setenv("VLM_MODEL_NAME", "my-vlm")
    monkeypatch.setenv("VALKEY_URL", "redis://valkey:6379/0")


_ALL_KEYS = {
    "VIRAG_SELF_HEAL", "VIRAG_RECONCILE_INTERVAL_SEC", "VIRAG_PENDING_TTL_SEC",
    "QDRANT_GRPC_PORT", "QDRANT_TIMEOUT_SEC", "QDRANT_MAX_RETRIES", "QDRANT_CHUNK_SIZE", "QDRANT_MAX_BATCH_CHUNKS",
    "VIRAG_HOST", "VIRAG_PORT", "TILER_DB_SECRET", "RETRIEVER_TOP_K", "RETRIEVER_THRESHOLD",
    "VIRAG_FRAMING_TEXT", "QDRANT_BASE_URL", "QDRANT_COLLECTION_NAME", "QDRANT_VECTOR_SIZE",
    "QDRANT_DISTANCE_TYPE", "RETRIEVER_BASE_URL", "RETRIEVER_MODEL_NAME", "RETRIEVER_API_KEY",
    "RETRIEVER_BATCH_SIZE", "RETRIEVER_MAX_CONCURRENCY", "RETRIEVER_BULK_PRIORITY",
    "RETRIEVER_IMAGE_BATCHING", "VIRAG_ENROLL_BATCH_SIZE", "RETRIEVER_TIMEOUT_SEC", "TILER_BASE_URL", "TILER_TIMEOUT_SEC",
    "VLM_BASE_URL", "VLM_MODEL_NAME", "VLM_API_KEY", "VLM_TIMEOUT_SEC",
    "VLM_CONNECT_TIMEOUT_SEC", "VLM_MAX_CONNECTIONS", "VLM_MAX_CONCURRENCY", "VLM_MAX_RETRIES",
    "RETRIEVER_CONNECT_TIMEOUT_SEC", "RETRIEVER_MAX_CONNECTIONS", "RETRIEVER_MAX_RETRIES",
    "TILER_CONNECT_TIMEOUT_SEC", "TILER_MAX_CONNECTIONS", "TILER_MAX_RETRIES", "TILER_MAX_CONCURRENCY",
    "HTTP_RETRY_BACKOFF_BASE_SEC", "HTTP_RETRY_BACKOFF_CAP_SEC",
    "VALKEY_URL", "VALKEY_PREFIX", "VALKEY_TIMEOUT_SEC", "VALKEY_MAX_CONNECTIONS", "VIRAG_PROCESS_TTL_SEC",
    "RERANKER_BASE_URL", "RERANKER_MODEL_NAME", "RERANKER_API_KEY", "RERANKER_API", "RERANKER_TIMEOUT_SEC",
    "RERANKER_BATCH_SIZE", "RERANKER_TOP_N", "RERANKER_THRESHOLD", "RERANKER_CONNECT_TIMEOUT_SEC",
    "RERANKER_MAX_CONNECTIONS", "RERANKER_MAX_CONCURRENCY", "RERANKER_MAX_RETRIES",
}


def test_defaults(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env()

    assert s.host == "127.0.0.1"
    assert s.port == 8080
    assert s.db_secret == ""
    assert s.retriever_settings.top_k == 5
    assert s.retriever_settings.threshold is None
    assert s.rag_settings.framing_text == _DEFAULT_FRAMING_TEXT
    assert s.qdrant_settings.vector_size == 768
    assert s.qdrant_settings.distance_type == Distance.COSINE
    assert s.vlm_settings.base_url == "http://vlm/v1"
    assert s.vlm_settings.model_name == "my-vlm"


@pytest.mark.parametrize("missing", ["QDRANT_VECTOR_SIZE", "VLM_BASE_URL", "VLM_MODEL_NAME", "VALKEY_URL"])
def test_required_vars(monkeypatch, missing):
    _base_env(monkeypatch)
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_port_bounds(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_PORT", "70000")

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_top_k_must_be_positive(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_TOP_K", "0")

    with pytest.raises(SettingsError, match="RETRIEVER_TOP_K"):
        Settings.from_env()


def test_retriever_threshold_empty_is_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_THRESHOLD", "")
    assert Settings.from_env().retriever_settings.threshold is None


def test_retriever_threshold_parsed(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_THRESHOLD", "0.42")
    assert Settings.from_env().retriever_settings.threshold == pytest.approx(0.42)


def test_retriever_threshold_bad_float(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_THRESHOLD", "high")

    with pytest.raises(SettingsError, match="RETRIEVER_THRESHOLD"):
        Settings.from_env()


def test_framing_text_empty_falls_back(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_FRAMING_TEXT", "")
    assert Settings.from_env().rag_settings.framing_text == _DEFAULT_FRAMING_TEXT


def test_framing_text_override(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_FRAMING_TEXT", "custom frame")
    assert Settings.from_env().rag_settings.framing_text == "custom frame"


def test_bad_distance_type(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("QDRANT_DISTANCE_TYPE", "Hamming")

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_db_secret_from_tiler_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("TILER_DB_SECRET", "s3cret")
    assert Settings.from_env().db_secret == "s3cret"


def test_http_transport_defaults(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env()

    assert s.http_settings.backoff_base_sec == 0.25
    assert s.http_settings.backoff_cap_sec == 8.0
    for svc in (s.retriever_settings, s.tiler_settings, s.vlm_settings):
        assert svc.connect_timeout_sec == 10
        assert svc.max_connections == 100
        assert svc.max_retries == 3


def test_vlm_timeout_below_old_floor_allowed(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VLM_TIMEOUT_SEC", "30")

    assert Settings.from_env().vlm_settings.timeout_sec == 30


def test_vlm_concurrency_cannot_exceed_pool(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VLM_MAX_CONNECTIONS", "4")
    monkeypatch.setenv("VLM_MAX_CONCURRENCY", "8")

    with pytest.raises(SettingsError, match="VLM_MAX_CONCURRENCY"):
        Settings.from_env()


def test_backoff_bad_float(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("HTTP_RETRY_BACKOFF_CAP_SEC", "soon")

    with pytest.raises(SettingsError, match="HTTP_RETRY_BACKOFF_CAP_SEC"):
        Settings.from_env()


def test_retriever_and_enroll_defaults(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env()

    assert s.retriever_settings.batch_size == 32
    assert s.retriever_settings.max_concurrency == 32
    assert s.retriever_settings.bulk_priority == 0
    assert s.retriever_settings.image_batching == "auto"
    assert s.enroll_settings.batch_size == 32


def test_reranker_off_by_default(monkeypatch):
    _base_env(monkeypatch)
    rs = Settings.from_env().reranker_settings

    assert not rs.enabled
    assert (rs.api, rs.batch_size, rs.top_n, rs.threshold, rs.timeout_sec) == ("score", 8, 5, None, 30)
    assert rs.model_fields_set == set()             # nothing set: the startup warning lists no ignored vars


def test_reranker_on_with_base_url(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8000")
    monkeypatch.setenv("RERANKER_API", "rerank")
    rs = Settings.from_env().reranker_settings

    assert rs.enabled and rs.base_url == "http://reranker:8000" and rs.api == "rerank"


def test_reranker_set_vars_are_tracked_while_off(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RERANKER_TOP_N", "3")

    assert Settings.from_env().reranker_settings.model_fields_set == {"top_n"}


def test_reranker_top_n_cannot_exceed_top_k(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8000")
    monkeypatch.setenv("RETRIEVER_TOP_K", "4")
    monkeypatch.setenv("RERANKER_TOP_N", "5")

    with pytest.raises(SettingsError, match="RERANKER_TOP_N"):
        Settings.from_env()


def test_reranker_top_n_may_equal_top_k(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RERANKER_BASE_URL", "http://reranker:8000")
    monkeypatch.setenv("RETRIEVER_TOP_K", "5")
    monkeypatch.setenv("RERANKER_TOP_N", "5")

    assert Settings.from_env().reranker_settings.top_n == 5


def test_reranker_top_n_unbound_while_off(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_TOP_K", "4")
    monkeypatch.setenv("RERANKER_TOP_N", "5")

    assert Settings.from_env().reranker_settings.top_n == 5


@pytest.mark.parametrize("name, value", [
    ("RERANKER_THRESHOLD", "1.5"),
    ("RERANKER_THRESHOLD", "-0.1"),
    ("RERANKER_API", "listwise"),
    ("RERANKER_BATCH_SIZE", "0"),
])
def test_reranker_bounds(monkeypatch, name, value):
    _base_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(SettingsError, match=name):
        Settings.from_env()


def test_image_batching_mode_validated(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_IMAGE_BATCHING", "sometimes")

    with pytest.raises(SettingsError, match="RETRIEVER_IMAGE_BATCHING"):
        Settings.from_env()


def test_retriever_concurrency_cannot_exceed_pool(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("RETRIEVER_MAX_CONNECTIONS", "8")
    monkeypatch.setenv("RETRIEVER_MAX_CONCURRENCY", "16")

    with pytest.raises(SettingsError, match="RETRIEVER_MAX_CONCURRENCY"):
        Settings.from_env()


def test_tiler_concurrency_default_and_pool_bound(monkeypatch):
    _base_env(monkeypatch)
    assert Settings.from_env().tiler_settings.max_concurrency == 32

    monkeypatch.setenv("TILER_MAX_CONNECTIONS", "8")
    with pytest.raises(SettingsError, match="TILER_MAX_CONCURRENCY"):
        Settings.from_env()


def test_empty_required_var_counts_as_missing(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VLM_BASE_URL", "")

    with pytest.raises(SettingsError, match="VLM_BASE_URL"):
        Settings.from_env()


def test_aliased_var_is_named_in_errors(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_RECONCILE_INTERVAL_SEC", "-1")

    with pytest.raises(SettingsError, match=r"^`VIRAG_RECONCILE_INTERVAL_SEC`"):
        Settings.from_env()


def test_valkey_and_coordination_defaults(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env()

    assert s.valkey_settings.url == "redis://valkey:6379/0"
    assert s.valkey_settings.prefix == "virag"
    assert s.coordination_settings.process_ttl_sec == 30


def test_process_ttl_has_a_floor(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_PROCESS_TTL_SEC", "1")

    with pytest.raises(SettingsError, match="VIRAG_PROCESS_TTL_SEC"):
        Settings.from_env()
