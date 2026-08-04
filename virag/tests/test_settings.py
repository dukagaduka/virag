"""Settings.from_env — required vars, coercion, enum and bounds validation."""

from __future__ import annotations

import pytest
from qdrant_client.models import Distance

from virag.settings import Settings, SettingsError, _DEFAULT_FRAMING_TEXT


def _base_env(monkeypatch):
    """The minimum env for a valid Settings: the two required vars."""
    for key in list(_ALL_KEYS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_VECTOR_SIZE", "768")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm/v1")
    monkeypatch.setenv("VLM_MODEL_NAME", "my-vlm")


_ALL_KEYS = {
    "VIRAG_HOST", "VIRAG_PORT", "TILER_DB_SECRET", "VIRAG_TOP_N", "VIRAG_SCORE_THRESHOLD",
    "VIRAG_FRAMING_TEXT", "QDRANT_BASE_URL", "QDRANT_COLLECTION_NAME", "QDRANT_VECTOR_SIZE",
    "QDRANT_DISTANCE_TYPE", "EMBEDDINGS_BASE_URL", "VLLM_SERVED_MODEL_NAME", "VLLM_API_KEY",
    "EMBEDDINGS_CHUNK_SIZE", "EMBEDDINGS_TIMEOUT_SEC", "TILER_BASE_URL", "TILER_TIMEOUT_SEC",
    "VLM_BASE_URL", "VLM_MODEL_NAME", "VLM_API_KEY", "VLM_TIMEOUT_SEC",
}


def test_defaults(monkeypatch):
    _base_env(monkeypatch)
    s = Settings.from_env()

    assert s.host == "127.0.0.1"
    assert s.port == 8080
    assert s.db_secret == ""
    assert s.rag_settings.top_n == 5
    assert s.rag_settings.score_threshold is None
    assert s.rag_settings.framing_text == _DEFAULT_FRAMING_TEXT
    assert s.qdrant_settings.vector_size == 768
    assert s.qdrant_settings.distance_type == Distance.COSINE
    assert s.vlm_settings.base_url == "http://vlm/v1"
    assert s.vlm_settings.model_name == "my-vlm"


@pytest.mark.parametrize("missing", ["QDRANT_VECTOR_SIZE", "VLM_BASE_URL", "VLM_MODEL_NAME"])
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


def test_top_n_must_be_positive(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_TOP_N", "0")

    with pytest.raises(SettingsError):
        Settings.from_env()


def test_score_threshold_empty_is_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_SCORE_THRESHOLD", "")
    assert Settings.from_env().rag_settings.score_threshold is None


def test_score_threshold_parsed(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_SCORE_THRESHOLD", "0.42")
    assert Settings.from_env().rag_settings.score_threshold == pytest.approx(0.42)


def test_score_threshold_bad_float(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("VIRAG_SCORE_THRESHOLD", "high")

    with pytest.raises(SettingsError):
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
