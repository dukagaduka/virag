"""chat_completions._rag_config — the RAG trigger normalization and validation."""

from __future__ import annotations

import pytest

from virag.api.chat_completions import _rag_config
from virag.errors import ApiError
from virag.settings import RagSettings


DEFAULTS = RagSettings(top_n=5, score_threshold=0.1, framing_text="DEFAULT FRAME")


@pytest.mark.parametrize("raw", [None, False])
def test_off(raw):
    assert _rag_config(raw, DEFAULTS) is None


def test_true_uses_defaults():
    cfg = _rag_config(True, DEFAULTS)
    assert cfg == {
        "top_n": 5,
        "score_threshold": 0.1,
        "framing_text": "DEFAULT FRAME",
        "include_sources": False,
    }


def test_object_overrides():
    cfg = _rag_config(
        {"top_n": 8, "score_threshold": 0.5, "framing_text": "F", "include_sources": True},
        DEFAULTS,
    )
    assert cfg == {"top_n": 8, "score_threshold": 0.5, "framing_text": "F", "include_sources": True}


def test_object_partial_fills_defaults():
    cfg = _rag_config({"top_n": 3}, DEFAULTS)
    assert cfg["top_n"] == 3
    assert cfg["score_threshold"] == 0.1
    assert cfg["framing_text"] == "DEFAULT FRAME"
    assert cfg["include_sources"] is False


def test_threshold_can_be_null():
    cfg = _rag_config({"score_threshold": None}, DEFAULTS)
    assert cfg["score_threshold"] is None


@pytest.mark.parametrize("raw", [
    "yes",                              # not object/bool
    {"top_n": 0},                       # not positive
    {"top_n": -1},
    {"top_n": True},                    # bool is not a valid int here
    {"top_n": 1.5},                     # not int
    {"score_threshold": "high"},        # not number/null
    {"score_threshold": True},          # bool rejected
    {"framing_text": 123},              # not string
])
def test_invalid_raises_422(raw):
    with pytest.raises(ApiError) as e:
        _rag_config(raw, DEFAULTS)
    assert e.value.status_code == 422


def test_include_sources_coerced_to_bool():
    assert _rag_config({"include_sources": 1}, DEFAULTS)["include_sources"] is True
    assert _rag_config({}, DEFAULTS)["include_sources"] is False
