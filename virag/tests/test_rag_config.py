"""chat_completions._rag_config — the RAG trigger normalization and validation."""

from __future__ import annotations

import pytest

from virag.api.chat_completions import _rag_config
from virag.errors import ApiError
from virag.settings import RagSettings, RetrieverSettings


RAG = RagSettings(framing_text="DEFAULT FRAME")
RETRIEVER = RetrieverSettings(top_k=5, threshold=0.1)


def _config(raw):
    return _rag_config(raw, RAG, RETRIEVER)


@pytest.mark.parametrize("raw", [None, False])
def test_off(raw):
    assert _config(raw) is None


def test_true_uses_defaults():
    cfg = _config(True)
    assert cfg == {
        "top_k": 5,
        "retriever_threshold": 0.1,
        "framing_text": "DEFAULT FRAME",
        "include_sources": False,
    }


def test_object_overrides():
    cfg = _config({"top_k": 8, "retriever_threshold": 0.5, "framing_text": "F", "include_sources": True})
    assert cfg == {"top_k": 8, "retriever_threshold": 0.5, "framing_text": "F", "include_sources": True}


def test_object_partial_fills_defaults():
    cfg = _config({"top_k": 3})
    assert cfg["top_k"] == 3
    assert cfg["retriever_threshold"] == 0.1
    assert cfg["framing_text"] == "DEFAULT FRAME"
    assert cfg["include_sources"] is False


def test_threshold_can_be_null():
    cfg = _config({"retriever_threshold": None})
    assert cfg["retriever_threshold"] is None


@pytest.mark.parametrize("raw", [
    "yes",                              # not object/bool
    {"top_k": 0},                       # not positive
    {"top_k": -1},
    {"top_k": True},                    # bool is not a valid int here
    {"top_k": 1.5},                     # not int
    {"retriever_threshold": "high"},    # not number/null
    {"retriever_threshold": True},      # bool rejected
    {"framing_text": 123},              # not string
])
def test_invalid_raises_422(raw):
    with pytest.raises(ApiError) as e:
        _config(raw)
    assert e.value.status_code == 422


def test_include_sources_coerced_to_bool():
    assert _config({"include_sources": 1})["include_sources"] is True
    assert _config({})["include_sources"] is False
