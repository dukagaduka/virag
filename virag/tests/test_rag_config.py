"""chat_completions._rag_config — the RAG trigger normalization and validation."""

from __future__ import annotations

import pytest

from virag.api.chat_completions import _rag_config
from virag.errors import ApiError
from virag.settings import RagSettings, RerankerSettings, RetrieverSettings


RAG = RagSettings(framing_text="DEFAULT FRAME")
RETRIEVER = RetrieverSettings(top_k=5, threshold=0.1)
RERANKER = RerankerSettings(base_url="http://reranker", top_n=3, threshold=0.2)
NO_RERANKER = RerankerSettings(base_url=None, top_n=3, threshold=0.2)


def _config(raw, reranker=RERANKER):
    return _rag_config(raw, RAG, RETRIEVER, reranker)


@pytest.mark.parametrize("raw", [None, False])
def test_off(raw):
    assert _config(raw) is None


def test_true_uses_defaults():
    cfg = _config(True)
    assert cfg == {
        "top_k": 5,
        "retriever_threshold": 0.1,
        "top_n": 3,
        "reranker_threshold": 0.2,
        "framing_text": "DEFAULT FRAME",
        "include_sources": False,
    }


def test_object_overrides():
    cfg = _config({
        "top_k": 8, "retriever_threshold": 0.5, "top_n": 4, "reranker_threshold": 0.6,
        "framing_text": "F", "include_sources": True,
    })
    assert cfg == {
        "top_k": 8, "retriever_threshold": 0.5, "top_n": 4, "reranker_threshold": 0.6,
        "framing_text": "F", "include_sources": True,
    }


def test_object_partial_fills_defaults():
    cfg = _config({"top_k": 3})
    assert cfg["top_k"] == 3
    assert cfg["retriever_threshold"] == 0.1
    assert cfg["top_n"] == 3
    assert cfg["reranker_threshold"] == 0.2
    assert cfg["framing_text"] == "DEFAULT FRAME"
    assert cfg["include_sources"] is False


def test_thresholds_can_be_null():
    cfg = _config({"retriever_threshold": None, "reranker_threshold": None})
    assert cfg["retriever_threshold"] is None
    assert cfg["reranker_threshold"] is None


def test_top_n_may_equal_top_k():
    assert _config({"top_k": 4, "top_n": 4})["top_n"] == 4


def test_top_n_above_top_k_is_fine_without_a_reranker():
    assert _config({"top_k": 2, "top_n": 3}, reranker=NO_RERANKER)["top_n"] == 3


@pytest.mark.parametrize("raw", [
    "yes",                              # not object/bool
    {"top_k": 0},                       # not positive
    {"top_k": -1},
    {"top_k": True},                    # bool is not a valid int here
    {"top_k": 1.5},                     # not int
    {"retriever_threshold": "high"},    # not number/null
    {"retriever_threshold": True},      # bool rejected
    {"top_n": 0},
    {"top_n": "2"},
    {"top_k": 2, "top_n": 3},           # the reranker picks out of the candidates
    {"reranker_threshold": "high"},
    {"reranker_threshold": 1.5},        # a relevance is within [0, 1]
    {"reranker_threshold": -0.1},
    {"framing_text": 123},              # not string
])
def test_invalid_raises_422(raw):
    with pytest.raises(ApiError) as e:
        _config(raw)
    assert e.value.status_code == 422


def test_include_sources_coerced_to_bool():
    assert _config({"include_sources": 1})["include_sources"] is True
    assert _config({})["include_sources"] is False
