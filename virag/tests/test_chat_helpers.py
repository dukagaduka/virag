"""chat_completions message helpers: query extraction, tile injection, sources."""

from __future__ import annotations

from virag.api.chat_completions import _extract_query, _inject_tiles, _last_user_index, _source

from conftest import FakeHit


def test_extract_query_string_content():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "  hello  "},
    ]
    assert _extract_query(messages) == "hello"


def test_extract_query_last_user_wins():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert _extract_query(messages) == "second"


def test_extract_query_multimodal_concats_text_parts():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "what is"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "this"},
        ],
    }]
    assert _extract_query(messages) == "what is this"


def test_extract_query_no_user():
    assert _extract_query([{"role": "system", "content": "x"}]) == ""


def test_last_user_index():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert _last_user_index(messages) == 0


def test_inject_tiles_into_string_message():
    messages = [{"role": "user", "content": "original question"}]
    _inject_tiles(messages, ["AAA", "BBB"], "FRAME")

    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "FRAME"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,BBB"
    # original text preserved, appended last
    assert content[-1] == {"type": "text", "text": "original question"}


def test_inject_tiles_preserves_multimodal_original():
    original = [
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": {"url": "data:user-img"}},
    ]
    messages = [{"role": "user", "content": list(original)}]
    _inject_tiles(messages, ["TILE"], "FRAME")

    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "FRAME"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,TILE"
    # the caller's original parts are kept after the injected framing + tiles
    assert content[-2:] == original


def test_source_shape():
    hit = FakeHit(score=0.77, payload={
        "doc_id": "d1", "tile_num": 3, "metadata": {"name": "Report"},
    })
    assert _source(hit) == {"name": "Report", "doc_id": "d1", "tile_num": 3, "score": 0.77}


def test_source_missing_metadata():
    hit = FakeHit(score=0.5, payload={"doc_id": "d2", "tile_num": 0})
    src = _source(hit)
    assert src["name"] is None
    assert src["doc_id"] == "d2"
