"""virag's own keys: VIRAG_API_KEY opens /v1/*, VIRAG_ADMIN_API_KEY opens /enroll and
/revision* (and /v1/* too); an empty key leaves its routes open. Plus the keys virag
sends upstream: TILER_API_KEY and the /v1/models answer."""

from __future__ import annotations

import httpx2
import pytest

from virag.clients.tiler_client import TilerService
from virag.clients.vlm_client import VLMClient
from virag.errors import ApiError

from conftest import FakeRetriever, FakeStore, FakeTiler, FakeVLM, make_settings
from test_http_service import _HTTP, _mock, _run


USER, ADMIN = "user-key", "admin-key"


def _client(make_client, **keys):
    return make_client(
        settings=make_settings(**keys),
        vlm=FakeVLM(),
        store=FakeStore(),
        retriever=FakeRetriever(),
        tiler=FakeTiler(),
    )


def _bearer(key: str | None) -> dict:
    return {"Authorization": f"Bearer {key}"} if key is not None else {}


def _chat(client, key):
    return client.post("/v1/chat/completions", headers=_bearer(key), json={"model": "m", "messages": [{"role": "user", "content": "q"}]})


def _revision_status(client, key):
    return client.get("/revision", headers=_bearer(key))


@pytest.mark.parametrize("key, status", [(USER, 200), (ADMIN, 200), (None, 401), ("wrong", 401)])
def test_chat_takes_the_user_or_the_admin_key(make_client, key, status):
    client = _client(make_client, api_key=USER, admin_api_key=ADMIN)

    r = _chat(client, key)

    assert r.status_code == status
    if status == 401:
        assert r.json()["error"]["code"] == "UNAUTHORIZED"
        assert r.headers["www-authenticate"] == "Bearer"


def _passed(response) -> bool:
    """Through the key check (the route itself may still fail on the test's bare state)."""
    return response.status_code not in (401, 403)


@pytest.mark.parametrize("key, status", [(ADMIN, None), (USER, 403), (None, 401), ("wrong", 401)])
def test_admin_routes_take_the_admin_key_only(make_client, key, status):
    client = _client(make_client, api_key=USER, admin_api_key=ADMIN)

    for r in (_revision_status(client, key), client.post("/enroll", headers=_bearer(key), data={"doc_type": "PDF"})):
        assert _passed(r) if status is None else r.status_code == status


def test_empty_keys_leave_the_routes_open(make_client):
    client = _client(make_client)

    assert _chat(client, None).status_code == 200
    assert _passed(_revision_status(client, None))


def test_probes_stay_open(make_client):
    client = _client(make_client, api_key=USER, admin_api_key=ADMIN)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_proxies_are_gone(make_client):
    client = _client(make_client)

    for path in ("/qdrant/collections", "/tiler/info", "/retriever/models", "/v1/embeddings"):
        assert client.get(path).status_code in (404, 405), path


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

def _vlm() -> VLMClient:
    return VLMClient("http://vlm/v1", "vlm", "k-vlm", max_retries=0, **_HTTP)


def test_models_lists_only_the_configured_model():
    vlm = _vlm()
    upstream = [{"id": "other", "object": "model"}, {"id": "vlm", "object": "model", "owned_by": "vllm", "max_model_len": 32768}]
    seen = _mock(vlm, lambda req, n: httpx2.Response(200, json={"object": "list", "data": upstream}))

    assert _run(vlm.model_card()) == upstream[1]
    assert seen[0].headers["authorization"] == "Bearer k-vlm"


@pytest.mark.parametrize("response", [
    httpx2.Response(200, json={"data": [{"id": "other"}]}),
    httpx2.Response(500),
    httpx2.Response(200, content=b"not json"),
])
def test_models_falls_back_to_a_minimal_card(response):
    vlm = _vlm()
    _mock(vlm, lambda req, n: response)

    card = _run(vlm.model_card())

    assert card["id"] == "vlm" and card["object"] == "model"


def test_models_route_needs_the_user_key(make_client):
    class _CardVLM(FakeVLM):
        async def model_card(self):
            return {"id": "vlm", "object": "model"}

    client = make_client(settings=make_settings(api_key=USER), vlm=_CardVLM())

    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=_bearer(USER)).json() == {"object": "list", "data": [{"id": "vlm", "object": "model"}]}


# ---------------------------------------------------------------------------
# TILER_API_KEY
# ---------------------------------------------------------------------------

def _tiler(api_key: str = "") -> TilerService:
    return TilerService("http://tiler", api_key=api_key, max_retries=0, **_HTTP)


def test_tiler_key_is_sent_and_checked():
    tiler = _tiler("t-key")
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, json={"status": "ready"}))

    _run(tiler.check_connection())

    assert [r.url.path for r in seen] == ["/readyz", "/info"]
    assert all(r.headers["authorization"] == "Bearer t-key" for r in seen)


def test_tiler_without_key_sends_no_header():
    tiler = _tiler()
    seen = _mock(tiler, lambda req, n: httpx2.Response(200, json={}))

    _run(tiler.check_connection())

    assert "authorization" not in seen[0].headers


def test_tiler_rejecting_the_key_is_named():
    tiler = _tiler("wrong")
    _mock(tiler, lambda req, n: httpx2.Response(200 if req.url.path == "/readyz" else 401))

    with pytest.raises(ApiError, match="TILER_API_KEY"):
        _run(tiler.check_connection())
