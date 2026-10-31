"""Serving gate: virag serves only while its collection matches the tiler's render_version."""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from virag.api import chat_completions as chat_api
from virag.api import enroll as enroll_api
from virag.api import ops as ops_api
from virag.api import revision as revision_api
from virag.errors import ApiError, install_error_handlers
from virag.gate import ServingGate, require_serving
from virag.revision import RevisionManager
from virag.settings import EnrollSettings, ReconcileSettings, RevisionSettings

from conftest import FakeRetriever, FakeStore, FakeTiler, FakeVLM, make_settings


def _doc(doc_id, tile_count=1):
    return {
        "doc_id": doc_id, "doc_type": "PDF", "tile_count": tile_count, "complete": True,
        "render_version": "rv", "source_url": None,
        "metadata": {"name": doc_id, "virag_started_at": 1, "virag_enrolled_at": 1},
    }


def _gate(store, tiler, *, auto=True, secret=""):
    revision = RevisionManager(
        store,
        tiler,
        FakeRetriever(),
        EnrollSettings(batch_size=4),
        ReconcileSettings(self_heal=True, interval_sec=0, pending_ttl_sec=5),
    )

    return ServingGate(store, tiler, revision, RevisionSettings(revision_on_render_mismatch=auto), secret)


def _tiler(render_version="rv-test", *, serving=True, secret=""):
    return FakeTiler(info={
        "secret_hash": secret,
        "render_version": render_version,
        "revision": {"state": "succeeded" if serving else "running", "serving": serving},
    })


def _store(render_version="rv-test"):
    store = FakeStore()
    store.metadata = {"secret_hash": "", "render_version": render_version}

    return store


def test_opens_when_the_collection_matches_the_tiler():
    gate = _gate(_store(), _tiler())
    asyncio.run(gate.check())

    assert gate.serving and gate.reason is None


def test_waits_while_the_tiler_revises():
    gate = _gate(_store(), _tiler(serving=False))
    asyncio.run(gate.check())

    assert not gate.serving
    assert "tiler is not serving" in gate.reason
    assert not gate.revision.running                                    # nothing to rebuild yet


def test_mismatch_rebuilds_the_collection_once_then_opens():
    store, tiler = _store("rv-old"), _tiler("rv-new")
    tiler.docs = [_doc("a", 2)]
    gate = _gate(store, tiler)

    async def go():
        await gate.check()
        assert not gate.serving and "rv-old" in gate.reason
        assert gate.revision.running

        await gate.revision._task
        await gate.check()

    asyncio.run(go())

    assert gate.revision.status["state"] == "succeeded"
    assert store.metadata["render_version"] == "rv-new"
    assert gate.serving


def test_failed_rebuild_is_not_retried_for_the_same_render_version():
    store, tiler = _store("rv-old"), _tiler("rv-new")
    tiler.docs = [_doc("bad")]
    tiler.fail_tiles = {("bad", 0)}
    gate = _gate(store, tiler)

    async def go():
        await gate.check()
        first = gate.revision._task
        await first

        await gate.check()                                              # no automatic second try

        return first

    first = asyncio.run(go())

    assert gate.revision.status["state"] == "failed"
    assert gate.revision._task is first
    assert not gate.serving


def test_mismatch_without_auto_revision_waits_for_post():
    gate = _gate(_store("rv-old"), _tiler("rv-new"), auto=False)
    asyncio.run(gate.check())

    assert not gate.serving
    assert gate.revision._task is None


def test_secret_mismatch_closes_the_gate():
    gate = _gate(_store(), _tiler(secret="someone-else"))
    asyncio.run(gate.check())

    assert not gate.serving and "TILER_DB_SECRET" in gate.reason


def test_unreachable_tiler_keeps_the_gate_as_it_is():
    tiler = _tiler()
    gate = _gate(_store(), tiler)

    async def unreachable():
        raise ApiError(status_code=503, message="Cannot reach the tiler")

    async def go():
        await gate.check()
        tiler.get_info = unreachable
        await gate.check()

    asyncio.run(go())

    assert gate.serving


def _gated_app(gate) -> FastAPI:
    """Routed like create_app(): chat and enroll behind the gate, ops and revision in front."""
    app = FastAPI()
    install_error_handlers(app)

    serving = [Depends(require_serving)]

    app.include_router(ops_api.router)
    app.include_router(enroll_api.router, dependencies=serving)
    app.include_router(chat_api.router, dependencies=serving)
    app.include_router(revision_api.router)

    app.state.settings = make_settings()
    app.state.store = gate.store
    app.state.tiler = gate.tiler
    app.state.retriever = FakeRetriever()
    app.state.vlm = FakeVLM()
    app.state.revision = gate.revision
    app.state.gate = gate

    return app


def test_closed_gate_answers_503_on_chat_and_enroll_but_not_on_ops():
    gate = _gate(_store("rv-old"), _tiler("rv-new"), auto=False)
    asyncio.run(gate.check())

    with TestClient(_gated_app(gate), raise_server_exceptions=False) as client:
        chat = client.post("/v1/chat/completions", json={"model": "vlm", "messages": []})
        assert chat.status_code == 503
        assert chat.json()["error"]["code"] == "NOT_READY"
        assert "rv-old" in chat.json()["error"]["message"]

        enroll = client.post("/enroll", data={"doc_type": "PDF"}, files={"document": ("d.pdf", b"x")})
        assert enroll.status_code == 503

        ready = client.get("/ready")
        assert ready.status_code == 503 and "rv-old" in ready.json()["reason"]

        assert client.get("/health").status_code == 200
        assert client.get("/revision").status_code == 200


def test_open_gate_lets_chat_through():
    gate = _gate(_store(), _tiler())
    asyncio.run(gate.check())

    with TestClient(_gated_app(gate), raise_server_exceptions=False) as client:
        chat = client.post("/v1/chat/completions", json={"model": "vlm", "messages": [{"role": "user", "content": "hi"}]})
        assert chat.status_code == 200
        assert client.get("/ready").status_code == 200
