"""binding: the Qdrant collection is bound to the tiler volume it was embedded from."""

from __future__ import annotations

import asyncio

import pytest

from virag.binding import BindingError, enforce_binding, enforce_metadata_schema

from conftest import FakeStore, FakeTiler


def test_fresh_collection_is_bound():
    store = FakeStore()                                     # metadata None → unbound
    tiler = FakeTiler(info={"volume_id": "vol-1", "render_version": "rv-9"})

    rv = asyncio.run(enforce_binding(store, tiler))

    assert rv == "rv-9"
    assert store.metadata == {"tiler_volume_id": "vol-1", "render_version": "rv-9"}


def test_already_bound_same_volume_ok():
    store = FakeStore()
    store.metadata = {"tiler_volume_id": "vol-1", "render_version": "rv-9"}
    tiler = FakeTiler(info={"volume_id": "vol-1", "render_version": "rv-9"})

    assert asyncio.run(enforce_binding(store, tiler)) == "rv-9"


def test_another_volume_is_fatal():
    # e.g. the tiler came back on a fresh, empty volume: reconcile would take every point for an orphan
    store = FakeStore()
    store.metadata = {"tiler_volume_id": "vol-1", "render_version": "rv"}
    tiler = FakeTiler(info={"volume_id": "vol-2", "render_version": "rv"})

    with pytest.raises(BindingError, match="vol-1.*vol-2"):
        asyncio.run(enforce_binding(store, tiler))


def test_tiler_without_volume_id_is_fatal():
    store = FakeStore()
    tiler = FakeTiler(info={"render_version": "rv"})

    with pytest.raises(BindingError):
        asyncio.run(enforce_binding(store, tiler))
    assert store.metadata is None                           # nothing bound


def test_render_version_drift_is_not_fatal():
    store = FakeStore()
    store.metadata = {"tiler_volume_id": "vol-1", "render_version": "rv-OLD"}
    tiler = FakeTiler(info={"volume_id": "vol-1", "render_version": "rv-NEW"})

    rv = asyncio.run(enforce_binding(store, tiler))
    assert rv == "rv-NEW"        # not fatal: chat answers RENDER_VERSION_MISMATCH until a revision
    assert store.metadata["render_version"] == "rv-OLD"     # binding never rewrites it


def test_metadata_schema_complete_ok():
    tiler = FakeTiler()                                     # default schema has all three
    asyncio.run(enforce_metadata_schema(tiler))             # does not raise


def test_metadata_schema_missing_field_is_fatal():
    tiler = FakeTiler()
    tiler.schema = {"version": 1, "fields": [{"key": "name", "type": "text"}]}

    with pytest.raises(BindingError):
        asyncio.run(enforce_metadata_schema(tiler))


def test_metadata_schema_wrong_type_is_fatal():
    tiler = FakeTiler()
    tiler.schema = {
        "version": 1,
        "fields": [
            {"key": "name", "type": "text"},
            {"key": "virag_started_at", "type": "text"},        # should be number
            {"key": "virag_enrolled_at", "type": "number"},
        ],
    }

    with pytest.raises(BindingError):
        asyncio.run(enforce_metadata_schema(tiler))
