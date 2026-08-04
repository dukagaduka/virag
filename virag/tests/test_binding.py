"""binding: secret hashing and the fatal virag/tiler/qdrant secret binding."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from virag.binding import BindingError, enforce_binding, secret_hash

from conftest import FakeStore, FakeTiler


def test_secret_hash_empty():
    assert secret_hash("") == ""


def test_secret_hash_value():
    assert secret_hash("abc") == hashlib.sha256(b"abc").hexdigest()


def test_fresh_collection_is_bound():
    expected = secret_hash("shared")
    store = FakeStore()                                     # metadata None → unbound
    tiler = FakeTiler(info={"secret_hash": expected, "render_version": "rv-9"})

    rv = asyncio.run(enforce_binding(store, tiler, expected))

    assert rv == "rv-9"
    assert store.metadata == {"secret_hash": expected, "render_version": "rv-9"}


def test_already_bound_matching_secret_ok():
    expected = secret_hash("shared")
    store = FakeStore()
    store.metadata = {"secret_hash": expected, "render_version": "rv-9"}
    tiler = FakeTiler(info={"secret_hash": expected, "render_version": "rv-9"})

    rv = asyncio.run(enforce_binding(store, tiler, expected))
    assert rv == "rv-9"


def test_tiler_secret_mismatch_is_fatal():
    store = FakeStore()
    tiler = FakeTiler(info={"secret_hash": secret_hash("other"), "render_version": "rv"})

    with pytest.raises(BindingError):
        asyncio.run(enforce_binding(store, tiler, secret_hash("shared")))


def test_collection_secret_mismatch_is_fatal():
    expected = secret_hash("shared")
    store = FakeStore()
    store.metadata = {"secret_hash": secret_hash("someone-else"), "render_version": "rv"}
    tiler = FakeTiler(info={"secret_hash": expected, "render_version": "rv"})

    with pytest.raises(BindingError):
        asyncio.run(enforce_binding(store, tiler, expected))


def test_render_version_drift_is_not_fatal():
    expected = secret_hash("shared")
    store = FakeStore()
    store.metadata = {"secret_hash": expected, "render_version": "rv-OLD"}
    tiler = FakeTiler(info={"secret_hash": expected, "render_version": "rv-NEW"})

    rv = asyncio.run(enforce_binding(store, tiler, expected))
    assert rv == "rv-NEW"        # returns the tiler's version, only warns on drift
