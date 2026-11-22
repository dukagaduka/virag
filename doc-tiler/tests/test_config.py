"""Settings: every TILER_* variable lands in its section; a bad one is named."""

from __future__ import annotations

import pytest

from tiler.config import ConfigError, Settings


def test_every_section_reads_its_variables(monkeypatch):
    for name, value in {
        "TILER_TILE_WIDTH": "900",
        "TILER_PDF_SLICE_ENABLE": "true",          # an aliased name
        "TILER_JPEG_QUALITY": "70",
        "TILER_CHROMIUM_REVISION": "chromium-1",   # an aliased name
        "TILER_N_RENDER_WORKERS": "6",
        "TILER_RENDER_TIMEOUT_SEC": "30",
        "TILER_DB_PATH": "/x/meta.sqlite",
        "TILER_BLOB_DIR": "/x/blobs",
        "TILER_MAX_DOC_BYTES": "1000",
        "TILER_URL_ALLOW_PRIVATE": "true",
        "TILER_REVISION_CONCURRENCY": "2",
        "TILER_API_KEY": "k",
        "TILER_ALLOW_DELETE": "true",
        "TILER_TILE_OVERLAP": "",                  # empty counts as unset
    }.items():
        monkeypatch.setenv(name, value)

    monkeypatch.delenv("TILER_TMP_DIR", raising=False)     # so that it follows the blob dir

    s = Settings.from_env()

    assert (s.geometry_settings.tile_width, s.geometry_settings.pdf_slice, s.geometry_settings.tile_overlap) == (900, True, 0)
    assert (s.pixel_settings.jpeg_quality, s.pixel_settings.chromium_revision) == (70, "chromium-1")
    assert (s.render_settings.render_workers, s.render_settings.bulk_render_workers) == (6, 3)
    assert s.render_settings.render_timeout_sec == 30
    assert (s.store_settings.db_path, s.store_settings.temp_dir, s.store_settings.max_doc_bytes) == ("/x/meta.sqlite", "/x/tmp", 1000)
    assert (s.capture_settings.url_fetch_enabled, s.capture_settings.url_allow_private) == (True, True)
    assert (s.revision_settings.revision_on_render_mismatch, s.revision_settings.revision_concurrency) == (True, 2)
    assert (s.api_settings.api_key, s.api_settings.allow_delete) == ("k", True)


@pytest.mark.parametrize(
    "name, value",
    [
        ("TILER_JPEG_QUALITY", "0"),               # out of its range
        ("TILER_TILE_WIDTH", "wide"),              # not a number
        ("TILER_PDF_SLICE_ENABLE", "maybe"),       # an aliased name
        ("TILER_TILE_OVERLAP", "2000"),            # against another variable of its section
        ("TILER_SAFE_FULLPAGE_PX", "10"),          # against a variable of another section
        ("TILER_REVISION_CONCURRENCY", "0"),
        ("TILER_MAX_DOC_BYTES", "-1"),
    ],
)
def test_an_invalid_variable_is_named(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ConfigError, match=name):
        Settings.from_env()
