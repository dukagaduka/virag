"""Prometheus metrics (exposed at /metrics)."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

INGESTS = Counter("tiler_ingest_total", "Ingest requests", ["doc_type", "stored", "result"])
RENDERS = Counter(
    "tiler_renders_total",
    "Document renders actually performed (a cache/idempotency hit does not count)",
    ["doc_type", "path"],
)
TILES = Counter("tiler_tiles_total", "Tiles produced", ["doc_type", "path"])
RENDER_SECONDS = Histogram(
    "tiler_render_seconds",
    "Wall time of a full document render",
    ["doc_type", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
CACHE = Counter("tiler_cache_total", "Tile/page cache outcomes", ["cache", "outcome"])
ERRORS = Counter("tiler_errors_total", "Errors returned to clients", ["code"])
MISMATCHES = Counter("tiler_hash_mismatch_total", "Regenerated tile hash mismatches", ["doc_type"])
SSRF_BLOCKS = Counter("tiler_ssrf_blocked_total", "Requests blocked by the SSRF policy")
BROWSER_RECYCLES = Counter("tiler_browser_recycles_total", "Chromium process recycles")
DOCS = Gauge("tiler_docs", "Documents currently stored")
INFLIGHT = Gauge("tiler_inflight_renders", "Renders currently in flight", ["layer"])


def render_latest() -> tuple[bytes, str]:
    """Return the current metrics exposition bytes and their content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
