"""Operational endpoints: schema, health, readiness, metrics, info.

`router` holds the probes and metrics, open to orchestrators and Prometheus;
`keyed_router` holds what needs TILER_API_KEY (schema, info).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request, Response

from .. import __version__
from .. import metrics as m
from ..models import MetadataSchemaModel
from ..responses import OrjsonResponse

router = APIRouter()
keyed_router = APIRouter()


@keyed_router.get(
    "/schema/metadata",
    response_model=MetadataSchemaModel,
    summary="Return the operator-declared metadata schema (which keys are searchable).",
    operation_id="getMetadataSchema",
)
async def get_metadata_schema(request: Request) -> MetadataSchemaModel:
    """Return the operator-declared metadata schema."""
    return MetadataSchemaModel(**request.app.state.schema.as_dict())


@router.get("/healthz", summary="Liveness.", operation_id="healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    # Deliberately touches nothing so liveness stays fast under full load.
    return {"status": "ok"}


@router.get(
    "/readyz", summary="Readiness (DB, blob dir, Chromium pool, revision gate).", operation_id="readyz"
)
async def readyz(request: Request) -> Response:
    """Readiness probe checking the DB, blob dir, render pool, Chromium and revision gate."""
    state = request.app.state
    checks: dict[str, Any] = {}
    ok = True

    try:
        await state.store.ping()
        checks["db"] = "ok"

    except Exception as exc:
        checks["db"] = f"error: {exc}"
        ok = False

    try:
        await asyncio.to_thread(state.blobs.ensure_root)
        checks["blobs"] = "ok"

    except Exception as exc:
        checks["blobs"] = f"error: {exc}"
        ok = False

    pool_ok = await state.pool.healthy()
    checks["render_pool"] = "ok" if pool_ok else "down"

    if not pool_ok:
        ok = False

    browser = state.pool.browser

    if state.settings.render_settings.browser_enabled:
        checks["browser"] = "ok" if browser.available else f"unavailable: {browser.start_error}"

        if not browser.available:
            ok = False

    else:
        checks["browser"] = "disabled"

    # Not ready while docs from another render_version remain (see GET /revision).
    checks["revision"] = state.revision.status["state"]

    if not state.revision.serving:
        ok = False

    # Counted live: a re-ingest lifts a quarantine without any revision running.
    quarantined = await state.store.count_quarantined()
    m.QUARANTINED.set(quarantined)
    checks["quarantined"] = quarantined

    return OrjsonResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not ready", "checks": checks},
    )


@router.get("/metrics", summary="Prometheus metrics.", operation_id="metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics."""
    payload, content_type = m.render_latest()

    return Response(content=payload, media_type=content_type)


@keyed_router.get("/info", summary="Effective geometry, render_version and cache stats.")
async def info(request: Request) -> dict[str, Any]:
    """Report effective geometry, render_version, revision gate, policy, pools and cache stats."""
    state = request.app.state
    settings = state.settings
    render = settings.render_settings
    capture = settings.capture_settings

    return {
        "version": __version__,
        "geometry": settings.geometry_settings.tier,
        "render_version": settings.render_version,
        "volume_id": state.volume_id,
        "pixel_tier": settings.pixel_settings.tier,
        "revision": state.revision.status,
        "policy": {
            "revision_on_render_mismatch": settings.revision_settings.revision_on_render_mismatch,
            "url_fetch_enabled": capture.url_fetch_enabled,
            "url_allow_private": capture.url_allow_private,
            "allow_delete": settings.api_settings.allow_delete,
            "ssrf_guard_proxy": state.capture.guard_proxy.url if state.capture.guard_proxy else None,
        },
        "pools": {
            "browsers": render.browsers,
            "render_workers": render.render_workers,
            "bulk_browsers": render.bulk_browsers,
            "bulk_render_workers": render.bulk_render_workers,
            "browser_available": state.pool.browser.available,
            "render_worker_memory": render.render_worker_memory,
            "render_timeout_sec": render.render_timeout_sec,
            "render_worker_restarts": state.pool.worker_restarts,
        },
        "cache": {"tile": state.tiles.cache.stats(), "page": state.pool.page_cache.stats()},
        "docs": await state.store.count_docs(),
    }
