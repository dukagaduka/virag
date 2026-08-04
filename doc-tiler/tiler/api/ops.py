"""Operational endpoints: schema, health, readiness, metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from .. import __version__
from .. import metrics as m
from ..models import MetadataSchemaModel
from .deps import service

router = APIRouter()


@router.get(
    "/schema/metadata",
    response_model=MetadataSchemaModel,
    summary="Return the operator-declared metadata schema (which keys are searchable).",
    operation_id="getMetadataSchema",
)
async def get_metadata_schema(request: Request) -> MetadataSchemaModel:
    """Return the operator-declared metadata schema."""
    return MetadataSchemaModel(**service(request).schema.as_dict())


@router.get("/healthz", summary="Liveness.", operation_id="healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe."""
    # Deliberately touches nothing so liveness stays fast under full load.
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness (DB, blob dir, Chromium pool).", operation_id="readyz")
async def readyz(request: Request) -> Response:
    """Readiness probe checking the DB, blob dir and Chromium pool."""
    ok, checks = await service(request).ready()
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "not ready", "checks": checks},
    )


@router.get("/metrics", summary="Prometheus metrics.", operation_id="metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics."""
    payload, content_type = m.render_latest()
    return Response(content=payload, media_type=content_type)


@router.get("/info", summary="Effective geometry, render_version and cache stats.")
async def info(request: Request) -> dict[str, Any]:
    """Report effective geometry, render_version, policy, pools and cache stats."""
    svc = service(request)
    return {
        "version": __version__,
        "geometry": svc.settings.geometry,
        "render_version": svc.settings.render_version,
        "secret_hash": svc.settings.secret_hash,
        "pixel_tier": svc.settings.pixel_tier,
        "policy": {
            "render_mismatch": svc.settings.render_mismatch_policy,
            "url_fetch_enabled": svc.settings.url_fetch_enabled,
            "url_allow_private": svc.settings.url_allow_private,
            "allow_delete": svc.settings.allow_delete,
            "ssrf_guard_proxy": svc.guard_proxy.url if svc.guard_proxy else None,
        },
        "pools": {
            "browsers": svc.settings.browsers,
            "cpu_workers": svc.settings.cpu_workers,
            "browser_available": svc.browser.available,
        },
        "cache": {"tile": svc.tile_cache.stats(), "page": svc.page_cache.stats()},
        "docs": await svc.store.count_docs(),
    }
