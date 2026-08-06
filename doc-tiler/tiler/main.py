"""FastAPI app: startup wiring (guard, pools, DB, metadata indexes) and route registration."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__, guard, metrics
from .api import docs as docs_api
from .api import ingest as ingest_api
from .api import metadata as metadata_api
from .api import ops as ops_api
from .api import search as search_api
from .api import tiles as tiles_api
from .config import ConfigError, Settings, load_settings
from .errors import install_error_handlers
from .meta.schema import MetadataSchema, SchemaError, index_statements, load_schema
from .service import TilerService

log = logging.getLogger("tiler")

DESCRIPTION = (
    "Turns documents (PDF / URL / MHTML / IMG) into fixed-geometry image tiles "
    "(width=875, height<=1024) for a pixel-based RAG embedder, and regenerates any "
    "single tile on demand from the stored original."
)


def configure_logging() -> None:
    """Send structured logs to stdout at the LOG_LEVEL from the environment."""
    logging.basicConfig(
        level=os.environ.get("TILER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def bootstrap(settings: Settings, schema: MetadataSchema) -> TilerService:
    """Everything that must succeed before the first request is served."""
    svc = TilerService(settings, schema)

    os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)) or ".", exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    svc.blobs.ensure_root()

    await svc.store.connect()

    try:
        await svc.store.create_tables()
        # Hard geometry gate: refuse to start against a volume built with different geometry.
        await guard.enforce(svc.store, settings)

    except BaseException:
        await svc.store.close()
        raise

    created = await svc.store.create_metadata_indexes(index_statements(schema))
    dropped = await svc.store.drop_stale_metadata_indexes({f.index_name for f in schema.fields})

    log.info(
        "metadata schema v%s: %d searchable field(s) %s%s",
        schema.version,
        len(schema.fields),
        [f.key for f in schema.fields],
        f"; dropped stale index(es) {dropped}" if dropped else "",
    )
    log.debug("index statements: %s", created)

    # Give the planner statistics; without them a selective range filter can lose
    # out to the created_at index (see README, "search and the query planner").
    await svc.store.optimize()

    await svc.start_pools()
    metrics.DOCS.set(await svc.store.count_docs())

    log.info(
        "ready: geometry=%s render_version=%s browsers=%d cpu_workers=%d",
        settings.geometry,
        settings.render_version,
        settings.browsers,
        settings.cpu_workers,
    )
    
    return svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        settings = load_settings()
        schema = load_schema(settings.metadata_schema_path)

    except (ConfigError, SchemaError) as exc:
        log.critical("FATAL: %s", exc)
        raise SystemExit(1) from exc

    try:
        svc = await bootstrap(settings, schema)
    except guard.GeometryMismatch as exc:
        # Never serve a single request against a mismatched volume.
        raise SystemExit(1) from exc
    
    app.state.service = svc
    app.state.settings = settings

    sweeper = asyncio.create_task(_run_sweeper(svc, settings))

    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass

        await svc.stop_pools()
        await svc.store.close()


async def _run_sweeper(svc: TilerService, settings: Settings) -> None:
    """Periodically reap incomplete ingests older than the TTL (0 = disabled)."""
    if settings.incomplete_ttl_sec <= 0:
        return

    while True:
        await asyncio.sleep(settings.sweep_interval_sec)

        try:
            await svc.sweep_incomplete(settings.incomplete_ttl_sec)

        except asyncio.CancelledError:
            raise

        except Exception as exc:  # noqa: BLE001 - a sweep must never crash the service
            log.error("incomplete-ingest sweep failed: %r", exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tiler Service API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    install_error_handlers(app)

    app.include_router(ingest_api.router, tags=["tiles"])
    app.include_router(tiles_api.router, tags=["tiles"])
    app.include_router(docs_api.router, tags=["docs"])
    app.include_router(metadata_api.router, tags=["metadata"])
    app.include_router(search_api.router, tags=["docs"])
    app.include_router(ops_api.router, tags=["ops"])

    return app


app = create_app()
