"""FastAPI app: startup wiring (the volume, the pools, the revision gate) and route registration."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__, metrics
from .api import docs as docs_api
from .api import ingest as ingest_api
from .api import metadata as metadata_api
from .api import ops as ops_api
from .api import revision as revision_api
from .api import search as search_api
from .api import tiles as tiles_api
from .api.deps import require_api_key, require_serving
from .capture import UrlCapture
from .config import ConfigError, Settings, configure_logging
from .errors import install_error_handlers
from .ingest import Ingester
from .meta.schema import SchemaError, load_schema
from .render.pool import RenderPool
from .render.readers import RenderParams
from .render.renderer import Renderer
from .responses import OrjsonResponse
from .revision import Revision
from .tiles import Tiles
from .volume import GeometryMismatch, Volume

log = logging.getLogger("tiler")

DESCRIPTION = (
    "Turns documents (PDF / URL / MHTML / IMG) into fixed-geometry image tiles "
    "(width=875, height<=1024) for a pixel-based RAG embedder, and regenerates any "
    "single tile on demand from the stored original."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        settings = Settings.from_env()
        schema = load_schema(settings.store_settings.metadata_schema_path)

    except (ConfigError, SchemaError) as exc:
        log.critical("FATAL: %s", exc)
        raise SystemExit(1) from exc

    render_version = settings.render_version

    # Everything below must succeed before the first request is served. Whatever is
    # started registers its own shutdown at once, so a failed step still closes what
    # was already opened (LIFO).
    async with AsyncExitStack() as stack:
        try:
            # The volume as the configuration wants it — or no start at all: never a
            # single request against a volume built with another geometry.
            volume = await Volume.open(settings, schema)

        except GeometryMismatch as exc:
            raise SystemExit(1) from exc

        stack.push_async_callback(volume.close)
        store, blobs = volume.store, volume.blobs

        await volume.drop_unfinished()

        log.info(
            "metadata schema v%s: %d searchable field(s) %s",
            schema.version,
            len(schema.fields),
            [f.key for f in schema.fields],
        )

        # Give the planner statistics; without them a selective range filter can lose
        # out to the created_at index (see README, "search and the query planner").
        await store.optimize()

        pool = RenderPool(
            settings.render_settings,
            RenderParams.from_settings(settings.geometry_settings, settings.pixel_settings),
            temp_dir=settings.store_settings.temp_dir,
        )
        renderer = Renderer(pool)
        capture = UrlCapture(settings.capture_settings, pool)

        tiles = Tiles(store, blobs, renderer, settings.render_settings)
        ingester = Ingester(store, blobs, renderer, capture, tiles, schema, render_version=render_version)
        revision = Revision(store, blobs, renderer, tiles, settings.revision_settings, render_version=render_version)

        # The SSRF guard proxy first: Chromium starts knowing whether it goes through one.
        await capture.start()
        stack.push_async_callback(capture.stop)

        await pool.start(browser=settings.render_settings.browser_enabled, per_context_proxy=capture.proxied)
        stack.push_async_callback(pool.stop)

        metrics.DOCS.set(await store.count_docs())

        # Opens the gate, or starts / awaits the revision of docs from another render_version.
        stack.push_async_callback(revision.close)
        await revision.prepare()

        app.state.settings = settings
        app.state.schema = schema
        app.state.volume_id = volume.volume_id
        app.state.store = store
        app.state.blobs = blobs
        app.state.pool = pool
        app.state.capture = capture
        app.state.tiles = tiles
        app.state.ingester = ingester
        app.state.revision = revision

        log.info(
            "%s: geometry=%s render_version=%s browsers=%d render_workers=%d",
            "ready" if revision.serving else f"not serving (revision {revision.status['state']})",
            settings.geometry_settings.tier,
            render_version,
            settings.render_settings.browsers,
            settings.render_settings.render_workers,
        )

        if not settings.api_settings.api_key:
            log.warning("TILER_API_KEY is empty: every route is open to anyone who reaches the port")

        if not settings.render_settings.render_worker_memory:
            log.warning(
                "TILER_RENDER_WORKER_MEMORY is 0: a heavy document can exhaust the container's "
                "memory and draw the OOM killer instead of failing on its own"
            )

        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tiler Service API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        default_response_class=OrjsonResponse,
    )
    install_error_handlers(app)

    # Every route needs TILER_API_KEY but the probes and metrics; data routes are also
    # gated by the revision, while ops and /revision itself always answer.
    keyed = [Depends(require_api_key)]
    serving = [*keyed, Depends(require_serving)]

    app.include_router(ingest_api.router, tags=["docs"], dependencies=serving)
    app.include_router(tiles_api.router, tags=["tiles"], dependencies=serving)
    app.include_router(docs_api.router, tags=["docs"], dependencies=serving)
    app.include_router(metadata_api.router, tags=["metadata"], dependencies=serving)
    app.include_router(search_api.router, tags=["docs"], dependencies=serving)
    app.include_router(revision_api.router, tags=["revision"], dependencies=keyed)
    app.include_router(ops_api.keyed_router, tags=["ops"], dependencies=keyed)
    app.include_router(ops_api.router, tags=["ops"])

    return app


app = create_app()
