"""FastAPI app: startup wiring (guard, DB, metadata indexes, pools, revision gate) and route registration."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__, guard, metrics
from .api import docs as docs_api
from .api import ingest as ingest_api
from .api import metadata as metadata_api
from .api import ops as ops_api
from .api import revision as revision_api
from .api import search as search_api
from .api import tiles as tiles_api
from .api.deps import require_api_key, require_serving
from .capture import UrlCapture
from .config import ConfigError, load_settings
from .errors import install_error_handlers
from .ingest import Ingester
from .meta.schema import SchemaError, index_statements, load_schema
from .render.pool import RenderPool
from .responses import OrjsonResponse
from .revision import Revision
from .store.blobs import BlobStore
from .store.meta import MetaStore
from .tiles import Tiles

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    try:
        settings = load_settings()
        schema = load_schema(settings.metadata_schema_path)

    except (ConfigError, SchemaError) as exc:
        log.critical("FATAL: %s", exc)
        raise SystemExit(1) from exc

    os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)) or ".", exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)

    store = MetaStore(settings.db_path)
    blobs = BlobStore(settings.blob_dir)
    render = RenderPool(settings)
    capture = UrlCapture(settings, render)

    tiles = Tiles(store, blobs, render, settings)
    ingester = Ingester(store, blobs, render, capture, tiles, schema, settings)
    revision = Revision(store, blobs, render, tiles, settings)

    # Everything below must succeed before the first request is served. Whatever is
    # started registers its own shutdown at once, so a failed step still closes what
    # was already opened (LIFO).
    async with AsyncExitStack() as stack:
        blobs.ensure_root()

        await store.connect()
        stack.push_async_callback(store.close)

        await store.create_tables()

        try:
            # Hard geometry gate: refuse to start against a volume built with different geometry.
            volume_id = (await guard.enforce(store, settings)).volume_id

        except guard.GeometryMismatch as exc:
            # Never serve a single request against a mismatched volume.
            raise SystemExit(1) from exc

        # No ingest outlives its process: drop what a killed one left half-done.
        unfinished = await store.delete_incomplete()
        spooled = blobs.purge_incoming()

        if unfinished or spooled:
            log.warning(
                "removed %d unfinished ingest(s) and %d spooled upload(s) the previous run left behind",
                unfinished,
                spooled,
            )

        created = await store.create_metadata_indexes(index_statements(schema))
        dropped = await store.drop_stale_metadata_indexes({f.index_name for f in schema.fields})

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
        await store.optimize()

        # The SSRF guard proxy first: Chromium starts knowing whether it goes through one.
        await capture.start()
        stack.push_async_callback(capture.stop)

        await render.start(browser=settings.browser_enabled, per_context_proxy=capture.proxied)
        stack.push_async_callback(render.stop)

        metrics.DOCS.set(await store.count_docs())

        # Opens the gate, or starts / awaits the revision of docs from another render_version.
        stack.push_async_callback(revision.close)
        await revision.prepare()

        app.state.settings = settings
        app.state.schema = schema
        app.state.volume_id = volume_id
        app.state.store = store
        app.state.blobs = blobs
        app.state.render = render
        app.state.capture = capture
        app.state.tiles = tiles
        app.state.ingester = ingester
        app.state.revision = revision

        log.info(
            "%s: geometry=%s render_version=%s browsers=%d render_workers=%d",
            "ready" if revision.serving else f"not serving (revision {revision.status['state']})",
            settings.geometry,
            settings.render_version,
            settings.browsers,
            settings.render_workers,
        )

        if not settings.api_key:
            log.warning("TILER_API_KEY is empty: every route is open to anyone who reaches the port")

        if not settings.render_worker_memory:
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

    app.include_router(ingest_api.router, tags=["tiles"], dependencies=serving)
    app.include_router(tiles_api.router, tags=["tiles"], dependencies=serving)
    app.include_router(docs_api.router, tags=["docs"], dependencies=serving)
    app.include_router(metadata_api.router, tags=["metadata"], dependencies=serving)
    app.include_router(search_api.router, tags=["docs"], dependencies=serving)
    app.include_router(revision_api.router, tags=["revision"], dependencies=keyed)
    app.include_router(ops_api.keyed_router, tags=["ops"], dependencies=keyed)
    app.include_router(ops_api.router, tags=["ops"])

    return app


app = create_app()
