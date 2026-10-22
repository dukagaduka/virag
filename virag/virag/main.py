import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from virag.settings import Settings, SettingsError
from virag.errors import ApiError, install_error_handlers
from virag.clients.qdrant_store import QdrantStore
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import TilerService
from virag.clients.vlm_client import VLMClient
from virag.binding import enforce_binding, enforce_metadata_schema, secret_hash
from virag.reconcile import reconcile, run_periodic
from virag.revision import RevisionManager
from virag.api import enroll as enroll_api
from virag.api import chat_completions as chat_completions_api
from virag.api import proxy as proxy_api
from virag.api.proxy import QdrantProxyService
from virag.api import ops as ops_api
from virag.api import revision as revision_api


log = logging.getLogger("virag")


DESCRIPTION = "OpenAI-compatible /v1 proxy that enables pixel-RAG over Qdrant + doc-tiler on demand"


def configure_logging() -> None:
    """Send structured logs to stdout at the LOG_LEVEL from the environment."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def _require_dependency(name: str, coro) -> None:
    """Run a dependency check; a failure is fatal — the service is useless without it."""
    try:
        await coro
        log.info("dependency ok: %s", name)

    except ApiError as e:
        log.critical("dependency check failed (%s): %s | details=%s", name, e.message, e.details)
        raise


def _require_vector_size(dimension: int | None, vector_size: int) -> None:
    """The retriever's vectors must fit the collection, or the first upsert fails."""
    if dimension != vector_size:
        message = (
            f"retriever returns {dimension}-d vectors but QDRANT_VECTOR_SIZE is {vector_size}"
        )
        log.critical("FATAL: %s", message)

        raise SettingsError(message)


async def _cancel(task: asyncio.Task) -> None:
    """Cancel a background task and wait for it to unwind."""
    task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):

    configure_logging()

    settings = Settings.from_env()

    # Every client registers its own shutdown as soon as it exists, so a failed
    # dependency check below still closes whatever was already opened (LIFO).
    async with AsyncExitStack() as stack:
        store = QdrantStore.from_settings(settings.qdrant_settings)
        stack.push_async_callback(store.close)

        retriever = RetrieverClient.from_settings(settings.retriever_settings, settings.http_settings)
        stack.push_async_callback(retriever.close)

        tiler = TilerService.from_settings(settings.tiler_settings, settings.http_settings)
        stack.push_async_callback(tiler.close)

        vlm = VLMClient.from_settings(settings.vlm_settings, settings.http_settings)
        stack.push_async_callback(vlm.close)

        qdrant_http = QdrantProxyService.from_settings(settings.qdrant_settings, settings.http_settings)
        stack.push_async_callback(qdrant_http.close)

        # Every dependency is essential; a failed check refuses startup.
        await store.create_collection()
        await _require_dependency("retriever", retriever.check_connection())
        _require_vector_size(retriever.dimension, settings.qdrant_settings.vector_size)
        await _require_dependency("tiler", tiler.check_connection())
        await _require_dependency("vlm", vlm.check_connection())

        # Refuse to serve an unrelated Tiler/Qdrant pair; bind a fresh collection.
        render_version = await enforce_binding(store, tiler, secret_hash(settings.db_secret))
        # Reconcile depends on these indexed metadata fields being declared in the tiler.
        await enforce_metadata_schema(tiler)

        # Repair any partial/orphaned enrolls left by an earlier crash before serving.
        await reconcile(store, tiler, retriever, settings.reconcile_settings, settings.enroll_settings)

        # Registered last, so it is cancelled first — before the clients it uses close.
        reconcile_task = asyncio.create_task(
            run_periodic(store, tiler, retriever, settings.reconcile_settings, settings.enroll_settings)
        )
        stack.push_async_callback(_cancel, reconcile_task)

        app.state.settings = settings
        app.state.store = store
        app.state.retriever = retriever
        app.state.tiler = tiler
        app.state.vlm = vlm
        app.state.qdrant_http = qdrant_http

        revision = RevisionManager(
            store,
            tiler,
            retriever,
            settings.enroll_settings,
            settings.reconcile_settings,
        )
        stack.push_async_callback(revision.close)
        app.state.revision = revision

        log.info(
            "ready: qdrant=%s vector_size=%d collection=%s distance=%s retriever=%s tiler=%s vlm=%s model=%s render_version=%s",
            settings.qdrant_settings.base_url,
            settings.qdrant_settings.vector_size,
            settings.qdrant_settings.collection_name,
            settings.qdrant_settings.distance_type.value,
            settings.retriever_settings.base_url,
            settings.tiler_settings.base_url,
            settings.vlm_settings.base_url,
            settings.vlm_settings.model_name,
            render_version,
        )

        yield


def create_app() -> FastAPI:
    """Build the FastAPI app with error handlers and all routers mounted."""

    app = FastAPI(
        title="Virag Pixel-RAG Proxy",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    install_error_handlers(app)

    app.include_router(ops_api.router, tags=["ops"])
    app.include_router(enroll_api.router, tags=["enroll"])
    app.include_router(revision_api.router, tags=["revision"])
    app.include_router(chat_completions_api.router, tags=["v1"])
    app.include_router(proxy_api.router, tags=["proxy"])

    return app


app = create_app()
