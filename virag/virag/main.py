import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from virag.settings import Settings
from virag.errors import ApiError, install_error_handlers
from virag.clients.qdrant_store import QdrantStore
from virag.clients.embeddings_client import EmbeddingsClient
from virag.clients.tiler_client import TilerService
from virag.clients.vlm_client import VLMClient
from virag.binding import enforce_binding, warn_doc_id_overlap, secret_hash
from virag.api import enroll as enroll_api
from virag.api import chat_completions as chat_completions_api
from virag.api import proxy as proxy_api
from virag.api import ops as ops_api


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


@asynccontextmanager
async def lifespan(app: FastAPI):

    configure_logging()

    settings = Settings.from_env()

    store = QdrantStore.from_settings(settings.qdrant_settings)
    embeddings = EmbeddingsClient.from_settings(settings.embeddings_settings)
    tiler = TilerService.from_settings(settings.tiler_settings)
    vlm = VLMClient.from_settings(settings.vlm_settings)

    # Every dependency is essential; a failed check refuses startup.
    await store.create_collection()
    await _require_dependency("embeddings", embeddings.check_connection())
    await _require_dependency("tiler", tiler.check_connection())
    await _require_dependency("vlm", vlm.check_connection())

    # Refuse to serve an unrelated Tiler/Qdrant pair; bind a fresh collection.
    render_version = await enforce_binding(store, tiler, secret_hash(settings.db_secret))
    await warn_doc_id_overlap(store, tiler)

    app.state.settings = settings
    app.state.store = store
    app.state.embeddings = embeddings
    app.state.tiler = tiler
    app.state.vlm = vlm

    log.info(
        "ready: qdrant=%s vector_size=%d collection=%s distance=%s embeddings=%s tiler=%s vlm=%s model=%s render_version=%s",
        settings.qdrant_settings.base_url,
        settings.qdrant_settings.vector_size,
        settings.qdrant_settings.collection_name,
        settings.qdrant_settings.distance_type.value,
        settings.embeddings_settings.base_url,
        settings.tiler_settings.base_url,
        settings.vlm_settings.base_url,
        settings.vlm_settings.model_name,
        render_version,
    )

    try:
        yield

    finally:
        await store.close()


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
    app.include_router(chat_completions_api.router, tags=["v1"])
    app.include_router(proxy_api.router, tags=["proxy"])

    return app


app = create_app()
