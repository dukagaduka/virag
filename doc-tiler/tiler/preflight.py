"""Standalone startup validation: `python -m tiler.preflight`.

The entrypoint runs this before exec'ing uvicorn so a mismatched volume or a bad
config produces one clear log line and a non-zero exit — instead of a partially
started ASGI server reporting a generic startup failure.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import ConfigError, load_settings
from .guard import GeometryMismatch, enforce
from .main import configure_logging
from .meta.schema import SchemaError, index_statements, load_schema
from .store.meta import MetaStore

log = logging.getLogger("tiler.preflight")


async def _run() -> int:
    """Load config/schema, prepare storage, and enforce the geometry guard."""
    settings = load_settings()
    schema = load_schema(settings.metadata_schema_path)

    import os  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(settings.db_path)) or ".", exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)

    from .store.blobs import BlobStore  # noqa: PLC0415

    BlobStore(settings.blob_dir).ensure_root()

    store = MetaStore(settings.db_path)
    await store.connect()
    try:
        await store.create_tables()
        await enforce(store, settings)
        await store.create_metadata_indexes(index_statements(schema))
        docs = await store.count_docs()
    finally:
        await store.close()

    log.info(
        "preflight ok: geometry=%s render_version=%s docs=%d searchable=%s",
        settings.geometry,
        settings.render_version,
        docs,
        [f.key for f in schema.fields],
    )
    return 0


def main() -> int:
    """Run preflight, returning 0 on success or 1 on any fatal problem."""
    configure_logging()
    try:
        return asyncio.run(_run())
    except GeometryMismatch:
        return 1  # already logged at CRITICAL by the guard
    except (ConfigError, SchemaError) as exc:
        log.critical("FATAL: %s", exc)
        return 1
    except Exception as exc:  # storage/permission problems
        log.critical("FATAL: preflight failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
