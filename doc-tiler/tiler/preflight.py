"""Standalone startup validation: `python -m tiler.preflight`.

The entrypoint runs this before exec'ing uvicorn so a mismatched volume or a bad
config produces one clear log line and a non-zero exit — instead of a partially
started ASGI server reporting a generic startup failure. It opens the volume exactly
as the service is about to (`Volume.open`), and closes it.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import ConfigError, Settings, configure_logging
from .metadata import SchemaError, load_schema
from .volume import GeometryMismatch, Volume

log = logging.getLogger("tiler.preflight")


def main() -> int:
    """Run preflight, returning 0 on success or 1 on any fatal problem."""
    configure_logging()

    try:
        return asyncio.run(_run())

    except GeometryMismatch:
        return 1  # already logged at CRITICAL where it was found

    except (ConfigError, SchemaError) as exc:
        log.critical("FATAL: %s", exc)
        return 1

    except Exception as exc:  # storage/permission problems
        log.critical("FATAL: preflight failed: %s: %s", type(exc).__name__, exc)
        return 1


async def _run() -> int:
    """Load the config and the schema, open the volume, report what is in it."""
    settings = Settings.from_env()
    schema = load_schema(settings.store_settings.metadata_schema_path)

    volume = await Volume.open(settings, schema)

    try:
        docs = await volume.store.count_docs()
        revisable = await volume.store.count_revisable(settings.render_version)

    finally:
        await volume.close()

    log.info(
        "preflight ok: geometry=%s render_version=%s docs=%d to_revise=%d searchable=%s",
        settings.geometry_settings.tier,
        settings.render_version,
        docs,
        revisable,
        [f.key for f in schema.fields],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
