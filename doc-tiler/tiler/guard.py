"""Startup geometry guard.

A DB/volume is bound for life to the geometry it was created with, because
`tile_num` → pixel offsets are only meaningful under one geometry and tiles are
never persisted. Mismatch is fatal *before* serving anything, which is why there
is no per-request 410 anywhere in the API.
"""

from __future__ import annotations

import logging

from .config import Settings
from .store.meta import MetaStore, ServiceMeta

log = logging.getLogger("tiler.guard")


class GeometryMismatch(RuntimeError):
    """Fatal: the volume was built with a different geometry tier.

    Raised rather than `sys.exit()`-ing on the spot so the caller can release the
    DB connection first — a live aiosqlite thread would otherwise keep the
    interpreter from exiting at all, turning a fast fatal into a hang.
    """


    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SecretMismatch(GeometryMismatch):
    """Fatal: the volume was bound to a different TILER_DB_SECRET (or none).

    Subclasses GeometryMismatch so the same startup catch sites treat it as fatal.
    The secret is bound at init like geometry and immutable for the volume's life.
    """


class OutdatedVolume(GeometryMismatch):
    """Fatal: the volume predates columns the current schema needs.

    Old volumes are not migrated in place; subclassing GeometryMismatch makes the
    same startup catch sites treat it as fatal.
    """


async def enforce(store: MetaStore, settings: Settings) -> ServiceMeta:
    """Initialise service_meta on a fresh volume, or fail on geometry drift.

    A volume older than the current schema raises OutdatedVolume and a geometry-tier
    mismatch raises GeometryMismatch (both fatal); a render_version change is the
    revision's to resolve (see tiler/revision.py) and only logged here.
    """
    missing = await store.missing_columns()

    if missing:
        message = (
            f"FATAL: volume was created by an older tiler and lacks {missing}; it is not "
            "migrated in place. Refusing to start. Point TILER_DB_PATH/TILER_BLOB_DIR at "
            "a fresh volume and re-ingest."
        )
        log.critical(message)

        raise OutdatedVolume(message)

    row = await store.get_service_meta()
    geometry = settings.geometry
    render_version = settings.render_version

    if row is None:
        meta = await store.init_service_meta(geometry, render_version, settings.secret_hash)
        log.info(
            "initialised fresh volume: W=%d H=%d OVERLAP=%d MIN=%d render_version=%s",
            geometry["tile_width"],
            geometry["tile_height"],
            geometry["tile_overlap"],
            geometry["min_tile_height"],
            render_version,
        )

        return meta

    stored = (row.tile_width, row.tile_height, row.tile_overlap, row.min_tile_height)
    current = (
        geometry["tile_width"],
        geometry["tile_height"],
        geometry["tile_overlap"],
        geometry["min_tile_height"],
    )

    if stored != current:
        message = (
            "FATAL: DB initialised with W=%d H=%d OVERLAP=%d MIN=%d; env has "
            "W=%d H=%d OVERLAP=%d MIN=%d. Refusing to start. Restore the original "
            "geometry, or point TILER_DB_PATH/TILER_BLOB_DIR at a fresh volume."
            % (stored + current)
        )
        log.critical(message)

        raise GeometryMismatch(message)

    if row.secret_hash != settings.secret_hash:
        message = (
            "FATAL: volume bound to a different TILER_DB_SECRET (or none) than the "
            "current env. Refusing to start. Restore the original secret, or point "
            "TILER_DB_PATH/TILER_BLOB_DIR at a fresh volume."
        )
        log.critical(message)

        raise SecretMismatch(message)

    if row.render_version != render_version:
        # Pixel tier: no serving until the stored docs are revised to the new version.
        log.warning(
            "render_version changed since the last revision (%s -> %s): pixel params "
            "(TILER_JPEG_QUALITY/TILER_PDF_DPI/Chromium/Pillow/PyMuPDF) differ; stored docs "
            "are revised before the service turns ready.",
            row.render_version,
            render_version,
        )

    return row
