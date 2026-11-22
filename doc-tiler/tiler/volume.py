"""Volume: the metadata DB and the blob directory, opened and bound to their geometry.

A volume is bound for life to the geometry it was created with, because
`tile_num` → pixel offsets are only meaningful under one geometry and tiles are
never persisted. A mismatch is fatal *before* serving anything, which is why there
is no per-request 410 anywhere in the API.

`Volume.open` is everything storage must pass before the first request. The service
runs it at startup, and so does `python -m tiler.preflight` before uvicorn is
launched, so a volume that cannot be served is one clear line in the log.
"""

from __future__ import annotations

import logging
import os

from .config import Settings
from .metadata import MetadataSchema, index_statements
from .store.blobs import BlobStore
from .store.meta import MetaStore, ServiceMeta

log = logging.getLogger("tiler.volume")


class GeometryMismatch(RuntimeError):
    """Fatal: the volume was built with a different geometry tier.

    Raised rather than `sys.exit()`-ing on the spot so the caller can release the
    DB connection first — a live aiosqlite thread would otherwise keep the
    interpreter from exiting at all, turning a fast fatal into a hang.
    """


    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class OutdatedVolume(GeometryMismatch):
    """Fatal: the volume predates columns the current schema needs.

    Old volumes are not migrated in place; subclassing GeometryMismatch makes the
    same startup catch sites treat it as fatal.
    """


class Volume:
    """An opened volume: its stores and its identity."""


    def __init__(self, store: MetaStore, blobs: BlobStore, meta: ServiceMeta):
        self.store = store
        self.blobs = blobs
        self.meta  = meta


    @property
    def volume_id(self) -> str:
        """The id the volume got when it was created."""
        return self.meta.volume_id


    @classmethod
    async def open(cls, settings: Settings, schema: MetadataSchema) -> Volume:
        """Open the volume the settings point at — a fresh one is initialised — and
        bring it in line with the configuration: its directories, its tables, its
        geometry (GeometryMismatch when it was built with another one, OutdatedVolume
        when by an older tiler), the indexes of the metadata schema."""
        paths = settings.store_settings

        os.makedirs(os.path.dirname(os.path.abspath(paths.db_path)) or ".", exist_ok=True)
        os.makedirs(paths.temp_dir, exist_ok=True)

        blobs = BlobStore(paths.blob_dir)
        blobs.ensure_root()

        store = MetaStore(paths.db_path)
        await store.connect()

        try:
            await store.create_tables()
            meta = await cls._bind(store, settings)

            created = await store.create_metadata_indexes(index_statements(schema))
            dropped = await store.drop_stale_metadata_indexes({f.index_name for f in schema.fields})

        except BaseException:
            await store.close()
            raise

        log.debug("index statements: %s", created)

        if dropped:
            log.info("dropped stale metadata index(es) %s", dropped)

        return cls(store, blobs, meta)


    async def drop_unfinished(self) -> None:
        """Drop what a killed process left half-done — at startup, when no ingest can
        be alive: no ingest outlives its process."""
        unfinished = await self.store.delete_incomplete()
        spooled = self.blobs.purge_incoming()

        if unfinished or spooled:
            log.warning(
                "removed %d unfinished ingest(s) and %d spooled upload(s) the previous run left behind",
                unfinished,
                spooled,
            )


    async def close(self) -> None:
        """Release the DB."""
        await self.store.close()


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _bind(store: MetaStore, settings: Settings) -> ServiceMeta:
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
        geometry = settings.geometry_settings.tier
        render_version = settings.render_version

        if row is None:
            meta = await store.init_service_meta(geometry, render_version)
            log.info(
                "initialised fresh volume %s: W=%d H=%d OVERLAP=%d MIN=%d render_version=%s",
                meta.volume_id,
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
