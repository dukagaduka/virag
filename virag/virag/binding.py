from __future__ import annotations

import logging

from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerClient


log = logging.getLogger("virag.binding")


class BindingError(RuntimeError):
    """Fatal: the Qdrant collection belongs to another tiler volume than this tiler's."""


async def enforce_binding(store: QdrantStore, tiler: TilerClient) -> str:
    """Refuse to run unless the Qdrant collection was embedded from this tiler's volume.

    The tiler gives each volume a random `volume_id` at init. A fresh (unbound)
    collection is bound to it, with the tiler render_version, on first run; another id
    later is fatal (BindingError) — a fresh or another volume behind TILER_BASE_URL would
    otherwise make every point look orphaned to reconcile. A render_version that
    differs later is not fatal (RevisionManager.revise_if_stale). Returns the tiler
    render_version.
    """
    info = await tiler.get_info()
    volume_id = info.get("volume_id")
    tiler_rv = info.get("render_version")

    if not volume_id:
        message = "the tiler reports no volume_id; it is older than this virag"
        log.critical("FATAL: %s", message)

        raise BindingError(message)

    meta = await store.get_collection_metadata()
    bound_volume_id = meta.get("tiler_volume_id")

    if bound_volume_id is None:
        await store.set_collection_metadata({"tiler_volume_id": volume_id, "render_version": tiler_rv})
        log.info("bound qdrant collection to tiler volume %s, render_version=%s", volume_id, tiler_rv)

        return tiler_rv

    if bound_volume_id != volume_id:
        message = (
            f"qdrant collection was embedded from tiler volume {bound_volume_id}, but the "
            f"tiler at TILER_BASE_URL serves volume {volume_id}: restore that volume, or "
            f"drop the collection and re-enroll"
        )
        log.critical("FATAL: %s", message)

        raise BindingError(message)

    return tiler_rv
