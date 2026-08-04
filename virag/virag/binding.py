from __future__ import annotations

import asyncio
import hashlib
import logging

from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService


log = logging.getLogger("virag.binding")

_SAMPLE_SIZE = 5


class BindingError(RuntimeError):
    """Fatal: the Tiler and Qdrant are not the bound, matching pair for this virag."""


def secret_hash(secret: str) -> str:
    """sha256 of the binding secret, or '' when unset (an empty, unbound slot)."""
    return hashlib.sha256(secret.encode()).hexdigest() if secret else ""


async def enforce_binding(
    store: QdrantStore,
    tiler: TilerService,
    expected_secret_hash: str,
) -> str:
    """Refuse to run unless virag, the Tiler and the Qdrant collection share one secret.

    Fatal (BindingError) on any secret mismatch. A fresh (unbound) collection is
    bound to the secret + tiler render_version on first run. Returns the tiler
    render_version.
    """
    info = await tiler.get_info()
    tiler_secret = info.get("secret_hash", "")
    tiler_rv = info.get("render_version")

    if tiler_secret != expected_secret_hash:
        message = (
            "tiler secret does not match TILER_DB_SECRET — the tiler is not bound to "
            "this virag (or was initialised without the secret)"
        )
        log.critical("FATAL: %s", message)

        raise BindingError(message)

    meta = await store.get_collection_metadata()
    stored_secret = meta.get("secret_hash")

    if stored_secret is None:
        await store.set_collection_metadata(
            {"secret_hash": expected_secret_hash, "render_version": tiler_rv}
        )
        log.info("bound qdrant collection to the secret + render_version=%s", tiler_rv)

        return tiler_rv

    if stored_secret != expected_secret_hash:
        message = (
            "qdrant collection secret does not match TILER_DB_SECRET — the collection "
            "belongs to a different virag/tiler pair"
        )
        log.critical("FATAL: %s", message)

        raise BindingError(message)

    if meta.get("render_version") != tiler_rv:
        log.warning(
            "render_version drift: collection bound to %s but tiler now reports %s; "
            "stored image_hash values may not reproduce",
            meta.get("render_version"),
            tiler_rv,
        )

    return tiler_rv


async def warn_doc_id_overlap(store: QdrantStore, tiler: TilerService) -> None:
    """Best-effort: sample a few enrolled doc_ids and warn if the tiler lacks them."""
    try:
        doc_ids = await store.sample_doc_ids(_SAMPLE_SIZE)

    except Exception as e:
        log.warning("could not sample doc_ids for the overlap check: %r", e)

        return

    if not doc_ids:
        return

    results = await asyncio.gather(
        *(tiler.doc_exists(doc_id) for doc_id in doc_ids),
        return_exceptions=True,
    )

    missing = [doc_id for doc_id, exists in zip(doc_ids, results) if exists is False]

    if missing:
        log.warning(
            "doc_id overlap: %d of %d sampled doc_ids are NOT in the tiler (%s); "
            "qdrant and the tiler may be out of sync",
            len(missing),
            len(doc_ids),
            missing,
        )
    else:
        log.info("doc_id overlap: %d sampled doc_ids present in the tiler", len(doc_ids))
