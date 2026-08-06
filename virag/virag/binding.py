from __future__ import annotations

import hashlib
import logging

from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService


log = logging.getLogger("virag.binding")

# The indexed metadata fields virag relies on: `name` for search, and the two enroll
# markers reconcile queries by. All must be declared in the tiler's metadata schema
# (deploy/tiler_metadata_schema.yaml) or reconcile's search would 400.
REQUIRED_METADATA_FIELDS = {
    "name": "text",
    "virag_started_at": "number",
    "virag_enrolled_at": "number",
}


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


async def enforce_metadata_schema(tiler: TilerService) -> None:
    """Refuse to run unless the tiler declares every metadata field virag depends on.

    Fatal (BindingError): reconcile searches these keys, and an undeclared key would
    make the tiler reject the search — so a missing field is a misconfiguration, not a
    warning.
    """
    schema = await tiler.get_schema()
    declared = {f.get("key"): f.get("type") for f in schema.get("fields", [])}

    for key, expected_type in REQUIRED_METADATA_FIELDS.items():
        actual_type = declared.get(key)

        if actual_type is None:
            message = (
                f"tiler metadata schema is missing required field {key!r}; "
                f"declare it in tiler_metadata_schema.yaml"
            )
            log.critical("FATAL: %s", message)

            raise BindingError(message)

        if actual_type != expected_type:
            message = (
                f"tiler metadata field {key!r} is declared {actual_type!r} "
                f"but virag needs {expected_type!r}"
            )
            log.critical("FATAL: %s", message)

            raise BindingError(message)
