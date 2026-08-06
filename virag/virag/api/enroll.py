from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile

from virag.errors import ApiError


log = logging.getLogger(__name__)

router = APIRouter()

_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "virag.pixelrag")


def point_id(doc_id: str, tile_num: int) -> str:
    """Deterministic point id so re-enrolling a document overwrites its own tiles."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{doc_id}:{tile_num}"))


@router.post(
    "/enroll",
    summary="Tile a document, embed its tiles, and upsert them into the vector store",
    operation_id="enrollDocument",
)
async def enroll(
    request: Request,
    doc_type: str = Form(...),
    document: UploadFile | None = File(None),
    url: str | None = Form(None),
    metadata: str | None = Form(None),
    force: bool = False,
    allow_missing_name: bool = False,
):

    store = request.app.state.store
    embeddings = request.app.state.embeddings
    tiler = request.app.state.tiler

    meta_obj = _parse_metadata(metadata)
    has_name = _has_name(meta_obj)

    # Service policy: every enrolled document is named. The opt-out is explicit so
    # the caller knowingly diverges — it names the document by its content hash.
    if not has_name and not allow_missing_name:
        raise ApiError(
            status_code=422,
            message="Each enrolled document must include metadata.name. "
                    "Set allow_missing_name=true to fall back to the document content hash.",
        )

    doc_bytes = await document.read() if document is not None else None

    filename = document.filename if document is not None else None
    log.info("enroll request: doc_type=%s filename=%s force=%s", doc_type, filename, force)

    # Streaming state, filled from the `begin` event and each `tile`.
    doc_id: str | None = None
    doc_type_out = source_url = render_version = payload_meta = None
    batch: list[dict] = []
    enrolled = 0

    async def flush() -> None:
        # Embed one batch of tiles and upsert it; the tiler already persisted these
        # tiles (write-then-send), so every point stays backed by a tile (Qdrant ⊆ Tiler).
        nonlocal enrolled
        if not batch:
            return

        vectors = await embeddings.embed_images([t["data"] for t in batch])
        payloads = [
            {
                "doc_id": doc_id,
                "tile_num": t["tile_num"],
                "image_hash": t["image_hash"],
                "doc_type": doc_type_out,
                "source_url": source_url,
                "render_version": render_version,
                "metadata": payload_meta,
            }
            for t in batch
        ]
        ids = [point_id(doc_id, t["tile_num"]) for t in batch]

        await store.upsert(vectors, payloads, ids=ids)
        enrolled += len(batch)
        batch.clear()

    try:
        async for event in tiler.tile_stream(
            doc_type, document=doc_bytes, url=url, metadata=meta_obj, force=force
        ):
            kind = event.get("event")

            if kind == "begin":
                doc_id = event["doc_id"]
                doc_type_out = event["doc_type"]
                source_url = event.get("source_url")
                render_version = event["render_version"]
                payload_meta = meta_obj if has_name else {**(meta_obj or {}), "name": doc_id}
                if not has_name:
                    log.warning("no metadata.name; naming doc_id=%s by its content hash", doc_id)

                # Mark pending (virag semantics on the tiler's indexed metadata) and clear
                # any prior partial/stale points. A crash from here leaves a clean, detectable
                # state that reconcile finishes or rolls back.
                await tiler.patch_metadata(doc_id, {"virag_started_at": int(time.time()), "virag_enrolled_at": 0})
                await store.delete_by_doc(doc_id)

            elif kind == "tile":
                batch.append(event)
                if len(batch) >= embeddings.chunk_size:
                    await flush()

            elif kind == "committed":
                await flush()
                # Promote: the enroll is durable end-to-end — clear pending.
                await tiler.patch_metadata(doc_id, {"virag_enrolled_at": int(time.time())})
                log.info("enrolled: doc_id=%s points=%d", doc_id, enrolled)

                return _summary(
                    doc_id, doc_type_out, render_version, event.get("tile_count", enrolled), enrolled
                )

            elif kind == "error":
                raise ApiError(
                    status_code=502,
                    message=f"tiler stream error: {event.get('message')}",
                    details={"tiler_code": event.get("code")},
                )

        raise ApiError(status_code=502, message="tiler stream ended without a committed event")

    except Exception:
        # Roll the doc back out of Qdrant. It stays in the tiler (Tiler ⊇ Qdrant) and
        # pending, so reconcile heals or sweeps it later.
        if doc_id is not None:
            try:
                await store.delete_by_doc(doc_id)
            except Exception as cleanup_err:  # noqa: BLE001 - never mask the original failure
                log.error("rollback delete_by_doc failed for doc_id=%s: %r", doc_id, cleanup_err)
        raise


def _has_name(meta_obj: dict | None) -> bool:
    """True iff metadata carries a non-empty string `name`."""
    name = meta_obj.get("name") if meta_obj else None

    return isinstance(name, str) and name.strip() != ""


def _parse_metadata(raw: str | None) -> dict | None:
    """Parse the metadata form field (a JSON object) into a dict, or None if absent."""
    if raw is None:
        return None

    try:
        obj = json.loads(raw)

    except (ValueError, TypeError) as e:
        raise ApiError(
            status_code=422,
            message="`metadata` must be a valid JSON object",
            details={"base_message": str(e)},
        ) from e

    if not isinstance(obj, dict):
        raise ApiError(
            status_code=422,
            message="`metadata` must be a JSON object",
        )

    return obj


def _summary(doc_id: str, doc_type: str, render_version: str, tile_count: int, enrolled: int) -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "render_version": render_version,
        "tile_count": tile_count,
        "enrolled": enrolled,
    }
