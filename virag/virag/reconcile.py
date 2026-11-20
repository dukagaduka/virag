"""Enroll self-repair: keep Qdrant a consistent subset of the tiler.

Runs at startup and (optionally) on an interval — every process tries, the pass runs
in the one leading it (`Coordinator.lead_reconcile`). Two passes:

  1. Enforce the `Qdrant ⊆ Tiler` invariant — delete any Qdrant points whose doc_id
     the tiler no longer has (a dangling hit would 404 on tile fetch).
  2. Finish abandoned enrolls — a doc marked pending (`virag_enrolled_at == 0`) that
     nobody is enrolling was left by an enroll that failed or died; it is enrolled
     again from the tiler's stored tiles, or loses its partial points.

Neither pass touches a document that is being enrolled right now
(`Coordinator.enrolling`): it is its enroll's to finish. Because the collection is
bound to the tiler's volume and only virag holds TILER_API_KEY, every tiler doc carries
virag metadata, so there are no foreign docs to misjudge.
"""

from __future__ import annotations

import logging

from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService
from virag.coordination import Coordinator, EnrollInProgress
from virag.enroll import Enroller
from virag.settings import ReconcileSettings


log = logging.getLogger("virag.reconcile")


async def reconcile(
    store: QdrantStore,
    coordinator: Coordinator,
    tiler: TilerService,
    enroller: Enroller,
    settings: ReconcileSettings,
) -> None:
    """Run one reconcile pass if this process leads reconcile: enforce the invariant,
    then finish abandoned enrolls."""
    if not await coordinator.lead_reconcile():
        log.debug("reconcile: led by another process")

        return

    await enforce_subset(store, coordinator, tiler)
    await finish_abandoned(coordinator, tiler, enroller, settings)


async def enforce_subset(
    store: QdrantStore,
    coordinator: Coordinator,
    tiler: TilerService,
    *,
    collection: str | None = None,
) -> int:
    """Delete points whose doc_id is absent from the tiler (invariant repair) or
    quarantined by it, in the serving collection or in `collection` (a revision build).
    Returns how many docs."""
    qdrant_ids = await store.iter_doc_ids(collection=collection)
    docs = await tiler.iter_docs()

    # Read last, right before the delete: a document being enrolled now is left to its
    # enroll — the tiler lists it from `begin`, and a re-enroll lifts a quarantine.
    enrolling = set(await coordinator.enrolling_now())

    dangling = qdrant_ids - {doc["doc_id"] for doc in docs} - enrolling

    # A quarantined doc's tiles answer 410, so its points only crowd live hits out of top_k.
    quarantined = ({doc["doc_id"] for doc in docs if doc.get("quarantined")} & qdrant_ids) - enrolling

    if dangling:
        log.warning("reconcile: %d dangling doc_id(s) in qdrant not in tiler; deleting", len(dangling))

    if quarantined:
        log.warning("reconcile: %d doc_id(s) quarantined by the tiler; deleting their points", len(quarantined))

    stale = dangling | quarantined
    if not stale:
        return 0

    await store.delete_by_docs(sorted(stale), collection=collection)

    return len(stale)


async def finish_abandoned(
    coordinator: Coordinator,
    tiler: TilerService,
    enroller: Enroller,
    settings: ReconcileSettings,
) -> None:
    """Finish the enrolls nobody is running any more: a pending doc the tiler serves is
    enrolled again from its stored tiles (self-heal); any other loses its partial points
    and waits for a re-enroll."""
    # Read before the listing: a doc pending there and absent here is abandoned, or was
    # taken since — and then taking it below is refused.
    enrolling = await coordinator.enrolling_now()

    pending = await tiler.iter_docs(filters=[{"key": "virag_enrolled_at", "op": "eq", "value": 0}])
    abandoned = [doc for doc in pending if doc["doc_id"] not in enrolling]

    if not abandoned:
        return

    log.warning("reconcile: %d abandoned enroll(s)", len(abandoned))

    healed = 0

    for doc in abandoned:
        doc_id = doc["doc_id"]

        try:
            if settings.self_heal and embeddable(doc):
                await enroller.enroll_stored(doc_id)
                healed += 1

            else:
                # No servable tiles (the tiler doc is incomplete or quarantined), or
                # self-heal is off: only a re-enroll restores it.
                await enroller.drop(doc_id)
                log.info("reconcile: doc_id=%s left for re-enroll", doc_id)

        except EnrollInProgress:
            log.info("reconcile: doc_id=%s is being enrolled; left to its enroll", doc_id)

        except Exception as e:  # noqa: BLE001 - one doc must not stop the pass; it stays pending
            log.error("reconcile: self-heal failed for doc_id=%s: %r", doc_id, e)

    log.info("reconcile: self-healed %d doc(s)", healed)


def embeddable(doc: dict) -> bool:
    """A tiler doc whose tiles can be fetched: complete and not quarantined by a tiler
    revision (its tiles answer 410 until it is re-enrolled)."""
    return bool(doc.get("complete")) and not doc.get("quarantined")


def enrolled(doc: dict) -> bool:
    """A tiler doc an enroll has finished (not pending)."""
    return (doc.get("metadata") or {}).get("virag_enrolled_at", 0) > 0
