"""Enroll self-repair: keep Qdrant a consistent subset of the tiler.

The Reconciler runs at startup and (optionally) on an interval — every process tries,
the pass runs in the one leading it (`Coordinator.lead_reconcile`). Two passes:

  1. Enforce the `Qdrant ⊆ Tiler` invariant — delete any Qdrant points whose doc_id
     the tiler no longer has (a dangling hit would 404 on tile fetch).
  2. Finish abandoned enrolls — a doc marked pending (`TilerClient.pending_docs`) that
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
from virag.clients.tiler_client import TilerClient
from virag.coordination import Coordinator, EnrollInProgress
from virag.enroll import Enroller
from virag.settings import ReconcileSettings


log = logging.getLogger("virag.reconcile")


class Reconciler:
    """Keeps Qdrant a consistent subset of the tiler: one pass of both repairs (`run`),
    or the invariant alone on a collection a revision is about to serve
    (`enforce_subset`)."""

    def __init__(
        self,
        store: QdrantStore,
        coordinator: Coordinator,
        tiler: TilerClient,
        enroller: Enroller,
        reconcile_settings: ReconcileSettings,
    ):

        self.store       = store
        self.coordinator = coordinator
        self.tiler       = tiler
        self.enroller    = enroller

        self.self_heal = reconcile_settings.self_heal


    async def run(self) -> None:
        """Run one reconcile pass if this process leads reconcile: enforce the invariant,
        then finish abandoned enrolls."""
        if not await self.coordinator.lead_reconcile():
            log.debug("reconcile: led by another process")

            return

        await self.enforce_subset()
        await self.finish_abandoned()


    async def enforce_subset(self, *, collection: str | None = None) -> int:
        """Delete points whose doc_id is absent from the tiler (invariant repair) or
        quarantined by it, in the serving collection or in `collection` (a revision build).
        Returns how many docs."""
        qdrant_ids = await self.store.doc_ids(collection=collection)
        docs = await self.tiler.docs()

        # Read last, right before the delete: a document being enrolled now is left to its
        # enroll — the tiler lists it from `begin`, and a re-enroll lifts a quarantine.
        enrolling = set(await self.coordinator.enrolling_now())

        dangling = qdrant_ids - {doc.doc_id for doc in docs} - enrolling

        # A quarantined doc's tiles answer 410, so its points only crowd live hits out of top_k.
        quarantined = ({doc.doc_id for doc in docs if doc.quarantined} & qdrant_ids) - enrolling

        if dangling:
            log.warning("reconcile: %d dangling doc_id(s) in qdrant not in tiler; deleting", len(dangling))

        if quarantined:
            log.warning("reconcile: %d doc_id(s) quarantined by the tiler; deleting their points", len(quarantined))

        stale = dangling | quarantined
        if not stale:
            return 0

        await self.store.delete_by_docs(sorted(stale), collection=collection)

        return len(stale)


    async def finish_abandoned(self) -> None:
        """Finish the enrolls nobody is running any more: a pending doc the tiler serves is
        enrolled again from its stored tiles (self-heal); any other loses its partial points
        and waits for a re-enroll."""
        # Read before the listing: a doc pending there and absent here is abandoned, or was
        # taken since — and then taking it below is refused.
        enrolling = await self.coordinator.enrolling_now()

        pending = await self.tiler.pending_docs()
        abandoned = [doc for doc in pending if doc.doc_id not in enrolling]

        if not abandoned:
            return

        log.warning("reconcile: %d abandoned enroll(s)", len(abandoned))

        healed = 0

        for doc in abandoned:
            doc_id = doc.doc_id

            try:
                if self.self_heal and doc.embeddable:
                    await self.enroller.enroll_stored(doc_id)
                    healed += 1

                else:
                    # No servable tiles (the tiler doc is incomplete or quarantined), or
                    # self-heal is off: only a re-enroll restores it.
                    await self.enroller.drop(doc_id)
                    log.info("reconcile: doc_id=%s left for re-enroll", doc_id)

            except EnrollInProgress:
                log.info("reconcile: doc_id=%s is being enrolled; left to its enroll", doc_id)

            except Exception as e:  # noqa: BLE001 - one doc must not stop the pass; it stays pending
                log.error("reconcile: self-heal failed for doc_id=%s: %r", doc_id, e)

        log.info("reconcile: self-healed %d doc(s)", healed)
