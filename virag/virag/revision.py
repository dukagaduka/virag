"""Corpus revision: re-embed everything into a fresh collection and blue-green swap it in.

Runs as one background job per process (a second request gets 409). Writes that land
while it builds are not mirrored into the new collection; instead the store journals
the doc_ids of every writer (enroll, self-heal) active at the start or starting before
the swap, and the job syncs exactly those docs after the swap:

  1. Drop any leftover target, create `<serving>__vN+1`, start the journal.
  2. Re-embed every complete tiler doc into it (flat tile stream, failed docs retried).
     Docs still failing abort the job without a swap — the old collection keeps serving.
  3. Enforce `Qdrant ⊆ Tiler` on the new collection, then swap the alias. The replaced
     collection is retained as the rollback target; from now on writers hit the new one.
  4. Stop the journal and wait (up to the pending TTL) for writers of the old collection.
  5. Sync the journaled docs into the new collection, enforce the invariant once more.

A rollback repoints the alias to the retained collection, drops the abandoned one and
syncs the docs enrolled since that collection was retired.

The journal, lock and state live in this process: correct for a single virag replica.
"""

from __future__ import annotations

import asyncio
import logging
import time

from virag.clients.qdrant_store import QdrantStore
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import TilerService
from virag.errors import ApiError
from virag.reconcile import enforce_subset, reembed_docs, sync_docs
from virag.settings import EnrollSettings, ReconcileSettings


log = logging.getLogger("virag.revision")


class RevisionManager:
    """Owns the background revision/rollback job and its observable status."""

    def __init__(
        self,
        store: QdrantStore,
        tiler: TilerService,
        retriever: RetrieverClient,
        enroll_settings: EnrollSettings,
        reconcile_settings: ReconcileSettings,
    ):

        self.store     = store
        self.tiler     = tiler
        self.retriever = retriever

        self.batch_size    = enroll_settings.batch_size
        self.drain_timeout = reconcile_settings.pending_ttl_sec

        self._task: asyncio.Task | None = None
        self.status: dict = {"state": "idle"}


    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()


    def start(self) -> dict:
        """Launch a revision in the background; 409 while a job is already running."""
        self._launch("revision", self._revise())

        return self.status


    def start_rollback(self) -> dict:
        """Launch a rollback to the retained collection; 409 while a job is running."""
        if self.store.previous_name is None:
            raise ApiError(status_code=409, message="No previous collection to roll back to")

        self._launch("rollback", self._rollback())

        return self.status


    async def close(self) -> None:
        """Cancel a running job on shutdown; the next revision rebuilds from scratch."""
        if not self.running:
            return

        self._task.cancel()  # type: ignore[union-attr]

        try:
            await self._task  # type: ignore[misc]

        except asyncio.CancelledError:
            pass


    def _launch(self, kind: str, job) -> None:
        if self.running:
            job.close()

            raise ApiError(
                status_code=409,
                message=f"A {self.status['kind']} is already running",
                details={"status": self.status},
            )

        self.status = {"kind": kind, "state": "running", "started_at": int(time.time())}
        self._task = asyncio.create_task(self._guarded(job))


    async def _guarded(self, job) -> None:
        """Run a job and record its outcome; a failure never escapes the task."""
        try:
            await job

        except asyncio.CancelledError:
            self._finish("failed", error="cancelled (shutdown)")
            raise

        except Exception as e:  # noqa: BLE001 - surfaced through the status, not the loop
            log.exception("%s failed", self.status["kind"])
            self._finish("failed", error=repr(e))


    def _finish(self, state: str, **fields) -> None:
        self.status.update(fields, state=state, finished_at=int(time.time()))


    async def _revise(self) -> None:
        store = self.store

        target = await store.create_revision_collection()
        self.status["collection"] = target
        log.info("revision: building %s", target)

        store.start_journal()
        swapped = False

        try:
            docs = await self.tiler.iter_docs()
            complete = [doc for doc in docs if doc.get("complete")]

            # Incomplete docs aren't servable; leave them for a tiler revision / re-enroll.
            self.status.update(total=len(docs), skipped=len(docs) - len(complete))

            failed = await reembed_docs(
                store,
                self.tiler,
                self.retriever,
                complete,
                batch_size=self.batch_size,
                collection=target,
            )

            self.status["reembedded"] = len(complete) - len(failed)

            if failed:
                # A swap would silently drop these docs from search; keep the old one serving.
                self._finish(
                    "failed",
                    error="documents failed to re-embed; alias not swapped",
                    failed_doc_ids=sorted(failed),
                )

                return

            await enforce_subset(store, self.tiler, collection=target)

            old = store.physical_name
            await store.swap_alias(target)
            swapped = True
            log.info("revision: alias now → %s (previous %s retained)", target, old)

        finally:
            journal = store.stop_journal()

            if not swapped:
                try:
                    await store.drop_collection(target)

                except Exception as e:  # noqa: BLE001 - never mask the original outcome
                    log.error("revision: cannot drop aborted build %s: %r", target, e)

        if not await store.drain(old, self.drain_timeout):
            log.warning("revision: writers of %s still running after %ss; reconcile will pick them up", old, self.drain_timeout)

        sync_failed = await sync_docs(
            store,
            self.tiler,
            self.retriever,
            journal,
            batch_size=self.batch_size,
            collection=target,
        )

        await enforce_subset(store, self.tiler, collection=target)

        self.status["synced"] = len(journal) - len(sync_failed)
        log.info("revision: done, %d doc(s) synced after the swap", self.status["synced"])

        if sync_failed:
            # The swap happened; these docs need a re-enroll to come back into search.
            self._finish("degraded", sync_failed_doc_ids=sorted(sync_failed))

            return

        self._finish("succeeded")


    async def _rollback(self) -> None:
        store = self.store

        # Same journal as a revision: writers straddling the repoint wrote partly into
        # the collection being dropped.
        store.start_journal()

        try:
            restored, abandoned, retired_at = await store.rollback()

        finally:
            journal = store.stop_journal()

        self.status.update(collection=restored, abandoned=abandoned)
        log.warning("rollback: alias now → %s, dropped %s", restored, abandoned)

        if not await store.drain(abandoned, self.drain_timeout):
            log.warning("rollback: writers of %s still running after %ss; reconcile will pick them up", abandoned, self.drain_timeout)

        # Docs enrolled after the original swap exist only in the dropped collection.
        missed = journal | {
            doc["doc_id"]
            for doc in await self.tiler.iter_docs()
            if (doc.get("metadata") or {}).get("virag_enrolled_at", 0) >= retired_at
        }

        sync_failed = await sync_docs(
            store,
            self.tiler,
            self.retriever,
            missed,
            batch_size=self.batch_size,
        )

        await enforce_subset(store, self.tiler)

        self.status["synced"] = len(missed) - len(sync_failed)

        if sync_failed:
            self._finish("degraded", sync_failed_doc_ids=sorted(sync_failed))

            return

        self._finish("succeeded")
