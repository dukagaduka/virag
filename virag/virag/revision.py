"""Corpus revision: re-embed everything into a fresh collection and blue-green swap it in.

One job at a time across every virag process: a job holds `revision:on` (the
Coordinator) and any process answers its status; a second request gets 409, and a job
whose process died is taken over by the next request. Writes that land while it builds
are not mirrored into the new collection; instead every document being enrolled (by
/enroll or by self-heal) at the start, or taken meanwhile, is journaled, and the job
syncs exactly those docs after the swap:

  1. Resolve the serving collection, drop any leftover target, create `<serving>__vN+1`,
     start the journal.
  2. Re-embed every complete tiler doc into it (`reembed`: a flat tile stream across
     documents, failed docs retried). Docs still failing abort the job without a swap —
     the old collection keeps serving.
  3. Enforce `Qdrant ⊆ Tiler` on the new collection, stamp it with the tiler's
     render_version, check the job is still this process's, then swap the alias. The
     replaced collection is retained as the rollback target; whoever takes a document
     from now on writes into the new one.
  4. Read the journal, wait (up to VIRAG_REVISION_DRAIN_TIMEOUT_SEC) for the enrolls in
     flight: one of them may still be writing into the old collection.
  5. Sync the journaled docs into the new collection — each enrolled again from the
     tiler's stored tiles, by the Enroller — and enforce the invariant once more.

A collection is bound to the render_version its tiles were embedded from (collection
metadata; every point carries it too, and chat refuses a tile the tiler now renders by
another one): a revision starts only while the tiler is serving, and aborts without a
swap if the tiler's render_version changed meanwhile. It is how virag follows a new
tiler render (by POST /revision, or at startup with VIRAG_REVISION_ON_RENDER_MISMATCH)
and how a new retriever model is rolled out.

A rollback repoints the alias to the retained collection, waits for the enrolls in
flight (they may be writing into the abandoned one), drops it and syncs the docs
enrolled since the retained one was retired;
it is refused when that collection was embedded from another render_version than the
tiler's.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time

from virag.clients.qdrant_store import QdrantStore
from virag.clients.retriever_client import RetrieverClient
from virag.clients.tiler_client import TilerService
from virag.coordination import Coordinator, EnrollInProgress
from virag.enroll import Enroller, point_id
from virag.errors import ApiError
from virag.reconcile import embeddable, enforce_subset, enrolled
from virag.settings import EnrollSettings, RevisionSettings
from virag.utils import chunked, gather


log = logging.getLogger("virag.revision")


class RevisionManager:
    """Runs this process's revision/rollback job; the job's lock and status live in the
    Coordinator, so every process answers for the cluster."""

    # Tiles assumed in flight when the tiler has no bulk concurrency cap; sizes the
    # window of a re-embed.
    UNCAPPED_TILES = 128


    def __init__(
        self,
        store: QdrantStore,
        coordinator: Coordinator,
        tiler: TilerService,
        retriever: RetrieverClient,
        enroller: Enroller,
        enroll_settings: EnrollSettings,
        revision_settings: RevisionSettings,
    ):

        self.store       = store
        self.coordinator = coordinator
        self.tiler       = tiler
        self.retriever   = retriever
        self.enroller    = enroller

        self.batch_size    = enroll_settings.batch_size
        self.drain_timeout = revision_settings.revision_drain_timeout_sec

        # This process's job, if it runs one; `status` is what it publishes.
        self._task: asyncio.Task | None = None
        self.status: dict = {}


    @property
    def running(self) -> bool:
        """Whether this process runs a job right now."""
        return self._task is not None and not self._task.done()


    async def get_status(self) -> dict:
        """The current or last job's status, whichever process runs it."""
        return await self.coordinator.get_status()


    async def start(self) -> dict:
        """Launch a revision in the background; 409 while a job is already running or
        while the tiler is not serving."""
        async def prepare():
            render_version = await self._tiler_render_version()

            return self._revise(render_version)

        return await self._launch("revision", prepare)


    async def start_rollback(self) -> dict:
        """Launch a rollback to the retained collection; 409 while a job is running, or
        when that collection was embedded from another render_version than the tiler's."""
        async def prepare():
            abandoned, restored = await self.store.resolve()
            if restored is None:
                raise ApiError(status_code=409, message="No previous collection to roll back to")

            render_version = await self._tiler_render_version()
            meta = await self.store.get_collection_metadata(collection=restored)

            if meta.get("render_version") != render_version:
                raise ApiError(
                    status_code=409,
                    message=f"Previous collection {restored} was embedded from "
                            f"render_version {meta.get('render_version')} but the tiler renders "
                            f"{render_version}; chat would answer RENDER_VERSION_MISMATCH",
                )

            return self._rollback(abandoned, restored)

        return await self._launch("rollback", prepare)


    async def revise_if_stale(self, *, auto: bool) -> None:
        """Startup check: a collection embedded from another render_version than the
        tiler's answers chat with 409 RENDER_VERSION_MISMATCH until it is rebuilt. With
        `auto` (VIRAG_REVISION_ON_RENDER_MISMATCH) the rebuild starts right here, unless
        another process already runs it; otherwise the mismatch is logged and left to
        POST /revision."""
        render_version = await self._tiler_render_version()
        embedded = (await self.store.get_collection_metadata()).get("render_version")

        if embedded == render_version:
            return

        log.warning(
            "render_version mismatch: the collection was embedded from %s but the tiler "
            "renders %s; chat answers RENDER_VERSION_MISMATCH until it is rebuilt (%s)",
            embedded,
            render_version,
            "starting the revision" if auto else "POST /revision",
        )

        if not auto:
            return

        try:
            await self.start()

        except ApiError as e:
            if e.status_code != 409:
                raise

            log.info("revision: not started here: %s", e.message)


    async def close(self) -> None:
        """Cancel this process's job on shutdown; the next revision rebuilds from scratch."""
        if not self.running:
            return

        self._task.cancel()  # type: ignore[union-attr]

        try:
            await self._task  # type: ignore[misc]

        except asyncio.CancelledError:
            pass


    async def reembed(self, docs: list[dict], collection: str) -> set[str]:
        """Re-embed complete tiler docs into `collection`, a build nobody else writes;
        docs that fail are retried once in per-document batches, so a doc that merely
        shared a batch with a broken one recovers. Returns the doc_ids that still failed."""
        failed = await self._reembed_pass(docs, collection)

        if not failed:
            return failed

        log.warning("reembed: retrying %d failed doc(s)", len(failed))

        return await self._reembed_pass([doc for doc in docs if doc["doc_id"] in failed], collection, per_doc=True)


    # ------------------------------------------------------------------ #
    # the job
    # ------------------------------------------------------------------ #
    async def _tiler_render_version(self) -> str:
        """The tiler's render_version. Refused (409) while the tiler is not serving: its
        docs are being revised, so their tiles are not yet the ones it settles on."""
        info = await self.tiler.get_info()
        tiler_revision = info.get("revision") or {}

        if not tiler_revision.get("serving", True):
            raise ApiError(
                status_code=409,
                message=f"The tiler is not serving (its revision is {tiler_revision.get('state')}); "
                        f"retry once it is",
            )

        return info.get("render_version")


    async def _launch(self, kind: str, prepare) -> dict:
        """Take the job (409 while a live process holds it), run `prepare` under it — its
        checks; a refusal hands the job back — and run the job it returns in the background."""
        holder = await self.coordinator.acquire_revision()
        if holder is not None:
            raise ApiError(
                status_code=409,
                message=f"A {holder.get('kind', 'revision')} is already running",
                details={"status": holder},
            )

        job = None

        try:
            job = await prepare()

            self.status = {
                "kind": kind,
                "state": "running",
                "started_at": int(time.time()),
                "owner": self.coordinator.identity,
            }
            await self.coordinator.set_status(self.status)

        except BaseException:
            if job is not None:
                job.close()                                         # never started

            await self.coordinator.release_revision()
            raise

        self._task = asyncio.create_task(self._guarded(job))

        return self.status


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

        finally:
            try:
                await self.coordinator.set_status(self.status)
                await self.coordinator.release_revision()

            except Exception as e:  # noqa: BLE001 - the job's own outcome is already logged
                log.error("%s: cannot publish the outcome or release the job: %r", self.status["kind"], e)


    async def _update(self, **fields) -> None:
        """Record progress and publish it."""
        self.status.update(fields)
        await self.coordinator.set_status(self.status)


    def _finish(self, state: str, **fields) -> None:
        self.status.update(fields, state=state, finished_at=int(time.time()))


    async def _revise(self, render_version: str) -> None:
        store = self.store
        coordinator = self.coordinator

        old, stale_previous = await store.resolve()

        target = await store.create_revision_collection(old)
        await self._update(collection=target, render_version=render_version, phase="building")
        log.info("revision: building %s from render_version %s", target, render_version)

        await coordinator.start_journal()
        swapped = False

        try:
            docs = await self.tiler.iter_docs()
            complete = [doc for doc in docs if embeddable(doc)]

            # Incomplete or quarantined docs have no servable tiles; leave them for a re-enroll.
            await self._update(total=len(docs), skipped=len(docs) - len(complete))

            failed = await self.reembed(complete, target)

            self.status["reembedded"] = len(complete) - len(failed)

            if failed:
                # A swap would silently drop these docs from search; keep the old one serving.
                self._finish(
                    "failed",
                    error="documents failed to re-embed; alias not swapped",
                    failed_doc_ids=sorted(failed),
                )

                return

            await enforce_subset(store, coordinator, self.tiler, collection=target)

            # The vectors must match what the tiler renders now, not when the job began.
            current = await self._tiler_render_version()
            if current != render_version:
                self._finish(
                    "failed",
                    error=f"the tiler's render_version changed during the build "
                          f"({render_version} → {current}); alias not swapped",
                )

                return

            await store.set_collection_metadata({"render_version": render_version}, collection=target)

            # Fencing: a process whose lease expired may already have been replaced by another.
            if not await coordinator.owns_revision():
                self._finish("failed", error="the job was lost (this process's lease expired); alias not swapped")

                return

            await store.swap_alias(target, old, stale_previous)
            swapped = True
            log.info("revision: alias now → %s (previous %s retained)", target, old)

        finally:
            if not swapped:
                try:
                    await store.drop_collection(target)

                except Exception as e:  # noqa: BLE001 - never mask the original outcome
                    log.error("revision: cannot drop aborted build %s: %r", target, e)

        journal = await coordinator.journal()
        await self._update(phase="syncing")

        if not await coordinator.drain(self.drain_timeout):
            log.warning("revision: enrolls that may be writing into %s still running after %ss", old, self.drain_timeout)

        sync_failed = await self._sync(journal)

        await enforce_subset(store, coordinator, self.tiler, collection=target)

        self.status["synced"] = len(journal) - len(sync_failed)
        log.info("revision: done, %d doc(s) synced after the swap", self.status["synced"])

        if sync_failed:
            # The swap happened; these docs stay pending, for reconcile to enroll again.
            self._finish("degraded", sync_failed_doc_ids=sorted(sync_failed))

            return

        self._finish("succeeded")


    async def _rollback(self, abandoned: str, restored: str) -> None:
        store = self.store
        coordinator = self.coordinator

        # Same journal as a revision: enrolls straddling the repoint wrote into the
        # collection being dropped.
        await coordinator.start_journal()

        if not await coordinator.owns_revision():
            self._finish("failed", error="the job was lost (this process's lease expired); alias not repointed")

            return

        retired_at = await store.rollback_alias(restored)
        journal = await coordinator.journal()

        await self._update(collection=restored, abandoned=abandoned, phase="syncing")
        log.warning("rollback: alias now → %s; dropping %s once the enrolls in flight are done", restored, abandoned)

        if not await coordinator.drain(self.drain_timeout):
            log.warning("rollback: enrolls that may be writing into %s still running after %ss", abandoned, self.drain_timeout)

        await store.drop_collection(abandoned)

        # Docs enrolled after the original swap exist only in the dropped collection.
        missed = journal | {
            doc["doc_id"]
            for doc in await self.tiler.iter_docs()
            if (doc.get("metadata") or {}).get("virag_enrolled_at", 0) >= retired_at
        }

        sync_failed = await self._sync(missed)

        await enforce_subset(store, coordinator, self.tiler, collection=restored)

        self.status["synced"] = len(missed) - len(sync_failed)

        if sync_failed:
            self._finish("degraded", sync_failed_doc_ids=sorted(sync_failed))

            return

        self._finish("succeeded")


    async def _sync(self, doc_ids: set[str]) -> set[str]:
        """Make the serving collection mirror the tiler for these docs, each through the
        Enroller: one the tiler holds complete and enrolled is enrolled again from its
        stored tiles; any other — gone, incomplete, quarantined, or pending — loses its
        points, exactly as the serving collection treats it. A doc being enrolled right
        now is left to its enroll, which writes it whole. Returns the doc_ids that failed."""
        if not doc_ids:
            return set()

        docs = {doc["doc_id"]: doc for doc in await self.tiler.iter_docs()}
        failed: set[str] = set()

        for doc_id in sorted(doc_ids):
            doc = docs.get(doc_id)

            try:
                if doc is not None and embeddable(doc) and enrolled(doc):
                    await self.enroller.enroll_stored(doc_id)

                else:
                    await self.enroller.drop(doc_id)

            except EnrollInProgress:
                log.info("%s: doc_id=%s is being enrolled; left to its enroll", self.status["kind"], doc_id)

            except Exception as e:  # noqa: BLE001 - a failed doc fails itself, not the job
                failed.add(doc_id)
                log.error("%s: cannot sync doc_id=%s: %r", self.status["kind"], doc_id, e)

        return failed


    # ------------------------------------------------------------------ #
    # the flat re-embed
    # ------------------------------------------------------------------ #
    async def _reembed_pass(self, docs: list[dict], collection: str, per_doc: bool = False) -> set[str]:
        """One pass of the flat tile stream.

        Every (doc_id, tile_num) of every doc is cut into batches of `batch_size` across
        document boundaries, so small docs share full batches. Each batch fetches its
        tiles concurrently (bulk, capped by the tiler's `max_concurrency`), embeds them in
        one bulk retriever call and upserts once. A window of batches stays in flight, so
        one batch fetches while another embeds; it is derived from the tiler cap — more
        would only queue at the guard. Point ids match enroll's (`point_id`), so this
        writes exactly what a fresh enroll would.

        A failed batch fails every doc it touched (their later tiles are skipped).
        `per_doc` keeps batches within one document (the retry pass).
        """
        by_id = {doc["doc_id"]: doc for doc in docs}
        failed: set[str] = set()

        async def process(batch: list[tuple[str, int]]) -> None:
            batch = [(doc_id, n) for doc_id, n in batch if doc_id not in failed]
            if not batch:
                return

            try:
                jpegs = await gather(self.tiler.get_tile(doc_id, n, bulk=True) for doc_id, n in batch)
                b64s = [base64.b64encode(j).decode("ascii") for j in jpegs]
                vectors = await self.retriever.embed_images(b64s, bulk=True)

                payloads = [
                    {
                        "doc_id": doc_id,
                        "tile_num": n,
                        "image_hash": hashlib.sha256(j).hexdigest(),
                        "render_version": by_id[doc_id].get("render_version") or "",
                    }
                    for (doc_id, n), j in zip(batch, jpegs)
                ]
                ids = [point_id(doc_id, n) for doc_id, n in batch]

                await self.store.upsert(vectors, payloads, ids=ids, collection=collection)

            except Exception as e:  # noqa: BLE001 - a failed batch fails its docs, not the stream
                touched = {doc_id for doc_id, _ in batch}
                failed.update(touched)
                log.error("reembed: batch failed for doc(s) %s: %r", sorted(touched), e)

        if per_doc:
            batches = (
                batch
                for doc_id, doc in by_id.items()
                for batch in chunked(((doc_id, n) for n in range(doc["tile_count"])), self.batch_size)
            )
        else:
            tiles = ((doc_id, n) for doc_id, doc in by_id.items() for n in range(doc["tile_count"]))
            batches = chunked(tiles, self.batch_size)

        window = max(1, (self.tiler.max_concurrency or self.UNCAPPED_TILES) // self.batch_size) + 1
        slots = asyncio.Semaphore(window)

        async def run(batch: list[tuple[str, int]]) -> None:
            try:
                await process(batch)

            finally:
                slots.release()

        async with asyncio.TaskGroup() as group:
            for batch in batches:
                await slots.acquire()
                group.create_task(run(batch))

        return failed
