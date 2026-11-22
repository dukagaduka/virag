"""Revision: bring every stored doc to the current render_version before serving.

`render_version` is bound to the volume like geometry: the tiler never serves a doc
rendered by another one, so a regenerated tile always reproduces its stored
`image_hash`. When startup finds docs from another render_version (a pixel-tier
change: Chromium, Pillow, PyMuPDF, DPI, JPEG quality, PDF slicing), the service comes
up not-ready — every data route answers 503 NOT_READY — and:

  - with TILER_REVISION_ON_RENDER_MISMATCH (default) the revision starts at once;
  - without it, the service waits for POST /revision.

The job:

  1. Backs the DB up once per source render_version (`<db>.bak-<rv>`): the previous
     image plus this backup is the rollback.
  2. Re-renders the revisable docs from their blobs, TILER_REVISION_CONCURRENCY at a
     time, keyset-paged by doc_id; each doc's tiles and layout are swapped in one
     transaction, its metadata untouched.
  3. Retries a failed doc once; one still failing is quarantined (its tiles answer
     410) rather than holding the whole service back.
  4. Turns the service ready once nothing revisable is left.

An infrastructure failure (render workers that cannot start, Chromium unavailable)
aborts the job instead of quarantining every doc after it, and the service stays
not-ready. A document that crashes a worker, exhausts its memory or times out is that
document's failure (RENDER_CRASHED, RENDER_TOO_LARGE, RENDER_TIMEOUT) and is quarantined. The
selection is by render_version, so a restart resumes where a job stopped (a forced
revision is not resumed). GET /revision reports progress.

The gate and the job live in this process: correct for the single tiler instance.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator

from . import metrics
from .config import RevisionSettings
from .errors import ApiError, conflict
from .render.renderer import Renderer
from .store.blobs import BlobStore
from .store.meta import DocRow, MetaStore, TileRow
from .tiles import Tiles

log = logging.getLogger("tiler.revision")


class Revision:
    """Owns the serving gate and the background revision job with its status."""

    PAGE = 100  # docs per keyset page
    PROGRESS_EVERY_SEC = 30  # log cadence of the done/total line
    MAX_LISTED = 100  # quarantined doc_ids kept in the status
    # Failures of the pools themselves: every doc after them would fail the same way.
    INFRASTRUCTURE = frozenset({"WORKER_CRASH", "BROWSER_UNAVAILABLE"})


    def __init__(
        self,
        store: MetaStore,
        blobs: BlobStore,
        renderer: Renderer,
        tiles: Tiles,
        settings: RevisionSettings,
        *,
        render_version: str,
    ):

        self.store    = store
        self.blobs    = blobs
        self.renderer = renderer
        self.tiles    = tiles

        self.render_version = render_version
        self.on_mismatch    = settings.revision_on_render_mismatch
        self.concurrency    = settings.revision_concurrency

        self.serving = False
        self._task: asyncio.Task | None = None
        self._status: dict[str, Any] = {"state": "idle"}
        self._started = 0.0
        self._reported = 0.0


    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()


    @property
    def status(self) -> dict[str, Any]:
        """Observable state of the gate and of the current or last job."""
        return {**self._status, "serving": self.serving}


    async def prepare(self) -> None:
        """At startup: serve right away, start the revision, or wait for POST /revision."""
        stale = await self.store.count_revisable(self.render_version)
        self._status = {
            "state": "idle",
            "render_version": self.render_version,
            "quarantined_total": await self.store.count_quarantined(),
        }

        if not stale:
            await self._open()
            return

        log.warning(
            "%d doc(s) were rendered by another render_version than %s; not serving until "
            "they are revised",
            stale,
            self.render_version,
        )

        if self.on_mismatch:
            self.start()
            return

        self._status.update(state="required", total=stale)
        log.warning("TILER_REVISION_ON_RENDER_MISMATCH is off; waiting for POST /revision")


    def start(self, force: bool = False) -> dict[str, Any]:
        """Launch a revision in the background (the service turns not-ready); 409 while
        one is already running."""
        if self.running:
            raise conflict(
                "a revision is already running", "REVISION_RUNNING", status=self.status
            )

        self.serving = False
        metrics.SERVING.set(0)

        self._started = self._reported = time.monotonic()
        self._status = {
            "state": "running",
            "render_version": self.render_version,
            "force": force,
            "started_at": int(time.time()),
            "total": None,
            "revised": 0,
            "quarantined": 0,
            "quarantined_doc_ids": [],
            "quarantined_total": self._status.get("quarantined_total", 0),
        }
        self._task = asyncio.create_task(self._guarded(self._revise(force)))

        return self.status


    async def close(self) -> None:
        """Cancel a running job on shutdown; the next start resumes it."""
        if not self.running:
            return

        self._task.cancel()  # type: ignore[union-attr]

        try:
            await self._task  # type: ignore[misc]

        except asyncio.CancelledError:
            pass


    async def _guarded(self, job) -> None:
        """Run the job and record its outcome; a failure never escapes the task."""
        try:
            await job

        except asyncio.CancelledError:
            self._finish("failed", error="cancelled (shutdown)")
            raise

        except ApiError as exc:
            log.error("revision aborted: %s: %s", exc.code, exc.message)
            self._finish("failed", error=f"{exc.code}: {exc.message}")

        except Exception as exc:  # noqa: BLE001 - surfaced through the status, not the loop
            log.exception("revision failed")
            self._finish("failed", error=repr(exc))


    def _finish(self, state: str, **fields: Any) -> None:
        self._status.update(fields, state=state, finished_at=int(time.time()))


    async def _revise(self, force: bool) -> None:
        total = await self.store.count_revisable(self.render_version, force=force)
        self._status["total"] = total
        log.info(
            "revision: %d doc(s) to re-render at render_version %s (force=%s)", total, self.render_version, force
        )

        await self._backup()

        slots = asyncio.Semaphore(self.concurrency)

        async def run(row: DocRow) -> None:
            try:
                await self._revise_one(row)

            finally:
                slots.release()
                self._progress()

        try:
            async with asyncio.TaskGroup() as group:
                async for row in self._candidates(force):
                    await slots.acquire()
                    group.create_task(run(row))

        except BaseExceptionGroup as eg:
            raise eg.exceptions[0]

        remaining = await self.store.count_revisable(self.render_version)
        quarantined = await self.store.count_quarantined()

        self._status["quarantined_total"] = quarantined
        metrics.QUARANTINED.set(quarantined)

        if remaining:  # pragma: no cover - the selection drained, so only a concurrent writer
            self._finish("failed", error=f"{remaining} doc(s) still at another render_version")
            return

        log.info(
            "revision: done in %.0fs — revised=%d quarantined=%d",
            time.monotonic() - self._started,
            self._status["revised"],
            self._status["quarantined"],
        )
        self._finish("succeeded")
        await self._open()


    async def _candidates(self, force: bool) -> AsyncIterator[DocRow]:
        """Every revisable doc, page by page in doc_id order."""
        after = ""

        while rows := await self.store.list_revisable(
            self.render_version, force=force, after=after, limit=self.PAGE
        ):
            for row in rows:
                yield row

            after = rows[-1].doc_id


    async def _revise_one(self, row: DocRow) -> None:
        """Re-render one doc, retrying once; quarantine it if it still fails."""
        for attempt in (1, 2):
            try:
                await self._revise_doc(row)

            except ApiError as exc:
                if exc.code in self.INFRASTRUCTURE:
                    raise

                reason = f"{exc.code}: {exc.message}"

            except Exception as exc:  # noqa: BLE001 - a broken doc must not stop the job
                reason = repr(exc)

            else:
                self._status["revised"] += 1
                metrics.REVISION_DOCS.labels(result="revised").inc()
                return

            log.warning("revision: doc_id=%s failed (attempt %d): %s", row.doc_id, attempt, reason)

        await self.store.quarantine(row.doc_id, self.render_version, reason)

        self._status["quarantined"] += 1
        metrics.REVISION_DOCS.labels(result="quarantined").inc()

        if len(self._status["quarantined_doc_ids"]) < self.MAX_LISTED:
            self._status["quarantined_doc_ids"].append(row.doc_id)

        log.error("revision: doc_id=%s quarantined: %s", row.doc_id, reason)


    async def _revise_doc(self, row: DocRow) -> None:
        """Re-render one stored doc from its blob and swap its tiles and height in place
        at the current render_version (doc_id and metadata are untouched)."""
        self.blobs.require(row.doc_id)

        async with self.renderer.open(row.doc_type, self.blobs.path(row.doc_id), purpose="revise") as doc:
            staged = [TileRow.from_rendered(row.doc_id, tile) async for tile in doc.tiles()]
            height = doc.height

        await self.store.replace_render(
            row.doc_id,
            height=height,
            tile_count=len(staged),
            render_version=self.render_version,
            tiles=staged,
        )
        await self.tiles.forget(row.doc_id)


    async def _backup(self) -> None:
        """Copy the DB aside once per source render_version, before the first write."""
        meta = await self.store.get_service_meta()
        path = f"{self.store.db_path}.bak-{meta.render_version}"

        if os.path.exists(path):
            log.info("revision: keeping the existing backup %s", path)

        else:
            # Via a temp file: a crash mid-copy must not leave a truncated "backup".
            partial = f"{path}.partial"

            if os.path.exists(partial):
                os.remove(partial)

            await self.store.backup(partial)
            os.replace(partial, path)
            log.info("revision: backed up the DB to %s", path)

        self._status["backup"] = path


    def _progress(self) -> None:
        """Refresh the ETA and log done/total, rate and ETA every PROGRESS_EVERY_SEC."""
        total = self._status["total"] or 0
        done = self._status["revised"] + self._status["quarantined"]
        now = time.monotonic()
        rate = done / max(now - self._started, 1e-9)
        eta = (total - done) / rate if rate else None

        self._status["eta_sec"] = round(eta) if eta is not None else None

        if now - self._reported < self.PROGRESS_EVERY_SEC:
            return

        self._reported = now
        log.info(
            "revision: %d/%d docs, %.2f docs/s, ETA %s",
            done,
            total,
            rate,
            f"{eta:.0f}s" if eta is not None else "?",
        )


    async def _open(self) -> None:
        """Nothing revisable is left: stamp the volume and start serving."""
        meta = await self.store.get_service_meta()

        if meta is not None and meta.render_version != self.render_version:
            await self.store.update_render_version(self.render_version)

        metrics.QUARANTINED.set(self._status.get("quarantined_total", 0))
        metrics.SERVING.set(1)
        self.serving = True
