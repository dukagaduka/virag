"""Serving gate: virag answers only while its collection matches what the tiler renders.

The tiler's render_version is bound to its volume like geometry, and a Qdrant
collection is bound to the render_version its tiles were embedded from (collection
metadata). Serving across a mismatch would fail the integrity check on every tile, so
chat and enroll answer 503 until both agree. The gate is evaluated at startup and every
VIRAG_TILER_WATCH_INTERVAL_SEC:

  - the tiler is not serving (its own revision is pending or running) → wait;
  - the tiler serves another render_version than the collection's → rebuild: with
    VIRAG_REVISION_ON_RENDER_MISMATCH (default) the revision starts by itself, once
    per tiler render_version; otherwise it waits for POST /revision;
  - both agree → serve.

A tiler bound to another secret closes the gate too. An unreachable tiler leaves the
gate as it was: chat then degrades per tile, as before.

The gate lives in this process: correct for a single virag replica.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import Request

from virag.clients.qdrant_store import QdrantStore
from virag.clients.tiler_client import TilerService
from virag.errors import ApiError
from virag.revision import RevisionManager
from virag.settings import RevisionSettings


log = logging.getLogger("virag.gate")


class ServingGate:
    """Open while the tiler serves the render_version the collection was embedded from."""

    def __init__(
        self,
        store: QdrantStore,
        tiler: TilerService,
        revision: RevisionManager,
        settings: RevisionSettings,
        secret_hash: str,
    ):

        self.store       = store
        self.tiler       = tiler
        self.revision    = revision
        self.secret_hash = secret_hash

        self.auto_revision = settings.revision_on_render_mismatch
        self.interval      = settings.tiler_watch_interval_sec

        self.reason: str | None = "not checked yet"
        self._attempted: str | None = None      # tiler render_version auto-revised for


    @property
    def serving(self) -> bool:
        return self.reason is None


    async def check(self) -> None:
        """Re-evaluate the gate against the tiler and the collection; start the
        revision a render_version mismatch calls for."""
        try:
            info = await self.tiler.get_info()
            bound = (await self.store.get_collection_metadata()).get("render_version")

        except ApiError as e:
            log.warning("gate: cannot check the tiler (%s); keeping the gate as it is", e.message)

            return

        if info.get("secret_hash", "") != self.secret_hash:
            self._close("the tiler is bound to another TILER_DB_SECRET")

            return

        tiler_revision = info.get("revision") or {}
        if not tiler_revision.get("serving", True):
            self._close(f"the tiler is not serving (its revision is {tiler_revision.get('state')})")

            return

        tiler_rv = info.get("render_version")
        if bound == tiler_rv:
            self._open()

            return

        self._close(f"collection embedded from render_version {bound}, tiler renders {tiler_rv}")

        if self.revision.running or not self.auto_revision or self._attempted == tiler_rv:
            return

        # Once per tiler render_version: a failed rebuild waits for POST /revision.
        self._attempted = tiler_rv
        log.warning("gate: rebuilding the collection for render_version %s", tiler_rv)

        try:
            await self.revision.start()

        except ApiError as e:
            log.error("gate: cannot start the revision: %s", e.message)


    async def run(self) -> None:
        """Re-check on the watch interval until cancelled."""
        while True:
            await asyncio.sleep(self.interval)

            try:
                await self.check()

            except asyncio.CancelledError:
                raise

            except Exception as e:  # noqa: BLE001 - the watch must never crash the loop
                log.error("gate check failed: %r", e)


    def _open(self) -> None:
        if self.reason is not None:
            log.info("gate: serving (was: %s)", self.reason)

        self.reason = None


    def _close(self, reason: str) -> None:
        if reason != self.reason:
            log.warning("gate: not serving — %s", reason)

        self.reason = reason


def require_serving(request: Request) -> None:
    """Route dependency: chat and enroll answer 503 while the gate is closed."""
    gate: ServingGate = request.app.state.gate

    if not gate.serving:
        raise ApiError(
            status_code=503,
            message=f"Not serving: {gate.reason}",
            details={"revision": request.app.state.revision.status},
        )
