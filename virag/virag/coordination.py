"""Coordination of virag processes through Valkey: who is alive, who enrolls which
document, who runs the revision, who leads reconcile.

Every process holds a lease, `procs:<token>` — the only key with a TTL. It renews it
every ttl/3 with SET XX (only while it still exists), so an expired lease never comes
back: a dead token stays dead.

Whatever a process holds is one key: its name says what is held, its value is the
holder's token, and it has no TTL of its own. It is taken with SET NX; once the
holder's lease is gone it is free, and is taken over with SET IFEQ <dead token>, so of
two processes racing for it exactly one wins; it is released with DELIFEQ <own token>.

Keys (under VALKEY_PREFIX):

  procs:<token>       the lease: {host, pid, started_at}, TTL VIRAG_PROCESS_TTL_SEC
  enroll:<doc_id>     held by the process writing this document's points
  revision:on         held by the process running a revision or rollback; while it
                      exists, enrolls journal their doc_ids
  reconcile:on        held by the reconcile leader
  revision:status     the job status, with its owner
  revision:journal    doc_ids enrolled while the job ran

Collections are not coordinated here: Qdrant's serving alias is the one atomic pointer
to the serving collection, and it is resolved afresh wherever it matters.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterable

from virag.clients.qdrant_store import QdrantStore
from virag.clients.valkey_store import ValkeyStore
from virag.errors import ApiError


log = logging.getLogger("virag.coordination")


REVISION_ON      = "revision:on"
REVISION_STATUS  = "revision:status"
REVISION_JOURNAL = "revision:journal"
RECONCILE_ON     = "reconcile:on"


class EnrollInProgress(ApiError):
    """The document is held by another enroll, in this process or in any other."""

    def __init__(self, doc_id: str) -> None:

        super().__init__(
            status_code=409,
            message=f"doc_id {doc_id} is being enrolled by another request; retry once it is done",
            details={"doc_id": doc_id},
            code="ENROLL_IN_PROGRESS",
        )


class Coordinator:
    """This process's lease, and every cluster-wide rule built on leases."""

    # How often drain looks at the documents it waits for.
    DRAIN_POLL_SEC = 0.5


    def __init__(self, valkey: ValkeyStore, store: QdrantStore, *, process_ttl_sec: int):

        self.valkey = valkey
        self.store  = store
        self.ttl    = process_ttl_sec

        self.token = uuid.uuid4().hex
        self.identity = {
            "token": self.token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": int(time.time()),
        }


    # ------------------------------------------------------------------ #
    # lease
    # ------------------------------------------------------------------ #
    async def register(self) -> None:
        """Take this process's lease (a fresh token, so it cannot exist yet)."""
        await self.valkey.set_ttl(self._lease(self.token), self.identity, self.ttl, nx=True)
        log.info("lease: %s (%s/%s)", self.token, self.identity["host"], self.identity["pid"])


    async def heartbeat(self) -> None:
        """Renew the lease every ttl/3 until cancelled. A lease found expired means the
        other processes already count this one dead and may have taken what it held, so
        it stops itself: SIGTERM, graceful shutdown, restarted under a new token."""
        while True:
            await asyncio.sleep(self.ttl / 3)

            try:
                renewed = await self.valkey.set_ttl(self._lease(self.token), self.identity, self.ttl, xx=True)

            except ApiError as e:
                log.warning("lease: cannot renew (%s); retrying", e.message)
                continue

            if not renewed:
                log.critical("lease: %s expired; other processes count this one dead — stopping", self.token)
                os.kill(os.getpid(), signal.SIGTERM)

                return


    async def close(self) -> None:
        """Hand back the reconcile lead and the lease, so the others need not wait out the TTL."""
        try:
            await self._release(RECONCILE_ON)
            await self.valkey.delete(self._lease(self.token))

        except ApiError as e:
            log.warning("lease: cannot release (%s); it expires in %ss", e.message, self.ttl)


    async def _alive(self, tokens: Iterable[str]) -> dict[str, bool]:
        tokens = sorted(set(tokens))
        found = await self.valkey.exists_each(self._lease(token) for token in tokens)

        return dict(zip(tokens, found))


    async def _owner_dead(self, token: str) -> bool:
        return not (await self._alive([token]))[token]


    # ------------------------------------------------------------------ #
    # what a process holds
    # ------------------------------------------------------------------ #
    async def _take(self, name: str) -> str | None:
        """Take `name` for this process — free, or held by a dead process (taken over).
        None once taken; otherwise the token of the live process holding it, which may
        be this one."""
        while True:
            if await self.valkey.set_nx(name, self.token):
                return None

            holder = await self.valkey.get(name)
            if holder is None:
                continue                                        # released meanwhile

            if not await self._owner_dead(holder):
                return holder

            if await self.valkey.set_ifeq(name, self.token, holder):
                log.warning("%s: taken over from dead process %s", name, holder)

                return None
            # Another process took it over first: the next round reads its token.


    async def _holds(self, name: str) -> bool:
        """Fencing before an irreversible step: this process still holds `name` and its
        lease has not expired — nobody may have taken `name` over."""
        return (
            await self.valkey.get(name) == self.token
            and await self.valkey.exists(self._lease(self.token))
        )


    async def _release(self, name: str) -> None:
        """Release `name` if this process still holds it."""
        await self.valkey.delete_ifeq(name, self.token)


    # ------------------------------------------------------------------ #
    # documents being enrolled
    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def enrolling(self, doc_id: str):
        """Hold a document for the whole write of its points, from `begin` to the commit
        or the rollback: nobody else writes or deletes them meanwhile, and a second
        enroll of it is refused (EnrollInProgress). Yields the collection the serving
        alias resolves to now, so every write and the rollback delete go there and the
        document lands in one collection whole.

        The document is held before the alias is resolved, and journaled while a job
        runs: an enroll that still lands in the collection a revision replaces is in
        its journal, and drain waits for it."""
        name = self._enroll(doc_id)

        if await self._take(name) is not None:
            raise EnrollInProgress(doc_id)

        try:
            if await self.valkey.exists(REVISION_ON):
                await self.valkey.sadd(REVISION_JOURNAL, doc_id)

            yield await self.store.serving_physical()

        finally:
            try:
                await self._release(name)

            except ApiError as e:  # never mask the write's own outcome
                log.error("enroll: cannot release doc_id=%s (%s); it is held until this process ends", doc_id, e.message)


    async def holds_enroll(self, doc_id: str) -> bool:
        """Fencing before the rollback delete: the document is still this process's."""
        return await self._holds(self._enroll(doc_id))


    async def enrolling_now(self) -> dict[str, str]:
        """doc_id → the token of the live process enrolling it, across the cluster; what
        dead processes held is dropped on the way."""
        names = await self.valkey.names(self._enroll("*"))
        held = {name: holder for name, holder in zip(names, await self.valkey.get_each(names)) if holder is not None}
        alive = await self._alive(held.values())

        for name, holder in held.items():
            if not alive[holder]:
                await self.valkey.delete_ifeq(name, holder)

        return {name.removeprefix(self._enroll("")): holder for name, holder in held.items() if alive[holder]}


    async def drain(self, timeout: float) -> bool:
        """Wait for every enroll in flight right now to end — called after an alias
        swap: one of them may still be writing into the collection that was replaced,
        while whoever takes a document from now on resolves the new one. False if the
        timeout ran out first."""
        waited = await self.enrolling_now()

        def busy(now: dict[str, str]) -> bool:
            return any(now.get(doc_id) == holder for doc_id, holder in waited.items())

        try:
            async with asyncio.timeout(timeout):
                while busy(await self.enrolling_now()):
                    await asyncio.sleep(self.DRAIN_POLL_SEC)

        except TimeoutError:
            return False

        return True


    # ------------------------------------------------------------------ #
    # revision / rollback job
    # ------------------------------------------------------------------ #
    async def acquire_revision(self) -> dict | None:
        """Take `revision:on` for a job of this process — free, or held by a dead process
        (taken over). None once taken; otherwise the status of the live job holding it."""
        if await self._take(REVISION_ON) is None:
            return None

        return await self.get_status()


    async def owns_revision(self) -> bool:
        """Fencing before an irreversible step: the job is still this process's."""
        return await self._holds(REVISION_ON)


    async def release_revision(self) -> None:
        """Release `revision:on` if this process still holds it. The journal stays: the
        next job starts its own."""
        await self._release(REVISION_ON)


    async def start_journal(self) -> None:
        """Start the job's journal: drop the last one, then record every document being
        enrolled. Enrolls starting from now journal themselves — `revision:on` is set
        before this snapshot and an enroll holds its document before checking it, so
        each is in one or the other. One that finished before the snapshot finished
        before the job read the tiler, so the job sees its result."""
        await self.valkey.delete(REVISION_JOURNAL)
        await self.valkey.sadd(REVISION_JOURNAL, *await self.enrolling_now())


    async def journal(self) -> set[str]:
        """Every doc_id journaled since `start_journal`."""
        return await self.valkey.smembers(REVISION_JOURNAL)


    async def set_status(self, status: dict) -> None:
        """Publish the job status — only while this process still holds the job, so one
        that lost it never overwrites the status of the job that took over."""
        if await self.valkey.get(REVISION_ON) == self.token:
            await self.valkey.set_json(REVISION_STATUS, status)


    async def get_status(self) -> dict:
        """The current or last job's status, from any process. A running job whose
        owner's lease is gone is reported failed: its process died mid-job."""
        status = await self.valkey.get_json(REVISION_STATUS) or {"state": "idle"}
        owner = status.get("owner") or {}

        if status.get("state") == "running" and owner.get("token") and await self._owner_dead(owner["token"]):
            return {**status, "state": "failed", "error": f"owner lost ({owner.get('host')}/{owner.get('pid')})"}

        return status


    async def orphaned_job(self) -> dict | None:
        """The status of a job that still reads `running` though no other live process
        holds it — its process died, or it stopped short of its end — as it was left.
        None when the last job concluded or another process is running one. (For a
        process that is not running a job itself.)"""
        status = await self.valkey.get_json(REVISION_STATUS) or {}
        if status.get("state") != "running":
            return None

        holder = await self.valkey.get(REVISION_ON)
        if holder not in (None, self.token) and not await self._owner_dead(holder):
            return None

        return status


    # ------------------------------------------------------------------ #
    # reconcile
    # ------------------------------------------------------------------ #
    async def lead_reconcile(self) -> bool:
        """Whether this process runs the reconcile pass: it leads already, takes the free
        lead, or takes it over from a dead leader. The lead is held while the leader lives."""
        return await self._take(RECONCILE_ON) in (None, self.token)


    @staticmethod
    def _lease(token: str) -> str:
        return f"procs:{token}"


    @staticmethod
    def _enroll(doc_id: str) -> str:
        return f"enroll:{doc_id}"
