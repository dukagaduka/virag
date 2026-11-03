"""Coordination of virag processes through Valkey: who is alive, who writes document
points, who runs the revision, who leads reconcile.

Every process holds a lease, `procs:<token>` — the only key with a TTL. It renews it
every ttl/3 with SET XX (only while it still exists), so an expired lease never comes
back: a dead token stays dead. Whatever a process owns references its token and has no
TTL of its own; once the owner's lease is gone the resource is free, and it is taken
over with SET IFEQ <dead token>, so of two processes racing for it exactly one wins.

Keys (under VALKEY_PREFIX):

  procs:<token>       the lease: {host, pid, started_at}, TTL VIRAG_PROCESS_TTL_SEC
  writers             hash writer_id → {proc, doc_id, physical}: in-flight document writes
  revision:on         the token of the process running a revision or rollback; while it
                      exists, writers journal their doc_ids
  revision:status     the job status, with its owner
  revision:journal    doc_ids written while the job ran
  reconcile:on        the token of the reconcile leader

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


WRITERS          = "writers"
REVISION_ON      = "revision:on"
REVISION_STATUS  = "revision:status"
REVISION_JOURNAL = "revision:journal"
RECONCILE_ON     = "reconcile:on"


class Coordinator:
    """This process's lease, and every cluster-wide rule built on leases."""

    # How often drain re-reads the writers registry.
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
            await self.valkey.delete_ifeq(RECONCILE_ON, self.token)
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
    # writers of document points (enroll, self-heal)
    # ------------------------------------------------------------------ #
    @asynccontextmanager
    async def writer(self, doc_ids: Iterable[str]):
        """Register in-flight writes of these documents' points for their whole duration
        (an enroll from `begin` to commit/rollback, a self-heal batch), pinned to the
        collection the serving alias resolves to now — yielded, so every write and
        rollback delete goes there and a document lands in one collection whole.

        Registered before the alias is resolved, and journaled while a job runs: a writer
        that still lands in the collection a revision replaces is in its journal, and
        drain waits for it."""
        doc_ids = list(doc_ids)
        writers = {uuid.uuid4().hex: doc_id for doc_id in doc_ids}

        await self.valkey.hset_json(
            WRITERS, {wid: {"proc": self.token, "doc_id": doc_id} for wid, doc_id in writers.items()}
        )

        try:
            if await self.valkey.exists(REVISION_ON):
                await self.valkey.sadd(REVISION_JOURNAL, *doc_ids)

            physical = await self.store.serving_physical()

            await self.valkey.hset_json(
                WRITERS,
                {wid: {"proc": self.token, "doc_id": doc_id, "physical": physical} for wid, doc_id in writers.items()},
            )

            yield physical

        finally:
            try:
                await self.valkey.hdel(WRITERS, *writers)

            except ApiError as e:  # never mask the write's own outcome
                log.error("writers: cannot unregister %s (%s); they linger until this process ends", doc_ids, e.message)


    async def live_writers(self) -> list[dict]:
        """Writers whose process is alive; the entries of dead ones are dropped on the way."""
        writers = await self.valkey.hgetall_json(WRITERS)
        alive = await self._alive(writer["proc"] for writer in writers.values())

        dead = [wid for wid, writer in writers.items() if not alive[writer["proc"]]]
        if dead:
            await self.valkey.hdel(WRITERS, *dead)

        return [writer for writer in writers.values() if alive[writer["proc"]]]


    async def writing(self) -> set[str]:
        """doc_ids with a write in flight anywhere in the cluster."""
        return {writer["doc_id"] for writer in await self.live_writers()}


    async def drain(self, physical: str, timeout: float) -> bool:
        """Wait until no live writer is pinned to `physical`, or still unpinned (it may yet
        resolve to it). False if the timeout ran out first."""
        def busy(writers: list[dict]) -> bool:
            return any(writer.get("physical") in (None, physical) for writer in writers)

        try:
            async with asyncio.timeout(timeout):
                while busy(await self.live_writers()):
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
        while True:
            if await self.valkey.set_nx(REVISION_ON, self.token):
                return None

            owner = await self.valkey.get(REVISION_ON)
            if owner is None:
                continue                                        # released meanwhile

            if not await self._owner_dead(owner):
                return await self.get_status()

            if await self.valkey.set_ifeq(REVISION_ON, self.token, owner):
                log.warning("revision: taking over from dead process %s", owner)

                return None
            # Another process took it over first: the next round reads its token.


    async def owns_revision(self) -> bool:
        """Fencing before an irreversible step: this process still holds `revision:on`
        and its lease has not expired."""
        return (
            await self.valkey.get(REVISION_ON) == self.token
            and await self.valkey.exists(self._lease(self.token))
        )


    async def release_revision(self) -> None:
        """Release `revision:on` if this process still holds it. The journal stays: the
        next job starts its own."""
        await self.valkey.delete_ifeq(REVISION_ON, self.token)


    async def start_journal(self) -> None:
        """Start the job's journal: drop the last one, then record every live writer.
        Writers starting from now journal themselves — `revision:on` is set before this
        snapshot and a writer registers before checking it, so each is in one or the
        other. One that finished before the snapshot finished before the job read the
        tiler, so the job sees its result."""
        await self.valkey.delete(REVISION_JOURNAL)

        doc_ids = await self.writing()
        await self.valkey.sadd(REVISION_JOURNAL, *doc_ids)


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


    # ------------------------------------------------------------------ #
    # reconcile
    # ------------------------------------------------------------------ #
    async def lead_reconcile(self) -> bool:
        """Whether this process runs the reconcile pass: it leads already, takes the free
        lead, or takes it over from a dead leader. The lead is held while the leader lives."""
        leader = await self.valkey.get(RECONCILE_ON)

        if leader == self.token:
            return True

        if leader is None:
            return await self.valkey.set_nx(RECONCILE_ON, self.token)

        if not await self._owner_dead(leader):
            return False

        taken = await self.valkey.set_ifeq(RECONCILE_ON, self.token, leader)
        if taken:
            log.warning("reconcile: taking the lead over from dead process %s", leader)

        return taken


    @staticmethod
    def _lease(token: str) -> str:
        return f"procs:{token}"
