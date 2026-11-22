"""Content-addressed blob storage on the local filesystem.

An upload is first spooled to `.incoming/` under the blob root — hashed as it is
written, never held in memory — and stays there while its document renders. Only a
finished ingest moves it into the store (`commit`), by an atomic rename on the same
file system; an abandoned one is dropped (`discard`), and whatever a killed process
left behind is purged at startup.

The layout `ab/cd/<hash>` is deliberately isomorphic to an S3 key, and everything
goes through this interface so the filesystem can be swapped for object storage.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, AsyncIterator, BinaryIO

import anyio

from ..errors import not_found


@dataclass(frozen=True)
class Incoming:
    """An upload spooled to disk and not in the store yet."""

    path: Path
    doc_id: str  # sha256 of the content
    size: int


class BlobStore:
    """Stores and retrieves blob bytes under a content-addressed directory tree."""

    INCOMING = ".incoming"
    CHUNK = 1 << 18  # bytes per read of a streamed blob


    def __init__(self, root: str | Path):
        self.root = Path(root)


    @property
    def incoming_dir(self) -> Path:
        """Where uploads wait while their documents render."""
        return self.root / self.INCOMING


    def ensure_root(self) -> None:
        """Create the blob root and verify it is writable."""
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        probe = self.root / ".writable"
        probe.write_bytes(b"")
        probe.unlink()


    @staticmethod
    def rel_path(doc_id: str) -> str:
        """The blob's path relative to the root: `ab/cd/<hash>` — deliberately
        isomorphic to an S3 key."""
        return f"{doc_id[0:2]}/{doc_id[2:4]}/{doc_id}"


    def path(self, doc_id: str) -> Path:
        """Return the blob's absolute filesystem path."""
        return self.root / self.rel_path(doc_id)


    def exists(self, doc_id: str) -> bool:
        """Return whether a blob for this doc_id is stored."""
        return self.path(doc_id).exists()


    def require(self, doc_id: str) -> None:
        """Fail with 404 when a document's blob is gone from storage."""
        if not self.exists(doc_id):
            raise not_found(f"blob for {doc_id!r} is missing from storage", "MISSING_BLOB")


    # ------------------------------------------------------------------ #
    # incoming
    # ------------------------------------------------------------------ #
    async def spool(self, chunks: AsyncIterable[bytes] | bytes) -> Incoming:
        """Write an upload to `.incoming/`, hashing it on the way; nothing is left
        behind when the source fails."""
        if isinstance(chunks, bytes):
            chunks = self._once(chunks)

        path = self.incoming_dir / uuid.uuid4().hex
        digest = hashlib.sha256()
        size = 0
        fh = await asyncio.to_thread(open, path, "wb")

        try:
            async for chunk in chunks:
                await asyncio.to_thread(self._write, fh, digest, chunk)
                size += len(chunk)

            await asyncio.to_thread(self._sync, fh)

        except BaseException:
            path.unlink(missing_ok=True)
            raise

        finally:
            fh.close()

        return Incoming(path=path, doc_id=digest.hexdigest(), size=size)


    async def head(self, incoming: Incoming, size: int = 4096) -> bytes:
        """The first bytes of a spooled upload (for content sniffing)."""
        return await asyncio.to_thread(self._head, incoming.path, size)


    async def commit(self, incoming: Incoming) -> str:
        """Move a spooled upload into the store and return its relative path. The
        blob may be there already (identical content): the upload is then dropped."""
        await asyncio.to_thread(self._commit, incoming.path, self.path(incoming.doc_id))

        return self.rel_path(incoming.doc_id)


    async def discard(self, incoming: Incoming) -> None:
        """Drop a spooled upload (a no-op once it is committed)."""
        await asyncio.to_thread(incoming.path.unlink, missing_ok=True)


    def purge_incoming(self) -> int:
        """Drop every spooled upload — at startup, when no ingest can be alive. Returns
        how many there were."""
        leftovers = [path for path in self.incoming_dir.iterdir() if path.is_file()]

        for path in leftovers:
            path.unlink(missing_ok=True)

        return len(leftovers)


    # ------------------------------------------------------------------ #
    # stored blobs
    # ------------------------------------------------------------------ #
    async def read(self, doc_id: str) -> bytes:
        """Return the blob's bytes."""
        return await asyncio.to_thread(self.path(doc_id).read_bytes)


    async def size(self, doc_id: str) -> int:
        """Return the blob's size in bytes."""
        return (await asyncio.to_thread(self.path(doc_id).stat)).st_size


    async def stream(self, doc_id: str, start: int, end: int) -> AsyncIterator[bytes]:
        """The blob's bytes from `start` to `end` inclusive, in chunks."""
        remaining = end - start + 1

        async with await anyio.open_file(self.path(doc_id), "rb") as fh:
            await fh.seek(start)

            while remaining > 0:
                chunk = await fh.read(min(self.CHUNK, remaining))

                if not chunk:
                    break

                remaining -= len(chunk)
                yield chunk


    async def delete(self, doc_id: str) -> bool:
        """Delete the blob; return whether it existed."""

        def _unlink() -> bool:
            """Unlink the file, returning whether it was present."""
            p = self.path(doc_id)

            try:
                p.unlink()
                return True

            except FileNotFoundError:
                return False

        return await asyncio.to_thread(_unlink)


    # ------------------------------------------------------------------ #
    @staticmethod
    async def _once(data: bytes) -> AsyncIterable[bytes]:
        """Bytes already in memory as a one-chunk upload."""
        yield data


    @staticmethod
    def _write(fh: BinaryIO, digest: hashlib._Hash, chunk: bytes) -> None:
        """Write and hash one chunk (off the event loop)."""
        fh.write(chunk)
        digest.update(chunk)


    @staticmethod
    def _sync(fh: BinaryIO) -> None:
        """Flush a spooled upload to disk."""
        fh.flush()
        os.fsync(fh.fileno())


    @staticmethod
    def _head(path: Path, size: int) -> bytes:
        with open(path, "rb") as fh:
            return fh.read(size)


    @staticmethod
    def _commit(source: Path, dest: Path) -> None:
        """Rename `source` to `dest` atomically, or drop it when `dest` exists."""
        if dest.exists():  # dedup: identical content, identical path
            source.unlink(missing_ok=True)

            return

        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, dest)
        dir_fd = os.open(dest.parent, os.O_RDONLY)

        try:
            os.fsync(dir_fd)

        finally:
            os.close(dir_fd)
