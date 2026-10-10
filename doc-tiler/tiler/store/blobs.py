"""Content-addressed blob storage on the local filesystem.

Write-once and atomic: temp file in the destination directory → fsync → rename.
The layout `ab/cd/<hash>` is deliberately isomorphic to an S3 key, and everything
goes through this interface so the filesystem can be swapped for object storage.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from ..hashing import blob_rel_path


class BlobStore:
    """Stores and retrieves blob bytes under a content-addressed directory tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)


    def ensure_root(self) -> None:
        """Create the blob root and verify it is writable."""
        self.root.mkdir(parents=True, exist_ok=True)
        probe = self.root / ".writable"
        probe.write_bytes(b"")
        probe.unlink()


    def rel_path(self, doc_id: str) -> str:
        """Return the blob's path relative to the root."""
        return blob_rel_path(doc_id)


    def path(self, doc_id: str) -> Path:
        """Return the blob's absolute filesystem path."""
        return self.root / blob_rel_path(doc_id)


    def exists(self, doc_id: str) -> bool:
        """Return whether a blob for this doc_id is stored."""
        return self.path(doc_id).exists()


    def _put_sync(self, doc_id: str, data: bytes) -> str:
        """Write the blob atomically, skipping if it already exists."""
        dest = self.path(doc_id)
        if dest.exists():  # dedup: identical content, identical path
            return blob_rel_path(doc_id)
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")

        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())

            os.replace(tmp, dest)
            dir_fd = os.open(dest.parent, os.O_RDONLY)

            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        
        return blob_rel_path(doc_id)


    async def put(self, doc_id: str, data: bytes) -> str:
        """Store the blob and return its relative path."""
        return await asyncio.to_thread(self._put_sync, doc_id, data)


    async def read(self, doc_id: str) -> bytes:
        """Return the blob's bytes."""
        return await asyncio.to_thread(self.path(doc_id).read_bytes)


    async def size(self, doc_id: str) -> int:
        """Return the blob's size in bytes."""
        return (await asyncio.to_thread(self.path(doc_id).stat)).st_size


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
