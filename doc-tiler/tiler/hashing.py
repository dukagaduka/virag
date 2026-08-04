"""Content-addressing helpers: blob hashing and doc_id paths."""

from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def blob_rel_path(doc_id: str) -> str:
    """`ab/cd/<hash>` — deliberately isomorphic to an S3 key."""
    return f"{doc_id[0:2]}/{doc_id[2:4]}/{doc_id}"


def is_doc_id(value: str) -> bool:
    """True if `value` is a 64-char lowercase hex SHA-256 string."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)
