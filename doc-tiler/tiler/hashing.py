"""Content-addressing helpers: the hash of a blob, and what a doc_id looks like."""

from __future__ import annotations

import hashlib


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def is_doc_id(value: str) -> bool:
    """True if `value` is a 64-char lowercase hex SHA-256 string."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)
