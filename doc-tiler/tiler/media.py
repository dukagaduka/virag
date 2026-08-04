"""Content sniffing: declared `doc_type` vs actual bytes (415 vs 422).

415 means "these bytes are not the type you declared"; 422 means "they are that
type but they are broken". Sniffing is intentionally shallow — the real decode
happens in the render workers.
"""

from __future__ import annotations

from .errors import unprocessable, unsupported_media

MIME_BY_TYPE = {
    "PDF": "application/pdf",
    "MHTML": "multipart/related",
    "IMG": "application/octet-stream",
}

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def _is_webp(head: bytes) -> bool:
    """True if the header bytes are a RIFF/WEBP container."""
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def sniff_image_mime(data: bytes) -> str | None:
    """Guess a raster image MIME type from magic bytes, else None."""
    head = data[:32]
    for magic, mime in _IMAGE_MAGIC:
        if head.startswith(magic):
            return mime
    if _is_webp(head):
        return "image/webp"
    if head.lstrip()[:5] == b"<?xml" or head.lstrip()[:4] == b"<svg":
        return None  # SVG needs a renderer we do not ship; treat as unsupported
    return None


def looks_like_mhtml(data: bytes) -> bool:
    """True if the leading bytes carry MIME/MHTML headers."""
    head = data[:4096].lower()
    return (
        b"mime-version:" in head
        or b"content-type: multipart/related" in head
        or b"content-location:" in head
        or b"snapshot-content-location:" in head
    )


def check_and_mime(doc_type: str, data: bytes) -> str:
    """Validate the payload against the declared type; return the stored mime type."""
    if not data:
        raise unprocessable("document is empty", "EMPTY_DOCUMENT")

    if doc_type == "PDF":
        if not data.lstrip()[:5].startswith(b"%PDF-"):
            raise unsupported_media(
                "declared doc_type=PDF but the payload does not start with %PDF-",
                declared="PDF",
            )
        return "application/pdf"

    if doc_type == "MHTML":
        if not looks_like_mhtml(data):
            raise unsupported_media(
                "declared doc_type=MHTML but the payload has no MIME/MHTML headers",
                declared="MHTML",
            )
        return "multipart/related"

    if doc_type == "IMG":
        mime = sniff_image_mime(data)
        if mime is None:
            raise unsupported_media(
                "declared doc_type=IMG but the payload is not a supported raster image "
                "(png/jpeg/gif/bmp/tiff/webp)",
                declared="IMG",
            )
        return mime

    raise unsupported_media(f"unsupported doc_type {doc_type!r}")


def extension_for(mime: str) -> str:
    """Map a stored MIME type to a file extension (default '.bin')."""
    return {
        "application/pdf": ".pdf",
        "multipart/related": ".mhtml",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }.get(mime, ".bin")
