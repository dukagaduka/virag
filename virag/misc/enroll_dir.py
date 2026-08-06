#!/usr/bin/env python3
"""Bulk-enroll a directory of documents into a running virag service.

Point it at a directory and it walks it for MHTML, PDF and image files, then
POSTs each one to virag's ``/enroll`` endpoint over plain HTTP. Every document is
named after its file (``metadata.name`` = the file's stem), so it shows up under
that name in retrieval sources and re-enrolling the same file overwrites its own
tiles instead of duplicating them.

Nothing here imports from the virag package — it only speaks the HTTP API, so it
runs anywhere ``requests`` (and, optionally, ``tqdm``) are installed.

Recognised inputs (by extension):

  * .pdf                         -> doc_type=PDF
  * .mhtml, .mht                 -> doc_type=MHTML
  * .png .jpg .jpeg .webp .gif .bmp .tif .tiff  -> doc_type=IMG

Usage:
    python misc/enroll_dir.py DIR --endpoint http://localhost:8080
    python misc/enroll_dir.py DIR -e http://host:8080 --recursive
    python misc/enroll_dir.py DIR -e http://host:8080 \
        --metadata '{"project":"atlas","year":2026}' --force

Extra ``--metadata`` (a JSON object) is merged into every document's metadata;
the per-file ``name`` always wins. See ``--help`` for the full flag list.

Requires the ``requests`` package; ``tqdm`` is used for the progress bar if
present and silently skipped otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("this script needs the 'requests' package (pip install requests)")

try:
    from tqdm import tqdm

    def _log(msg: str) -> None:
        """Print without corrupting the progress bar."""
        tqdm.write(msg)

except ImportError:  # tqdm is optional — fall back to a plain iterator + print
    def tqdm(iterable, **_):  # type: ignore
        return iterable

    def _log(msg: str) -> None:
        print(msg)

# Extension -> (doc_type, mime). Everything image-like is IMG; virag/tiler
# validates the bytes against the declared type on enroll.
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
EXT_TYPE = {
    ".pdf": ("PDF", "application/pdf"),
    ".mhtml": ("MHTML", "multipart/related"),
    ".mht": ("MHTML", "multipart/related"),
}
EXT_TYPE.update({ext: ("IMG", mime) for ext, mime in IMAGE_MIME.items()})


@dataclass
class Item:
    """One local file queued for enrollment."""

    path: Path
    doc_type: str  # PDF | MHTML | IMG
    mime: str
    name: str  # metadata.name — the file stem


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover(input_dir: Path, recursive: bool) -> list[Item]:
    """Scan a directory into an ordered enroll list, skipping unknown types."""
    walk = input_dir.rglob("*") if recursive else input_dir.iterdir()

    items: list[Item] = []

    for p in sorted(walk):
        if not p.is_file():
            continue

        entry = EXT_TYPE.get(p.suffix.lower())
        if entry is None:
            continue

        doc_type, mime = entry
        items.append(Item(path=p, doc_type=doc_type, mime=mime, name=p.stem))

    return items


# --------------------------------------------------------------------------- #
# Enroll
# --------------------------------------------------------------------------- #
def enroll(endpoint: str, item: Item, base_meta: dict, *, force: bool) -> tuple[bool, str]:
    """POST one file to /enroll. Returns (ok, one-line summary for logging)."""
    metadata = {**base_meta, "name": item.name}
    data = {"doc_type": item.doc_type, "metadata": json.dumps(metadata, ensure_ascii=False)}
    files = {"document": (item.path.name, item.path.read_bytes(), item.mime)}
    params = {"force": "true"} if force else None

    try:
        resp = requests.post(
            f"{endpoint}/enroll",
            params=params,
            data=data,
            files=files,
            timeout=600,
        )

    except requests.RequestException as exc:
        return False, f"request failed: {exc}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code} {resp.text[:200]}"

    body = resp.json()

    return True, (
        f"doc_id={body['doc_id'][:12]}… tiles={body['tile_count']} "
        f"enrolled={body['enrolled']}"
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bulk-enroll a directory of PDF/MHTML/image files into virag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Each file is named by its stem (metadata.name), so retrieval sources\n"
            "show that name and re-running overwrites a document's own tiles."
        ),
    )
    ap.add_argument("directory", type=Path, help="directory to scan for documents")
    ap.add_argument(
        "-e", "--endpoint",
        default="http://localhost:8080",
        help="virag base URL (default: %(default)s)",
    )
    ap.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="descend into subdirectories",
    )
    ap.add_argument(
        "--metadata",
        default="{}",
        help='extra metadata merged into every doc, as a JSON object (name always wins)',
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-tile documents that already exist",
    )
    args = ap.parse_args()

    endpoint = args.endpoint.rstrip("/")

    if not args.directory.is_dir():
        return f"not a directory: {args.directory}"

    try:
        base_meta = json.loads(args.metadata)
        if not isinstance(base_meta, dict):
            return "--metadata must be a JSON object"

    except json.JSONDecodeError as exc:
        return f"--metadata is not valid JSON: {exc}"

    try:
        requests.get(f"{endpoint}/ready", timeout=10).raise_for_status()

    except requests.RequestException as exc:
        return f"virag not ready at {endpoint}: {exc}"

    items = discover(args.directory, args.recursive)
    if not items:
        return f"no PDF/MHTML/image files found in {args.directory}"

    print(f"enrolling {len(items)} document(s) into {endpoint}")

    ok = 0
    failures: list[str] = []

    for item in tqdm(items, desc="enroll", unit="doc"):
        succeeded, summary = enroll(endpoint, item, base_meta, force=args.force)

        if succeeded:
            ok += 1
            _log(f"  + {item.name}: {summary}")
        else:
            failures.append(f"{item.path}: {summary}")
            _log(f"  ! {item.name}: {summary}")

    print(f"\ndone: {ok}/{len(items)} enrolled, {len(failures)} failed")

    for line in failures:
        print(f"  ! {line}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
