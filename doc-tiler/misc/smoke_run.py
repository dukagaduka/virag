#!/usr/bin/env python3
"""End-to-end smoke run against a live Tiler service.

Point it at a directory of source documents and it will, purely through the HTTP
API:

  1. INGEST  — POST every file to /tile, saving each returned tile to disk.
  2. SEARCH  — pull the documents back via /docs/search (list-all plus a few
               indexed filters) to exercise metadata search end to end.
  3. VERIFY  — regenerate every tile via /get_tile and check its bytes hash to
               the image_hash the ingest manifest promised.

Input types are inferred from file extension; a `urls.txt` in the directory (one
URL per line) is ingested as doc_type=URL. Nothing here is specific to a given
document set — any directory with the right file formats works.

Usage:
    python misc/smoke_run.py [--input DIR] [--out DIR] [--base-url URL]

Requires the `requests` package on the host running the script.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("this script needs the 'requests' package (pip install requests)")

# Extension -> (doc_type, mime). Everything image-like is IMG; the service
# validates the bytes against the declared type on ingest.
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
EXT_TYPE = {".pdf": ("PDF", "application/pdf"), ".mhtml": ("MHTML", "multipart/related")}
EXT_TYPE.update({ext: ("IMG", mime) for ext, mime in IMAGE_MIME.items()})


@dataclass
class Item:
    """One thing to ingest: a local file or a URL."""

    name: str  # safe label used for the output subdirectory
    doc_type: str  # PDF | MHTML | IMG | URL
    path: Path | None = None
    url: str | None = None
    mime: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Record:
    """What we learned about one successfully ingested document."""

    item: Item
    doc_id: str
    tiles: list[dict]  # manifest tile entries: tile_num, image_hash, ...
    out_dir: Path


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    """Turn an arbitrary label into a filesystem-safe directory name."""
    keep = "-_.() "
    cleaned = "".join(c if c.isalnum() or c in keep else "_" for c in name).strip()
    return cleaned[:120] or "doc"


def _url_name(url: str) -> str:
    """Derive a readable label from a URL: host plus the last path segment."""
    parts = urlsplit(url)
    tail = unquote(parts.path.rstrip("/").rsplit("/", 1)[-1]) or "index"
    return _safe(f"{parts.hostname}__{tail}")


def discover(input_dir: Path) -> list[Item]:
    """Scan a directory into an ordered ingest list (files first, then urls.txt)."""
    items: list[Item] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_file():
            continue
        entry = EXT_TYPE.get(p.suffix.lower())
        if entry is None:
            continue
        doc_type, mime = entry
        meta = {"project": "smoke", "year": 2026, "source": {"host": "local"}}
        items.append(Item(name=_safe(p.stem), doc_type=doc_type, path=p, mime=mime, metadata=meta))

    urls_file = input_dir / "urls.txt"
    if urls_file.exists():
        for line in urls_file.read_text().splitlines():
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            host = urlsplit(url).hostname or "url"
            meta = {"project": "smoke", "year": 2026, "source": {"host": host}}
            items.append(Item(name=_url_name(url), doc_type="URL", url=url, metadata=meta))
    return items


# --------------------------------------------------------------------------- #
# Phase 1 — ingest
# --------------------------------------------------------------------------- #
def ingest(base_url: str, item: Item, out_root: Path) -> Record | None:
    """POST one item to /tile, then write every returned tile to disk."""
    params = {"store": "true", "include_bytes": "true"}
    data = {"doc_type": item.doc_type, "metadata": json.dumps(item.metadata)}
    files = None
    if item.doc_type == "URL":
        data["url"] = item.url
    else:
        files = {"document": (item.path.name, item.path.read_bytes(), item.mime)}

    try:
        resp = requests.post(f"{base_url}/tile", params=params, data=data, files=files, timeout=300)
    except requests.RequestException as exc:
        print(f"  ! {item.name}: request failed: {exc}")
        return None
    if resp.status_code != 200:
        print(f"  ! {item.name}: /tile -> {resp.status_code} {resp.text[:200]}")
        return None

    manifest = resp.json()
    out_dir = out_root / item.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile in manifest["tiles"]:
        (out_dir / f"tile_{tile['tile_num']:03d}.jpg").write_bytes(base64.b64decode(tile["data"]))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(
        f"  + {item.name}: {item.doc_type} -> doc_id={manifest['doc_id'][:12]}… "
        f"tiles={manifest['tile_count']} page_height={manifest['page_height']}"
    )
    return Record(item=item, doc_id=manifest["doc_id"], tiles=manifest["tiles"], out_dir=out_dir)


# --------------------------------------------------------------------------- #
# Phase 2 — search
# --------------------------------------------------------------------------- #
def _search(base_url: str, filters: list[dict] | None) -> list[dict]:
    """Page through /docs/search and return every matching summary."""
    items: list[dict] = []
    cursor = None
    while True:
        body: dict = {"limit": 100}
        if filters is not None:
            body["filters"] = filters
        if cursor:
            body["cursor"] = cursor
        resp = requests.post(f"{base_url}/docs/search", json=body, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        items.extend(page["items"])
        cursor = page.get("next_cursor")
        if not cursor:
            return items


def search_phase(base_url: str, records: list[Record]) -> None:
    """Exercise list-all plus indexed filters; report what came back."""
    stored = {r.doc_id for r in records}

    all_docs = _search(base_url, None)
    got = {d["doc_id"] for d in all_docs}
    print(f"  list-all: {len(all_docs)} docs; all ingested present: {stored <= got}")

    by_project = _search(base_url, [{"key": "project", "op": "eq", "value": "smoke"}])
    print(f"  project=smoke: {len(by_project)} docs")

    by_year = _search(base_url, [{"key": "year", "op": "eq", "value": 2026}])
    print(f"  year=2026 (numeric index): {len(by_year)} docs")

    # Selective host filter only matches the URL documents.
    hosts = {r.item.metadata["source"]["host"] for r in records if r.item.doc_type == "URL"}
    for host in sorted(hosts):
        hits = _search(base_url, [{"key": "source.host", "op": "eq", "value": host}])
        print(f"  source.host={host}: {len(hits)} docs")


# --------------------------------------------------------------------------- #
# Phase 3 — verify regeneration
# --------------------------------------------------------------------------- #
def verify_phase(base_url: str, records: list[Record]) -> tuple[int, int]:
    """Regenerate every tile via /get_tile and compare to the stored hash+file."""
    ok = bad = 0
    for r in records:
        mism = []
        for tile in r.tiles:
            n = tile["tile_num"]
            resp = requests.get(
                f"{base_url}/get_tile",
                params={"doc_id": r.doc_id, "tile_num": n, "verify": "true"},
                timeout=120,
            )
            if resp.status_code != 200:
                mism.append(f"#{n} HTTP {resp.status_code}")
                continue
            regen = hashlib.sha256(resp.content).hexdigest()
            saved = hashlib.sha256((r.out_dir / f"tile_{n:03d}.jpg").read_bytes()).hexdigest()
            if regen != tile["image_hash"]:
                mism.append(f"#{n} regen≠manifest")
            elif regen != saved:
                mism.append(f"#{n} regen≠saved")
        if mism:
            bad += 1
            print(f"  ✗ {r.item.name}: {len(mism)} mismatch(es): {', '.join(mism[:5])}")
        else:
            ok += 1
            print(f"  ✓ {r.item.name}: {len(r.tiles)} tiles regenerate byte-identically")
    return ok, bad


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Tiler end-to-end smoke run.")
    ap.add_argument("--input", type=Path, default=Path("artifacts/inputs"), help="source documents dir")
    ap.add_argument("--out", type=Path, default=Path("artifacts/tiles"), help="where tiles are written")
    ap.add_argument("--base-url", default="http://localhost:8077", help="running Tiler base URL")
    args = ap.parse_args()

    if not args.input.is_dir():
        return f"input dir not found: {args.input}"
    try:
        requests.get(f"{args.base_url}/readyz", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        return f"service not ready at {args.base_url}: {exc}"

    items = discover(args.input)
    if not items:
        return f"no ingestable files found in {args.input}"

    print(f"\n== INGEST ({len(items)} inputs) ==")
    records = [rec for it in items if (rec := ingest(args.base_url, it, args.out)) is not None]

    print(f"\n== SEARCH ==")
    search_phase(args.base_url, records)

    print(f"\n== VERIFY (regenerate vs stored) ==")
    ok, bad = verify_phase(args.base_url, records)

    total_tiles = sum(len(r.tiles) for r in records)
    print(
        f"\n== SUMMARY ==\n"
        f"  ingested {len(records)}/{len(items)} documents, {total_tiles} tiles total\n"
        f"  verify: {ok} documents byte-identical, {bad} with mismatches\n"
        f"  tiles written under: {args.out}"
    )
    return 0 if (records and bad == 0 and len(records) == len(items)) else 1


if __name__ == "__main__":
    sys.exit(main())
