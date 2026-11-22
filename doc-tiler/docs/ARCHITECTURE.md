# Architecture

The rationale the wire contract (`openapi.yaml`) can't express: how the Tiler is
put together and why. This is the reference for anyone changing the internals.

- **Stack:** Python 3.11+, FastAPI, SQLite (JSON1 + WAL + expression indexes),
  Playwright (Chromium, pinned), PyMuPDF, Pillow.
- **Deployment:** a single container, one named volume for state, one mounted config
  file for the metadata schema. See [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 1. Purpose & scope

The Tiler turns documents (**PDF / URL / MHTML / IMG**) into fixed-geometry **image
tiles** for a pixel-based (screenshot) RAG embedder, and can **regenerate any single
tile on demand** from the stored original. It also maintains **operator-declared
metadata** with fast indexed search so a human can tell what is in the store.

**In scope:** ingest (one-shot or streaming), blob storage, deterministic tiling, tile
regeneration, in-place re-render (revision), original download, metadata (declared-schema
fast search), doc listing.

**Out of scope:** embedding, Qdrant / vector search, retrieval, the LLM/VLM reader,
the RAG gateway. The Tiler only produces and serves image bytes + identifiers +
metadata. The `(doc_id, tile_num)` pair returned with every tile is the contract the
embedding stage stores as its vector-DB payload to later ask for the tile
(`GET /doc/{doc_id}/tiles/{tile_num}`).

### Design invariants (non-negotiable)

1. **Geometry is fixed and global:** `TILER_TILE_WIDTH = 875`,
   `TILER_TILE_HEIGHT = 1024`. Matches the target embedder (Qwen3-VL-Embedding). A
   DB/volume is bound to one geometry for life (see "Geometry guard" below).
2. **Do not persist tiles.** Persist the original blob + metadata; regenerate any
   tile deterministically.
3. **`doc_id = sha256(stored_blob_bytes)` (hex).** Content-addressed ⇒ idempotent,
   deduplicated.
4. **A tile is `(doc_id, tile_num)`.** No separate `tile_id`; `tile_num` is 0-based.
5. **Metadata never affects `doc_id`.** Mutable annotation over immutable content.
6. **Reproducible rendering:** pinned Chromium, fixed viewport/fonts/DSR=1.

---

## 2. Concurrency model — three execution layers

Three layers, split by the *nature* of the work and the GIL.

### Layer 1 — event loop (one thread)

Uvicorn/FastAPI runs a single asyncio thread doing **only network I/O**. It must never
block. A handler reads the request → dispatches heavy work to a pool → `await`s →
returns. Any blocking/CPU call made directly in a handler freezes all in-flight
requests.

Run **one** Uvicorn worker (`--workers 1`); concurrency comes from the loop plus the
pools below. Multiple workers would each spawn their own Chromium pool and SQLite
writer and fight over the volume. Scale with more containers.

### Layer 2 — browser (Playwright/Chromium), I/O-bound

Web render / MHTML capture is mostly *waiting* (nav, network, JS, layout) — I/O, not
Python CPU. Playwright async runs on the event loop over CDP; no extra thread. Cap
concurrent renders with `asyncio.Semaphore(TILER_N_BROWSERS)` (each open context costs
Chromium RAM/CPU).

- Few long-lived Chromium instances, launched in `lifespan`.
- **Fresh `BrowserContext` per request** for isolation; closed after use.
- Recycle a Chromium process after `TILER_RECYCLE_AFTER` renders; detect crashed
  contexts and replace the process.

### Layer 3 — Render workers (PDF raster, PIL resize/slice, JPEG), CPU-bound

Done inline it blocks the loop; in threads the GIL kills parallelism. So:
a process pool of `TILER_N_RENDER_WORKERS` — separate processes, true multi-core. It is
`pebble.ProcessPool` rather than the stdlib `ProcessPoolExecutor`: pebble can stop one
running task (it kills and replaces that worker), and a dead worker fails only its own
task, where the stdlib pool breaks as a whole. Dispatch via `pool.schedule(fn, args,
timeout=...)`, awaited through `asyncio.wrap_future`.

The render pool uses the **`spawn`** start method: the parent runs an event loop and
Playwright threads, and forking those into workers deadlocks.

### Documents, readers and the RenderPool

Every document, whatever its format, is **one image of width `W` and height `H`**, and
a tile is a run of its rows `[offset, offset + height)`. The service works on that
alone, and there is one way from a file to its tiles — the `Renderer`
(`render/renderer.py`): an ingest, a revision and a request for a stored doc's tiles all
open the document through it (`renderer.open(...)`) and read `doc.tiles()`. The document
(`render/documents.py`) measures itself and cuts itself into tiles (`plan()`), unless
the cut is given, and renders them batch by batch (`render()`). Each tile's `offset` and
`height` are stored with it, so a tile request and a stored doc's stream pass that cut back
and never cut the document again. The Renderer also records every render in the metrics
(`path` = ingest / tiles / revise).

How the rows are produced is the document's own business:

- `WorkerDocument` (PDF, images) works entirely inside the render workers, through the
  format's **reader** (`render/readers.py`), which exists only there:
  - `PdfReader` stacks the pages top to bottom — where each page sits is known to it
    alone — and rasterizes just the requested rows of a page from its display list (the
    page is parsed once), at `TILER_PDF_DPI` but never wider than 4 × `W`, then
    downscales them to `W`;
  - `ImageReader` decodes the image once (a JPEG straight at a 1/2ᵏ scale when that
    still covers `W`) and crops.
- `MhtmlDocument` loads the archive in a Chromium tab on the event loop: one full-page
  capture up to `TILER_SAFE_FULLPAGE_PX` (cached per document), a clip capture per tile
  beyond it; the workers encode. The path is chosen by `H` alone, so ingest and
  a tile request always take the same one.

`RenderPool` (`render/pool.py`) only executes: it runs a task in a render worker or
hands out a Chromium slot, bounds the work and keeps itself alive, and knows nothing
about document types. A worker task names a document by type and path (never its
bytes); the worker opens the reader for it and keeps it open between the tasks of one
document, so a PDF stays parsed. The render params are the same for every task, so a
worker gets them once, at start.

**Memory of a PDF tile.** A tile depends only on its own rows: a read rasterizes those
rows (plus the resampling margin) and nothing else, so a tile request never rasterizes a
page for one strip of it. The supersampling cap (a pixel-tier constant, in
`render_version`) holds the raster to 4 × `W` wide, so a 1024-row tile takes at most
~43 MB of source raster; what a taller tile (an unsliced page) needs is bounded by the
worker's memory cap, below.

**Limits and recovery.**

- Each worker caps its address space at `TILER_RENDER_WORKER_MEMORY`: a document that
  needs more fails alone with `422 RENDER_TOO_LARGE`, the worker survives, and the
  kernel's OOM killer never sees it.
- A task may execute for `TILER_RENDER_TIMEOUT_SEC` (time spent queued does not count);
  past it its worker is killed and replaced, and the document fails with
  `504 RENDER_TIMEOUT`. The other workers' tasks are untouched.
- A worker that dies (a segfault in a decoder) fails only its own task, and the pool
  replaces it. Since a worker can also be killed from outside (the kernel's OOM killer),
  the task runs once more; a document that kills the second worker too fails with
  `422 RENDER_CRASHED`. `tiler_render_worker_restarts_total` counts the replacements.
- pebble gives the whole pool up when a worker dies while **no** task runs (it takes
  that for a failing initializer). The RenderPool then replaces the pool and proves the
  new one starts with a no-op task; the next task or `/readyz` triggers it.
- A cancelled task (the ingest it belongs to failed or its client went away) stops
  too: its worker is killed and replaced.
- A revision quarantines a document that fails these ways; only workers that cannot
  start at all (`WORKER_CRASH`) or an unavailable Chromium abort it.

### Why not simpler

All-threads ⇒ CPU-bound + GIL = no speedup. One pool including the browser ⇒ Chromium
is expensive to start; keep it warm and shared behind a semaphore.

### Flow examples

- **PDF ingest:** handler (loop) spools the upload to `.incoming`, hashing it → the row
  goes in, incomplete → one worker task lays the document out (its height and its tiles)
  → batches of ≤16 tiles fan out across the pool (a task carries the file's path, never
  its bytes) → each batch's tiles go to the stream in order → the commit moves the blob
  into the store and writes all the tile rows in one transaction.
- **URL ingest:** handler takes the browser semaphore → Playwright navigates +
  `captureSnapshot(mhtml)` → releases → the snapshot is stored and read like any MHTML
  archive → DB → response.

### Tile cache

In-process **LRU** (`TILER_RENDER_CACHE_SIZE`), async-safe, with **in-flight
coalescing** — a per-key future map so concurrent requests for the same tile render
once.

- Key `(doc_id, tile_num)` → final JPEG bytes (all types).
- Web/MHTML additionally caches the full-page capture per document (in the render pool)
  so sequential strip requests reuse one render.

---

## 3. Data model (SQLite)

Startup pragmas: `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`,
`foreign_keys=ON`. SQLite is used through two connections: writes go through one,
serialized by a lock; reads through another, opened read-only. In WAL mode the reader
sees committed data only, so a request never observes a commit half-way through (a doc
already complete, its tiles not written yet), and reads do not queue behind writes. The
workload is metadata-only (blobs live on disk), so transactions stay trivial while the
event loop stays free.

```sql
-- Singleton service-config guard (see "Geometry guard"). Exactly one row.
CREATE TABLE IF NOT EXISTS service_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    tile_width      INTEGER NOT NULL,
    tile_height     INTEGER NOT NULL,
    tile_overlap    INTEGER NOT NULL,
    min_tile_height INTEGER NOT NULL,
    render_version  TEXT    NOT NULL,   -- geometry + pixel tier
    volume_id       TEXT    NOT NULL,   -- random at init; identifies the volume
    initialized_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS docs (
    doc_id         TEXT PRIMARY KEY,     -- sha256(blob) hex
    doc_type       TEXT NOT NULL,        -- 'PDF' | 'MHTML' | 'IMG' (URL stored as MHTML)
    blob_rel_path  TEXT NOT NULL,        -- 'ab/cd/<hash>' relative to TILER_BLOB_DIR
    source_url     TEXT,                 -- original URL if fetched, else NULL
    height         INTEGER NOT NULL,     -- height of the document image at tile width (PDF: its pages, stacked)
    tile_count     INTEGER NOT NULL,
    mime_type      TEXT NOT NULL,
    metadata       TEXT,                 -- raw JSON object (single source of truth); NULL if none
    render_version TEXT NOT NULL,        -- stamp (informational; geometry uniform by the startup guard)
    complete       INTEGER NOT NULL DEFAULT 1,  -- 0 while the doc's first ingest is in flight; see §9
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tiles (
    doc_id      TEXT NOT NULL,
    tile_num    INTEGER NOT NULL,        -- 0-based
    offset      INTEGER NOT NULL,        -- first row of the tile in the document image
    height      INTEGER NOT NULL,        -- actual px (variable for PDF & final strip)
    image_hash  TEXT NOT NULL,           -- sha256 of JPEG bytes at ingest
    PRIMARY KEY (doc_id, tile_num),
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_docs_created ON docs(created_at, doc_id);  -- cursor pagination
```

Metadata is **not a table** — it lives in `docs.metadata` as JSON and is made
searchable by expression indexes created at startup, one per declared field (see
"Metadata: declared schema + expression indexes").
`tiles.width` is not stored (always `TILER_TILE_WIDTH`).

A doc's row is inserted `complete=0` when its ingest starts and flipped when the
ingest commits (see §9); nothing ever turns a complete doc incomplete again.

Everything storage-related sits behind `BlobStore` / `MetaStore` interfaces, keeping
the FS→S3 and SQLite→Postgres swaps to a driver change.

---

## 4. Blob storage (content-addressed)

- Root `TILER_BLOB_DIR`; layout `<hash[0:2]>/<hash[2:4]>/<hash>`, `hash = doc_id`.
- **Spooled first.** An upload is written to `<root>/.incoming/<random>`, hashed as it
  goes (never held in memory), and stays there while its document renders.
- **Write-once, atomic commit:** `fsync` at the end of the spool, then one `os.rename`
  into the content-addressed path when the ingest commits (same file system). If the
  final path exists, the upload is dropped (dedup).
- An ingest that does not commit removes its upload; `.incoming` is emptied at startup.
- SQLite holds metadata only; blobs never in the DB.
- Deliberately **isomorphic to S3** (`ab/cd/<hash>` → bucket key).

**The blob is committed before the tile rows** (the reverse of the naive order). A blob
without its tiles is inert and dedups away on the next ingest; tiles without their blob
are a tile that can never be served.

---

## 5. Tiling geometry (exact)

`W = TILER_TILE_WIDTH`, `TH = TILER_TILE_HEIGHT`, `OV = TILER_TILE_OVERLAP`,
`STEP = TH - OV`, `MIN = TILER_MIN_TILE_HEIGHT`.

A document is an image of width exactly `W` and height `H`; a tile is its rows
`[offset, offset + height)`. `docs.height = H`, and every tile's `offset` and `height`
are stored in `tiles`.

**Reverse (a tile request):** nothing to compute — the tile's stored rows are read back from
the blob. Ingest is the only place a document is cut, so the two cannot drift.

### 5.1 Web / MHTML / IMG — portrait strips

```
tiles = []; offset = 0
while True:
    h = min(TH, H - offset)
    if h >= MIN: tiles.append({tile_num: len(tiles), offset, height: h})
    if offset + TH >= H: break
    offset += STEP
tile_count = len(tiles)
```

Every offset is `tile_num * STEP` because a sliver is only ever dropped as the final
element (a property test over random `H`).

A page shorter than `MIN` yields **zero tiles** (the only strip is a sliver, and
slivers are dropped); such a doc is stored with `tile_count = 0`.

### 5.2 PDF — one tile per page (default), or sliced (opt-in)

Raster page `p` at `TILER_PDF_DPI` (capped at 4 × `W` wide) → `(w0, h0)` →
proportional resize to width `W` (`new_h = round(h0 * W / w0)`, LANCZOS). The pages are
stacked top to bottom into the document image: a page starts where the one before it
ends, `H = Σ new_h`. Pages may differ in size.

**Default (`TILER_PDF_SLICE_ENABLE=false`):** a page is a tile: `tile_num = p`,
`offset` = the page's top, `height = new_h`. `tile_count = pages`.

**Sliced (`TILER_PDF_SLICE_ENABLE=true`):** each page is cut into strips of
`TILER_PDF_TILE_HEIGHT` (overlap `TILER_PDF_TILE_OVERLAP`) by the same rule as 5.1 —
**strictly within a page, never across pages**. Strips get a contiguous global
`tile_num` in page order. The PDF slicing sizes join `render_version` only while
slicing is on.

Where the pages sit is not stored: the `PdfReader` works it out from the blob when it
opens the document (page boxes only, no rasterization) and maps a tile's rows to their
page.

### 5.3 IMG

As for Web/MHTML/IMG (portrait strips) above, after proportional resize to width `W`.

### 5.4 Alignment

Do **not** round to 28 here. Width stays 875; heights are exact crops. The embedder
does its own 28px patch alignment and L2 normalization downstream — the Tiler must not
pre-normalize or 28-align.

---

## 6. Metadata: declared schema + expression indexes

The operator declares the exact keys that get indexed, in a mounted file
(`TILER_METADATA_SCHEMA_PATH`). Only declared keys are searchable; anything else in the
JSON is stored but not searchable. See [CONFIG.md](CONFIG.md) for the schema file
format and [API.md](API.md) for the search/CRUD endpoints.

### Indexing (expression indexes, auto-maintained)

At startup, for each declared field, create an expression index on the JSON path:

```python
def meta_expr(key: str) -> str:                    # SINGLE source of the extraction expression
    return f"json_extract(metadata, '$.{key}')"    # dotted keys work: '$.source.host'
```

SQLite computes the extraction once per row, stores it in a B-tree, and auto-maintains
it on every write of `docs.metadata` — no dual-write, no separate table. Adding a field
later is the same `CREATE INDEX IF NOT EXISTS`; SQLite backfills it. Stale
`idx_meta_*` indexes for removed fields are dropped at startup. That is the whole
"reindex" story; there is no admin endpoint.

- **Textual match:** SQLite uses the expression index only when the query's `WHERE`
  expression is *textually identical* to the index's. The query builder MUST build
  predicates through the same `meta_expr(key)` helper. Verified with `EXPLAIN QUERY
  PLAN` in tests: expect `SEARCH ... USING INDEX idx_meta_*`, not `SCAN`.
- **STAT4 is required for selective range filters.** A numeric range (`year > 2050`)
  uses `idx_meta_year` only if the planner can tell the range is selective, which needs
  range-selectivity histograms — built only when SQLite is compiled with
  `SQLITE_ENABLE_STAT4`. CPython's bundled `sqlite3` lacks STAT4, so the service routes
  SQLite through **`pysqlite3-binary`** (a STAT4 build): `tiler/__init__.py` swaps
  `sys.modules["sqlite3"]` before `aiosqlite` imports it, with a silent stdlib
  fallback. `PRAGMA optimize` at startup refreshes those statistics. Without STAT4 a
  selective range silently falls back to walking `idx_docs_created` — functional but
  slow; equality filters are unaffected. `pysqlite3` is therefore a functional
  requirement, not just an optimization.

### The query planner, honestly

Every predicate is built through `meta_expr()`, so expressions match textually. Two
caveats about what the planner then does:

- **`ne` (`<>`) is not a seekable B-tree constraint** — in any SQL engine. Queries
  using it walk `idx_docs_created` in `created_at DESC` order and filter as they go.
  Cheap in practice (`ne` matches nearly every row, so `LIMIT` is satisfied almost
  immediately) but not an index seek. `contains` was dropped for this reason; `ne`
  stays because it is in the OpenAPI contract.
- **A low-selectivity range** (`year > 1900` when everything matches) is also answered
  by the ordered scan — the planner correctly judges that seeking the index then
  sorting ~all rows costs more. A *selective* range (`year > 2050`) uses
  `idx_meta_year`. Neither case is an unindexed table scan, and `LIMIT` bounds the
  work.

### Metadata typing

`text` accepts strings (numbers coerced with `str()`); `number` accepts int/float and
numeric strings; booleans, objects and arrays are rejected for both (`422`). A declared
key present but `null` counts as absent — and therefore violates `required`. Undeclared
keys are stored verbatim and never touched. Schema keys must be dotted plain
identifiers (enforced at parse time, because the key is interpolated into the SQL /
JSON-path text).

---

## 7. Geometry guard (startup) & integrity

### One DB, one geometry, for life

A DB/volume is permanently bound to the geometry it was created with. On startup:

```
row = SELECT * FROM service_meta WHERE id = 1
cur = {TILER_TILE_WIDTH, TILER_TILE_HEIGHT, TILER_TILE_OVERLAP, TILER_MIN_TILE_HEIGHT}
if row is None:
    INSERT service_meta(cur..., render_version, now)       # fresh volume
elif geometry(row) != cur:
    log.FATAL("DB initialised with a different geometry; refusing to start.")
    sys.exit(1)                                            # HARD FAIL
if docs rendered by another render_version exist:         # pixel tier
    start NOT SERVING; revise them, then serve             # see §9 "Revision"
```

Geometry mismatch is caught **once, globally, before serving anything** — there is no
per-request geometry gate. To change geometry, deploy against a fresh volume and
re-ingest. The guard is a part of opening the volume (`Volume.open`, `volume.py`), which
runs both in `lifespan` and standalone as `python -m tiler.preflight` — the container
entrypoint runs that before uvicorn, so a mismatched volume produces one clear log line
instead of a generic ASGI startup failure.

The `volume_id` column is a random id given to the volume at init and reported on
`GET /info`, so a consumer can tell this volume from a fresh or another one behind the
same URL. It is not a secret. See [SECURITY.md](SECURITY.md).

### `render_version`: bound to the volume, resolved by revision

- **Geometry tier** = `{TILER_TILE_WIDTH, TILER_TILE_HEIGHT, TILER_TILE_OVERLAP,
  TILER_MIN_TILE_HEIGHT}` — enforced hard at startup.
- **Pixel tier** = `{TILER_JPEG_QUALITY, JPEG_SUBSAMPLING (4:2:0), JPEG_PROGRESSIVE
  (off), TILER_PDF_DPI, resize_filter=LANCZOS, TILER_CHROMIUM_REVISION,
  pillow_version, pymupdf_version}` (+ the PDF slicing sizes when slicing is on) —
  folded into `render_version`.

`render_version` is a stable short hash of both tiers (orjson with sorted keys,
byte-identical to the stdlib `json.dumps(sort_keys=True, separators=(",", ":"))` it
replaced — a test pins that), stamped on each doc and on `service_meta`. It is as
fundamental as geometry: **the tiler never serves a doc rendered by another
`render_version`**. Unlike geometry, a change is not fatal — the docs keep their blobs,
so the service starts *not serving* and re-renders them first (§9 "Revision"). Library versions are in the pixel tier because a Pillow
upgrade changes JPEG bytes exactly like a quality change does.
`TILER_CHROMIUM_REVISION` defaults to the revision Playwright has pinned (read from its
`browsers.json` without launching a browser), so the value is stable even in a PDF-only
deployment.

`image_hash` (stored at ingest or revision) is the anchor: a tile request (`verify=true`, the default)
recomputes and compares. Since every served doc was rendered by the current
`render_version`, a mismatch can only be non-determinism (web/MHTML re-render drift, a
base-image change `render_version` does not see) and always answers `409`. PDF/IMG are
byte-deterministic, so a mismatch there is a bug.

---

## 8. Rendering determinism (Chromium)

Pin the Chromium build (`TILER_CHROMIUM_REVISION`, baked in; no auto-update). Launched
headless via Playwright (`headless=True` runs the lightweight `chromium-headless-shell`;
the full browser is not shipped). Install deterministic fonts (`fonts-noto`,
`fonts-noto-cjk`, `fonts-noto-color-emoji`, `fonts-liberation`) — from a dated Debian
snapshot tied to the pinned base image, so they do not drift between builds (fonts are
not part of `render_version`; see [DEPLOYMENT.md](DEPLOYMENT.md)). Viewport width
`TILER_TILE_WIDTH`, `device_scale_factor=1`, `TZ=UTC`, fixed locale. Disable
animations; wait for `document.fonts.ready` + optional network-idle.

The determinism CSS is injected via `page.evaluate` (building a `<style>` node in-page),
**not** `page.add_style_tag` — the latter waits for a load event that never fires on a
static MHTML archive and would stall every render. Launch flags:
`--hide-scrollbars --disable-gpu --force-color-profile=srgb --font-render-hinting=none
--disable-lcd-text --disable-dev-shm-usage --mute-audio` plus background-throttle
disables.

For `doc_type=URL` the browser snapshots the page to MHTML first, and that blob — not
the live page — is what gets tiled, so ingest and a tile request render the same source.
Tall pages (> `TILER_SAFE_FULLPAGE_PX`) are captured per-strip via
`page.screenshot(clip=...)` with a viewport-relative clip, scrolling into view so lazy
content rasterizes. Ingest and regeneration call the *same* functions in
`render/pixels.py`.

---

## 9. Idempotency, dedup, deletion

- Idempotent ingest keyed by `doc_id = sha256(blob)`; identical bytes answer
  `result=idempotent` (the stream replays the stored tiles) without re-render, and
  their metadata is left as it is. A quarantined doc is rendered again, which lifts
  the quarantine.
- Blob commit skipped if the path exists.
- URL caveat: re-fetching a changed URL yields a new `doc_id`; `source_url` is
  traceability, not identity.
- `DELETE /doc/{doc_id}` (optional, flag-guarded by `TILER_ALLOW_DELETE`):
  cascade-delete rows + unlink blob.

### An ingest, step by step

`POST /doc`, with `tiles=true` or without (see [API.md](API.md)), runs one pipeline:

1. **Spool.** The upload goes to `.incoming`, hashed on the way: `doc_id`.
2. **Known already?** A doc that is stored and servable is not rendered: its tiles are
   replayed (`tiles=true`) or its summary returned. The replay — the tiles rendered
   again, each verified against its hash — is also what `GET /doc/{id}/tiles` answers
   to a consumer that has the id but not the bytes.
3. **Row.** A new doc's row is inserted, `complete=0`, without tiles. The primary key
   settles a race: of two requests bringing the same bytes one inserts, the other is
   refused (`409 INGEST_IN_PROGRESS`), as is any later one until the first ends.
4. **Render** from the spooled file. Tile rows are kept in memory; the stream route
   sends each tile as it renders.
5. **Commit.** The blob is renamed into the store, then one transaction writes every
   tile row, the height and `complete=1`.

- An **incomplete doc is listed but not served**: its tiles (`GET /doc/{id}/tiles`,
  `GET /doc/{id}/tiles/{n}`) and `GET /doc/{id}` answer `404 DOC_INCOMPLETE`, `DELETE` answers `409` — it belongs to its ingest. A
  consumer can already refer to it (its metadata can be patched).
- **An ingest that does not commit leaves nothing**: on an error or a client that went
  away it removes its upload and the row it inserted, even while being cancelled; what
  a killed process left (`complete=0` rows, `.incoming`) is removed at startup, when no
  ingest can be alive.
- **Known limitation: a client that vanishes without closing the connection.** "Went
  away" above is what the tiler can see — a closed or reset connection. A stream's
  client whose host or network went down says nothing: the tiler keeps writing into
  the connection, and only the kernel's own give-up (about 15–30 min with Linux's
  default `tcp_retries2=15`) ends the ingest. Until then the doc stays incomplete and
  the same bytes are refused with `409 INGEST_IN_PROGRESS`; it clears by itself, and a
  restart of the tiler clears it at once. Left as it is on purpose: the socket-level
  cure (`TCP_USER_TIMEOUT`) would also cut a client that is alive and merely slow to
  read.
- **A stored doc is never touched before a commit.** The one stored doc that is
  rendered again is a quarantined one; its row and tiles stay as they are until the
  new render commits, so a failed attempt costs nothing.

### Revision (re-render before serving)

`tiler/revision.py` owns a **serving gate** and the job that opens it. At startup the
tiler counts the *revisable* docs: complete, rendered by another `render_version`, not
already quarantined at this one. None → it serves. Otherwise data routes answer
`503 NOT_READY` (a router dependency; ops routes and `/revision` stay up) and the job runs
— at once with `TILER_REVISION_ON_RENDER_MISMATCH=true`, else on `POST /revision`:

1. **Backup.** The DB is copied with `VACUUM INTO` (via a `.partial` file, then renamed)
   to `<db>.bak-<service_meta.render_version>`, once per source version — a resumed job
   keeps the first copy. Old image + backup is the rollback; blobs are never touched.
2. **Re-render.** Revisable docs are read keyset-paged by `doc_id` (stable while
   re-rendered docs leave the set) and re-rendered from their blobs,
   `TILER_REVISION_CONCURRENCY` at a time. Each doc's tiles, `tile_count`, `height`
   and `render_version` are swapped in **one transaction** (`replace_render`); `doc_id`,
   metadata and identity stay as they are. Revision renders do not take bulk slots —
   nothing interactive runs meanwhile.
3. **Quarantine.** A doc that fails twice goes to the `quarantine` table (its own table,
   so existing volumes need no migration) with the reason and the `render_version` it
   failed at; its tiles answer `410`. A re-ingest of its bytes, a forced revision, or a
   revision at another `render_version` retries it. A render-worker crash or an
   unavailable Chromium is not the doc's fault: it aborts the job (`failed`, still not
   serving) instead of quarantining everything after it.
4. **Open.** With nothing revisable left, `service_meta.render_version` is stamped and
   the gate opens.

Incomplete docs do not outlive a start, so a revision never meets one. Selection by `render_version`
makes the job resumable for free; a forced one (`force=true`: every complete doc) is
not. Progress is in `GET /revision` and a `done/total, rate, ETA` log line every 30 s.
Because nothing is served or written during the job, it races with nothing: no metadata
PATCH to lose, no ingest in flight, no cached tile of the old layout (the process starts
with empty caches). The gate, like the caches, lives in the one tiler process. The render path is counted under `path="revise"` in the metrics; it never
touches any downstream (Qdrant).

### Bulk slots (ingest vs a tile request)

The render pool and the browser pool are FIFO and shared. Background work (an ingest,
URL capture, a stored doc's stream `GET /doc/{id}/tiles`) holds at most `TILER_N_BULK_RENDER_WORKERS` pool tasks and
`TILER_N_BULK_BROWSERS` Chromium renders at once (default: half of each pool), so a
900-page PDF fanning out over the pool never queues an interactive tile request behind all
its chunks. Bulk slots are granted in task-creation order, so the in-order consumer of a
sliced PDF stream never waits on a group that is itself waiting for a slot.

---

## 10. Module layout

```
tiler/
  __init__.py        # routes sqlite3 → pysqlite3 (STAT4) before aiosqlite loads; __version__
  main.py            # app, lifespan (opens the volume, wires the components below, starts the pools and the revision gate), routes
  ingest.py          # Ingester: a document's way into the store (spool, render, one commit) and out of it
  events.py          # what a tile stream consists of: Begin, Tile, Committed (Error on the wire); an event's NDJSON line
  tiles.py           # Tiles: a stored doc's tiles by its id, rendered from its blob (one tile, or its event stream); the tile cache
  capture.py         # URL capture: SSRF policy + guard proxy + MHTML snapshot
  metadata.py        # the declared metadata schema: load, meta_expr(), validation, CREATE INDEX statements
  revision.py        # serving gate + the job that brings stored docs to the current render_version
  config.py          # the TILER_* variables in sections (geometry, pixel, render, store, capture, revision, api) + render_version
  volume.py          # Volume.open: the DB and the blob dir opened, bound to their geometry (service_meta), indexed by the schema
  preflight.py       # standalone startup validation (runs before uvicorn): opens the volume as the service will
  media.py  metrics.py  errors.py  models.py  responses.py  # mime sniff, Prometheus, error envelope, pydantic, orjson response
  api/  deps.py  ingest.py  tiles.py  docs.py  metadata.py  search.py  revision.py  ops.py
  render/ renderer.py   # Renderer: the one way from a document's file to its tiles; records the renders
          documents.py  # what gets rendered: WorkerDocument (PDF, images), MhtmlDocument (Chromium)
          readers.py    # DocumentReader: PdfReader, ImageReader — a format read as one image, inside a worker
          pool.py       # RenderPool: pebble process pool (memory cap, timeout), Chromium, bulk slots
          worker.py     # what runs in a worker: lay a document out, read and encode its tiles
          browser.py  pixels.py  geometry.py  types.py  # Chromium (BrowserPool loads a document, Tab captures it); pixel primitives; Tile/Tiling; result dataclasses
  store/  blobs.py  meta.py  cache.py
  ssrf.py  ssrf_proxy.py  hashing.py               # URL policy + the proxy that pins Chromium to it + sha256
```

`meta_expr()` is the single source of the JSON extraction expression used by both index
creation and query building. `tini` is PID 1 to reap Chromium's zombie renderers.
