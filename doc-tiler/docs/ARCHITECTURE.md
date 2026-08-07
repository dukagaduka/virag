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
embedding stage stores as its vector-DB payload to later call `/get_tile`.

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

### Layer 3 — CPU (PDF raster, PIL resize/slice, hashing), CPU-bound

Done inline it blocks the loop; in threads the GIL kills parallelism. So:
`ProcessPoolExecutor(max_workers=TILER_N_CPU_WORKERS)` — separate processes, true
multi-core. Dispatch via `loop.run_in_executor(process_pool, fn, ...)` and `await`.

The CPU pool uses the **`spawn`** start method: the parent runs an event loop and
Playwright threads, and forking those into workers deadlocks.

### Why not simpler

All-threads ⇒ CPU-bound + GIL = no speedup. One pool including the browser ⇒ Chromium
is expensive to start; keep it warm and shared behind a semaphore.

### Flow examples

- **PDF ingest:** handler (loop) reads bytes → process pool rasters + slices → await →
  DB → stream response. Pages fan out across the pool in chunks of ≤16, so a 200-page
  PDF uses every core without pickling one giant result.
- **URL ingest:** handler takes the browser semaphore → Playwright navigates +
  `captureSnapshot(mhtml)` → releases → process pool slices the snapshot → await → DB
  → response.

### `get_tile` cache

In-process **LRU** (`TILER_RENDER_CACHE_SIZE`), async-safe, with **in-flight
coalescing** — a per-key future map so concurrent requests for the same tile render
once.

- Key `(doc_id, tile_num)` → final JPEG bytes (all types).
- Web/MHTML additionally caches `doc_id` → full-page image so sequential strip
  requests reuse one render.

---

## 3. Data model (SQLite)

Startup pragmas: `journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`,
`foreign_keys=ON`. SQLite runs on a single connection serialized by one lock. The
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
    secret_hash     TEXT    NOT NULL DEFAULT '',  -- binding secret (see SECURITY.md)
    initialized_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS docs (
    doc_id         TEXT PRIMARY KEY,     -- sha256(blob) hex
    doc_type       TEXT NOT NULL,        -- 'PDF' | 'MHTML' | 'IMG' (URL stored as MHTML)
    blob_rel_path  TEXT NOT NULL,        -- 'ab/cd/<hash>' relative to TILER_BLOB_DIR
    source_url     TEXT,                 -- original URL if fetched, else NULL
    page_height    INTEGER NOT NULL,     -- full-page height (web/mhtml/img); 0 for PDF
    tile_count     INTEGER NOT NULL,
    mime_type      TEXT NOT NULL,
    metadata       TEXT,                 -- raw JSON object (single source of truth); NULL if none
    render_version TEXT NOT NULL,        -- stamp (informational; geometry uniform by the startup guard)
    complete       INTEGER NOT NULL DEFAULT 1,  -- 0 while a streaming ingest is mid-flight; see §9
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tiles (
    doc_id      TEXT NOT NULL,
    tile_num    INTEGER NOT NULL,        -- 0-based
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

The `complete` column **auto-migrates** on existing volumes: an `ALTER TABLE docs ADD
COLUMN complete INTEGER NOT NULL DEFAULT 1` runs at startup, and pre-existing rows
backfill to `complete=1` (they were written atomically). A non-streaming `POST /tile`
always writes `complete=1` (atomic full write); a streaming ingest writes the row
`complete=0` and flips it on the final tile (see §9).

Everything storage-related sits behind `BlobStore` / `MetaStore` interfaces, keeping
the FS→S3 and SQLite→Postgres swaps to a driver change.

---

## 4. Blob storage (content-addressed)

- Root `TILER_BLOB_DIR`; layout `<hash[0:2]>/<hash[2:4]>/<hash>`, `hash = doc_id`.
- **Write-once, atomic:** temp file in the same dir → `fsync` → `os.rename`. If the
  final path exists, skip (dedup).
- SQLite holds metadata only; blobs never in the DB.
- Deliberately **isomorphic to S3** (`ab/cd/<hash>` → bucket key).

**The blob is written before the DB row** (the reverse of the naive order). A blob
without a row is inert and dedups away on the next ingest; a row without a blob is a
permanently broken `/get_tile`.

---

## 5. Tiling geometry (exact)

`W = TILER_TILE_WIDTH`, `TH = TILER_TILE_HEIGHT`, `OV = TILER_TILE_OVERLAP`,
`STEP = TH - OV`, `MIN = TILER_MIN_TILE_HEIGHT`.

### 5.1 Web / MHTML / IMG — portrait strips

Full-page image of width exactly `W`; `H = full_page_height`.

```
tiles = []; offset = 0
while True:
    h = min(TH, H - offset)
    if h >= MIN: tiles.append({tile_num: len(tiles), offset, height: h})
    if offset + TH >= H: break
    offset += STEP
tile_count = len(tiles)
```

**Reverse (get_tile):** `offset = tile_num * STEP`, `height = min(TH, H - offset)`.
Consistent because a sliver is only dropped as the final element. Forward (ingest) and
reverse (get_tile) must produce byte-identical bounds or `image_hash` will not match —
this is pinned by a property test over random `H`. `page_height = H` stored.

A page shorter than `MIN` yields **zero tiles** (the only strip is a sliver, and
slivers are dropped); such a doc is stored with `tile_count = 0`.

### 5.2 PDF — one tile per page

Raster page `p` at `TILER_PDF_DPI` → `(w0, h0)` → proportional resize to width `W`
(`new_h = round(h0 * W / w0)`, LANCZOS). `tile_num = p`, `height = new_h`, no slicing.
`tile_count = pages`, `page_height = 0`.

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
if row.render_version != current_render_version:          # pixel tier (soft)
    log.WARNING("Pixel params changed since init; regenerated tiles may not match "
                "stored image_hash — governed by TILER_RENDER_MISMATCH_POLICY.")
```

Geometry mismatch is caught **once, globally, before serving anything** — there is no
per-request geometry gate. To change geometry, deploy against a fresh volume and
re-ingest. The guard runs both in `lifespan` and standalone as `python -m
tiler.preflight`, which the container entrypoint runs before uvicorn so a mismatched
volume produces one clear log line instead of a generic ASGI startup failure.

The `secret_hash` column carries the **binding secret** that pairs this volume with a
specific consumer/vector-DB; its guard behaves like the geometry guard (immutable per
volume, fatal on mismatch). See [SECURITY.md](SECURITY.md).

### `render_version` and pixel-tier drift (soft)

- **Geometry tier** = `{TILER_TILE_WIDTH, TILER_TILE_HEIGHT, TILER_TILE_OVERLAP,
  TILER_MIN_TILE_HEIGHT}` — enforced hard at startup.
- **Pixel tier** = `{TILER_JPEG_QUALITY, JPEG_SUBSAMPLING (4:2:0), JPEG_PROGRESSIVE
  (off), TILER_PDF_DPI, resize_filter=LANCZOS, TILER_CHROMIUM_REVISION,
  pillow_version, pymupdf_version}` — folded into `render_version`, checked softly.

`render_version` is a stable short hash of both tiers, stored once in `service_meta`
and stamped on each doc. Library versions are in the pixel tier because a Pillow
upgrade changes JPEG bytes exactly like a quality change does.
`TILER_CHROMIUM_REVISION` defaults to the revision Playwright has pinned (read from its
`browsers.json` without launching a browser), so the value is stable even in a PDF-only
deployment.

`image_hash` (stored at ingest) is the anchor: `get_tile?verify=true` recomputes and
compares, catching web/MHTML re-render drift or a Chromium/JPEG/DPI change.
`TILER_RENDER_MISMATCH_POLICY`: `strict` → `409`; `lenient` → serve fresh render + warn
(use during a deliberate Chromium upgrade). PDF/IMG are byte-deterministic, so a
mismatch there is a bug.

---

## 8. Rendering determinism (Chromium)

Pin the Chromium build (`TILER_CHROMIUM_REVISION`, baked in; no auto-update). Launched
headless via Playwright (`headless=True` runs the lightweight `chromium-headless-shell`;
the full browser is not shipped). Install deterministic fonts (`fonts-noto`,
`fonts-noto-cjk`, `fonts-noto-color-emoji`, `fonts-liberation`). Viewport width
`TILER_TILE_WIDTH`, `device_scale_factor=1`, `TZ=UTC`, fixed locale. Disable
animations; wait for `document.fonts.ready` + optional network-idle.

The determinism CSS is injected via `page.evaluate` (building a `<style>` node in-page),
**not** `page.add_style_tag` — the latter waits for a load event that never fires on a
static MHTML archive and would stall every render. Launch flags:
`--hide-scrollbars --disable-gpu --force-color-profile=srgb --font-render-hinting=none
--disable-lcd-text --disable-dev-shm-usage --mute-audio` plus background-throttle
disables.

For `doc_type=URL` the browser snapshots the page to MHTML first, and that blob — not
the live page — is what gets tiled, so ingest and `/get_tile` render the same source.
Tall pages (> `TILER_SAFE_FULLPAGE_PX`) are captured per-strip via
`page.screenshot(clip=...)` with a viewport-relative clip, scrolling into view so lazy
content rasterizes. Ingest and regeneration call the *same* functions in
`render/cpu.py`.

---

## 9. Idempotency, dedup, deletion

- Idempotent ingest keyed by `doc_id = sha256(blob)`; identical bytes return the stored
  manifest without re-render (unless `force`).
- Blob write skipped if the path exists.
- URL caveat: re-fetching a changed URL yields a new `doc_id`; `source_url` is
  traceability, not identity.
- `DELETE /doc/{doc_id}` (optional, flag-guarded by `TILER_ALLOW_DELETE`):
  cascade-delete rows + unlink blob.

### Incomplete-doc lifecycle (streaming ingest)

`POST /tile/stream` (see [API.md](API.md)) persists a doc **incrementally**: it writes the
`docs` row as `complete=0`, appends `tiles` rows as they render, and flips the row to
`complete=1` on the final tile. `POST /tile` (non-streaming) always writes `complete=1`
(one atomic full write).

- An **incomplete (`complete=0`) doc is not servable**: `GET /get_tile` on it returns
  `404` with error code `DOC_INCOMPLETE`.
- The **idempotency short-circuit only fires for complete docs** — an incomplete (crashed
  or in-flight) doc is re-rendered rather than replayed. So a stream that dies mid-way
  leaves a swept-able partial, never a servable half-doc.
- `DocSummary` (search results) exposes `complete` and `render_version` so an operator can
  see partials.

### Incomplete-ingest sweeper (background)

A background task (started in `lifespan`, interval `TILER_SWEEP_INTERVAL_SEC`) periodically
deletes incomplete (`complete=0`) docs older than `TILER_INCOMPLETE_TTL_SEC`, **excluding
any ingest currently in flight** (tracked in memory). Deletion is whole-doc (blob + rows).
This reclaims storage from crashed/abandoned streaming ingests; because an incomplete doc
was never fully enrolled downstream, reaping it is safe. The TTL defaults to `0` (disabled)
and **must exceed the longest render** so an active render is never reaped.

### Revision (re-render in place)

`POST /revision` re-renders, from their stored original blobs, every doc whose stored
`render_version` differs from the current one (or all docs with `force=true`), plus any
incomplete doc. Tiles are replaced atomically in place — `doc_id` is stable — so after a
pixel-tier change `get_tile` reproduces at the current `render_version`. It touches only
the tiler's own tiles, never any downstream (Qdrant); the `/revision` render path is
counted under `path="revise"` in the metrics.

---

## 10. Module layout

```
tiler/
  __init__.py        # routes sqlite3 → pysqlite3 (STAT4) before aiosqlite loads; __version__
  main.py            # app, lifespan (guard, pools, DB, schema+index creation), routes
  service.py         # core: ingest, regeneration, the three-layer pool wiring
  config.py          # env + render_version + geometry/pixel tiers
  guard.py           # startup geometry + binding-secret guard (service_meta)
  preflight.py       # standalone startup validation (runs before uvicorn)
  media.py  metrics.py  errors.py  models.py       # mime sniff, Prometheus, error envelope, pydantic
  api/  deps.py  ingest.py  tiles.py  docs.py  metadata.py  search.py  ops.py
  render/ browser.py  cpu.py  geometry.py          # cpu = PDF raster/PIL/hash; geometry = tiling forward+reverse
  store/  blobs.py  meta.py  cache.py
  meta/   schema.py                                # load schema, meta_expr(), CREATE INDEX loop
  ssrf.py  netguard.py  hashing.py                 # URL policy + pinning proxy + sha256
```

`meta_expr()` is the single source of the JSON extraction expression used by both index
creation and query building. `tini` is PID 1 to reap Chromium's zombie renderers.
