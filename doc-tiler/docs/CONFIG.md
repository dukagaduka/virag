# Configuration

Every runtime knob is an environment variable prefixed **`TILER_`** (so it never
collides with unrelated env in a shared container). Non-config system vars the image
sets — `TZ`, `LANG`, `HOME`, `PLAYWRIGHT_BROWSERS_PATH`, the `APP_UID`/`APP_GID`
identity constants — are not part of this namespace.

A ready-to-edit template lives at [`deploy/.env.example`](../deploy/.env.example).

---

## Geometry tier — fixed for the life of the volume

These four define the tiling geometry. A DB/volume is permanently bound to the geometry
it was created with; booting an existing volume with different values is a **fatal
startup error** (see [ARCHITECTURE.md](ARCHITECTURE.md), "Geometry guard"). To change
them, deploy against a fresh volume and re-ingest.

| Variable | Default | Meaning |
|---|---|---|
| `TILER_TILE_WIDTH` | `875` | Target width (px). Matches the Qwen3-VL embedder. |
| `TILER_TILE_HEIGHT` | `1024` | Max strip height (px). |
| `TILER_TILE_OVERLAP` | `0` | Overlap between strips. `STEP = HEIGHT − OVERLAP`; `0 ≤ OVERLAP < HEIGHT`. |
| `TILER_MIN_TILE_HEIGHT` | `28` | Drop final strips shorter than this. |
| `TILER_PDF_SLICE_ENABLE` | `false` | Slice each PDF page into strips (within page bounds) instead of one tile per page. Off = one tile per page. On (or changing the sizes below) alters `render_version` → corpus re-renders. |
| `TILER_PDF_TILE_HEIGHT` | `1024` | Strip height (px) for PDF slicing; used only when slicing is on. `0 < HEIGHT`. |
| `TILER_PDF_TILE_OVERLAP` | `0` | Overlap between PDF strips; `0 ≤ OVERLAP < TILER_PDF_TILE_HEIGHT`. |

## Pixel tier — folded into `render_version` (soft check)

Changing these does not stop startup, but regenerated tiles may no longer match the
`image_hash` recorded at ingest — governed by `TILER_RENDER_MISMATCH_POLICY`.

| Variable | Default | Meaning |
|---|---|---|
| `TILER_JPEG_QUALITY` | `85` | Output JPEG quality. |
| `TILER_PDF_DPI` | `200` | PDF rasterization DPI. |
| `TILER_CHROMIUM_REVISION` | Playwright's pin | Exact Chromium build; read from Playwright's `browsers.json` without launching. |

(JPEG subsampling 4:2:0, progressive off, and `LANCZOS` resize are pinned in code;
Pillow and PyMuPDF versions are also folded into `render_version`.)

## Rendering & capture

| Variable | Default | Meaning |
|---|---|---|
| `TILER_SAFE_FULLPAGE_PX` | `30000` | Above this page height, capture strip-by-strip via clip. |
| `TILER_NAV_TIMEOUT_MS` | `30000` | Chromium nav/render timeout. |
| `TILER_WAIT_NETWORK_IDLE` | `true` | Wait for network-idle before capture. |
| `TILER_BROWSER_ENABLED` | `true` | Launch the Chromium pool. `false` disables URL/MHTML rendering (PDF/IMG-only or test runs). |

## Concurrency & caching

| Variable | Default | Meaning |
|---|---|---|
| `TILER_N_BROWSERS` | `min(cpu, 4)` | Max concurrent Chromium renders (semaphore). |
| `TILER_N_RENDER_WORKERS` | `cpu_count` | Process-pool size (PDF raster / PIL / hashing). |
| `TILER_RECYCLE_AFTER` | `500` | Recycle a Chromium process after N renders. |
| `TILER_RENDER_CACHE_SIZE` | `64` | LRU capacity for `get_tile`. |

## Limits & policy

| Variable | Default | Meaning |
|---|---|---|
| `TILER_MAX_DOC_BYTES` | `67108864` | Reject larger uploads (`413`). |
| `TILER_RENDER_MISMATCH_POLICY` | `strict` | `strict` → `409` on `image_hash` mismatch; `lenient` → serve + warn. |
| `TILER_ALLOW_DELETE` | `false` | If `false`, `DELETE /doc/{id}` → `405`. |
| `TILER_URL_FETCH_ENABLED` | `true` | Allow `URL` input type. |
| `TILER_URL_ALLOW_PRIVATE` | `false` | If `false`, SSRF-block private targets (see [SECURITY.md](SECURITY.md)). |

## Incomplete-ingest sweeper

A streaming ingest (`POST /tile/stream`) persists a doc incrementally and leaves it
incomplete (`complete=0`) if it crashes mid-stream. This background task reclaims that
storage. See [ARCHITECTURE.md](ARCHITECTURE.md) ("Incomplete-ingest sweeper").

| Variable | Default | Meaning |
|---|---|---|
| `TILER_INCOMPLETE_TTL_SEC` | `0` | Delete incomplete docs older than this many seconds. `0` disables the sweeper. **Must exceed the longest render** so an in-flight ingest is never reaped (an active ingest is also excluded in memory). |
| `TILER_SWEEP_INTERVAL_SEC` | `300` | How often (seconds) the sweeper runs. |

## State (inside the container / named volume)

| Variable | Default | Meaning |
|---|---|---|
| `TILER_DB_PATH` | `/data/meta.sqlite` | SQLite file. |
| `TILER_BLOB_DIR` | `/data/blobs` | Content-addressed blob root. |
| `TILER_TMP_DIR` | `<TILER_BLOB_DIR>/../tmp` | Scratch dir for MHTML archive temp files. |
| `TILER_METADATA_SCHEMA_PATH` | `/config/metadata_schema.yaml` | Declared indexed-metadata schema (below). |
| `TILER_LOG_LEVEL` | `INFO` | Logging level. |

## Binding secret

| Variable | Default | Meaning |
|---|---|---|
| `TILER_DB_SECRET` | `""` | Brands this volume so only the matching consumer/vector-DB pair connects. Set **before first init** (immutable per volume, like geometry); empty = unbindable slot. See [SECURITY.md](SECURITY.md). |

---

## Metadata schema file

`TILER_METADATA_SCHEMA_PATH` points at a mounted YAML file declaring which metadata
keys are **searchable**. Only declared keys get an expression index; any other key you
send is still stored verbatim in `docs.metadata`, just not searchable. The schema is
loaded and validated at startup and exposed read-only via `GET /schema/metadata`.

```yaml
# metadata_schema.yaml
version: 1
fields:
  - key: project        # dotted JSON path into the metadata object
    type: text          # text | number
    required: false
  - key: source.host    # dotted keys index nested JSON: $.source.host
    type: text
  - key: year
    type: number
```

- `key` must be a dotted plain identifier (it is interpolated into the SQL/JSON path).
- `type`: `text` (strings; numbers coerced with `str()`) or `number` (int/float and
  numeric strings). Booleans, objects and arrays are rejected on write (`422`).
- `required: true` → the key must be present and non-`null` on every write, else `422`.
- **Adding a field later** = add it here and restart; SQLite backfills the index over
  existing rows automatically. Removing a field drops its stale index at startup. No
  admin/reindex endpoint exists or is needed.

See [ARCHITECTURE.md](ARCHITECTURE.md) ("Metadata: declared schema + expression
indexes") for how the indexes and the query planner interact, including the `pysqlite3`
/ STAT4 requirement for selective numeric ranges.
