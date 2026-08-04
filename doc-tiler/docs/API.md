# API guide

Human-oriented tour of the HTTP surface. The **authoritative wire contract is
[`openapi.yaml`](openapi.yaml)** — request/response schemas, field types and every
parameter live there and are served interactively at `GET /docs` (Swagger UI) on a
running instance. This page explains what each endpoint is *for* and the semantics that
a schema can't convey.

Base URL in examples: `http://localhost:8080`.

---

## Documents

### `POST /tile` — ingest, tile, optionally store

`multipart/form-data`. Renders a document at the fixed geometry and returns the tile
manifest.

Body fields:

- `doc_type` — `PDF | URL | MHTML | IMG` (required).
- `document` — the file bytes (required unless `doc_type=URL`).
- `url` — the URL (required iff `doc_type=URL`).
- `metadata` — a JSON **object** serialized as a string (optional).

Query flags:

- `store` (default `true`) — persist blob + rows + metadata so tiles are regenerable
  via `/get_tile`. `store=false` is a **pure transform**: it slices and returns tiles,
  writes nothing, ignores `metadata`, forces `include_bytes=true`, and a later
  `/get_tile` for that `doc_id` will `404`.
- `include_bytes` (default `true`) — include base64 JPEG bytes per tile.
- `force` (default `false`) — re-render even if this `doc_id` already exists.

**Idempotent** on `doc_id = sha256(blob)`: identical bytes return the stored manifest
without re-rendering (unless `force`). The response is an `IngestManifest`: `doc_id,
doc_type, source_url, render_version, stored, page_height, tile_count, tiles[]`, each
tile `{tile_num, width, height, image_hash, data?}` (`data` = base64 JPEG when
`include_bytes=true`).

```bash
curl -s -X POST 'localhost:8080/tile?include_bytes=false' \
     -F doc_type=PDF -F document=@paper.pdf \
     -F 'metadata={"project":"atlas","year":2024,"source":{"host":"arxiv.org"}}' | jq
```

> Base64 adds ~33% overhead; JSON+base64 is the canonical contract. A
> `multipart/mixed` variant may be offered later via `Accept`.

### `GET /get_tile?doc_id=&tile_num=&verify=` — regenerate one tile

Deterministically regenerates a single tile from the stored original and returns
`image/jpeg` with `X-Doc-Id` / `X-Tile-Num` / `X-Image-Hash` / `X-Tile-Height` /
`X-Tile-Width` headers. `verify=true` recomputes the hash and compares against the value
recorded at ingest; on mismatch the response follows `TILER_RENDER_MISMATCH_POLICY`
(`strict` → `409`, `lenient` → serve fresh + warn).

```bash
curl -s 'localhost:8080/get_tile?doc_id=<sha256>&tile_num=3' -o tile3.jpg
```

### `GET /doc/{doc_id}` — download the original

Returns the stored blob with its original `Content-Type` and
`Content-Disposition: attachment`. HTTP `Range` is supported (`416` if not satisfiable).

### `DELETE /doc/{doc_id}` — delete (flag-guarded)

Cascade-deletes rows and unlinks the blob. Disabled by default:
`405` unless `TILER_ALLOW_DELETE=true`.

---

## Metadata

Metadata is a mutable JSON object over immutable content — it never affects `doc_id`.
Only keys **declared in the schema** are searchable; see [CONFIG.md](CONFIG.md).

- `GET /doc/{doc_id}/metadata` → the JSON object (or `{}`).
- `PUT /doc/{doc_id}/metadata` → replace the whole object (validated).
- `PATCH /doc/{doc_id}/metadata` → shallow merge; incoming keys overwrite, a `null`
  value deletes a key (validated).

Writes update `docs.metadata`; the expression indexes update automatically — no manual
reindex.

### `POST /docs/search` — search and listing (one endpoint)

Body: `{filters?, limit, cursor}`.

- Filters are `{key, op, value}`, AND-combined. `op ∈ {eq, ne, gt, gte, lt, lte}`;
  `gt/gte/lt/lte` require a `number` field.
- **Every filter key must be declared** (i.e. indexed). An undeclared key → `400`
  naming it. There is no fallback scan.
- **No filters / omitted** → list all documents, newest first.
- Cursor pagination on `(created_at, doc_id)`; response `{items: DocSummary[],
  next_cursor}`.

```bash
curl -s -X POST localhost:8080/docs/search -H 'content-type: application/json' \
     -d '{"filters":[{"key":"year","op":"gte","value":2020}],"limit":20}' | jq
```

---

## Operational

| Endpoint | Purpose |
|---|---|
| `GET /schema/metadata` | The operator-declared metadata schema (which keys are searchable). |
| `GET /info` | Effective geometry, `render_version`, pixel tier, policies, pool + cache stats, doc count. |
| `GET /healthz` | Liveness — touches nothing, stays fast under load. |
| `GET /readyz` | Readiness — checks DB, blob dir, Chromium pool; `503` when not ready. |
| `GET /metrics` | Prometheus metrics (see [METRICS.md](METRICS.md)). |

---

## Error codes

Uniform JSON error envelope (the `Error` schema in `openapi.yaml`).

| Code | When |
|---|---|
| `400` | Unknown `doc_type`, `tile_num` out of range, undeclared filter key, unsupported op, numeric op on a text field, bad cursor. |
| `403` | SSRF policy block. |
| `404` | Unknown `doc_id` (including `store=false` docs). |
| `405` | `DELETE` with `TILER_ALLOW_DELETE=false`. |
| `409` | `image_hash` mismatch under `strict` policy. |
| `413` | Upload exceeds `TILER_MAX_DOC_BYTES`. |
| `415` | Content-type mismatch. |
| `416` | `Range` not satisfiable (`GET /doc`). |
| `422` | Corrupt document, invalid metadata JSON, schema violation, request validation. |
| `502` | Upstream fetch/render failure. |
| `504` | Render timeout. |

**Startup** is separate: a geometry- or binding-secret mismatch makes the process exit
non-zero before serving anything (see [ARCHITECTURE.md](ARCHITECTURE.md) and
[SECURITY.md](SECURITY.md)). All 5xx from rendering are caught so one bad document never
crashes a worker or the event loop.
