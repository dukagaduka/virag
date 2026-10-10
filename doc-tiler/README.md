# Tiler Service

Turns **PDF / URL / MHTML / IMG** documents into fixed-geometry image tiles
(875 × ≤1024 JPEG) for a pixel-based (screenshot) RAG embedder, and regenerates any
single tile on demand from the stored original. Tiles are never persisted — the original
blob plus deterministic rendering is the source of truth. It also keeps
operator-declared metadata with fast indexed search.

**Scope:** ingest (one-shot or streaming), blob storage, deterministic tiling, tile
regeneration, in-place re-render (revision), original download, metadata search. **Not**
in scope: embedding, the vector DB, retrieval, the LLM/VLM reader, the RAG gateway. The `(doc_id, tile_num)` pair returned with every tile
is the contract the embedding stage stores as its payload to later call `/get_tile`.

---

## Quickstart

```bash
# optional but recommended: vendor Chromium before building so the build needs no CDN
python -m playwright install chromium-headless-shell && ./scripts/vendor-chromium.sh

cd deploy
cp .env.example .env          # set TILER_DB_SECRET, adjust geometry/pixel if needed
docker compose up --build
curl -s localhost:8080/readyz | jq

# ingest a PDF, store it, get the manifest without the base64 payload
curl -s -X POST 'localhost:8080/tile?include_bytes=false' \
     -F doc_type=PDF -F document=@paper.pdf \
     -F 'metadata={"project":"atlas","year":2024,"source":{"host":"arxiv.org"}}' | jq

# regenerate one tile
curl -s 'localhost:8080/get_tile?doc_id=<sha256>&tile_num=3' -o tile3.jpg

# search by a declared key
curl -s -X POST localhost:8080/docs/search -H 'content-type: application/json' \
     -d '{"filters":[{"key":"year","op":"gte","value":2020}],"limit":20}' | jq
```

Interactive API docs: <http://localhost:8080/docs>. Runtime introspection (geometry,
`render_version`, cache stats, pool sizes): `GET /info`.

## Documentation

| Doc | What's in it |
|---|---|
| [docs/API.md](docs/API.md) | Endpoint guide, request/response semantics, error codes. |
| [docs/openapi.yaml](docs/openapi.yaml) | The authoritative wire contract (served at `/docs`). |
| [docs/CONFIG.md](docs/CONFIG.md) | Every `TILER_*` variable + the metadata schema file. |
| [docs/METRICS.md](docs/METRICS.md) | Prometheus metrics at `/metrics`, alerts. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Concurrency model, storage, geometry, guard, determinism — the rationale. |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Image, Compose, volume, non-root runtime, scaling. |
| [docs/SECURITY.md](docs/SECURITY.md) | Binding secret, SSRF protection, container hardening. |

## Design invariants

1. **Geometry is fixed and global** (`875 × 1024`), matching the Qwen3-VL embedder. A
   volume is bound to one geometry for life.
2. **Tiles are never persisted** — the original blob plus deterministic rendering is the
   source of truth.
3. **`doc_id = sha256(blob)`** — content-addressed, so ingest is idempotent and
   deduplicated.
4. **A tile is `(doc_id, tile_num)`** — `tile_num` is 0-based; there is no separate tile
   id.
5. **Metadata never affects `doc_id`** — mutable annotation over immutable content.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning and the full data
model.

## Layout

```
tiler/            the service package
  main.py         app + lifespan (guard → indexes → pools), routes
  service.py      core: ingest, regeneration, the three-layer pool wiring
  config.py       env, geometry/pixel tiers, render_version
  guard.py        startup geometry + binding-secret guard
  api/            ingest, tiles, docs, metadata, search, ops
  render/         geometry (forward+reverse), render (PDF/PIL), browser (Chromium), types
  store/          blobs (content-addressed FS), meta (SQLite), cache (async LRU)
  meta/schema.py  declared schema, meta_expr(), validation
  ssrf.py netguard.py   URL policy + the pinning proxy
docs/             documentation (see the table above)
deploy/           docker-compose.yaml + .env.example
docker/           entrypoint.sh (writability + preflight guard)
tests/            the suite (see below)
```

## Testing

```bash
mamba run -n tiler pip install -e '.[dev]'
mamba run -n tiler python -m pytest -q                    # browser tests skip without Chromium libs
docker compose --profile test run --rm tests              # full suite incl. MHTML rendering
```

The suite covers round-trip identity per doc type, idempotency (asserted against the
render counter), the forward/reverse geometry property over random heights and overlaps,
`store=false`, expression-index usage via `EXPLAIN QUERY PLAN`, metadata CRUD with
immediate index visibility, cursor pagination, the fatal geometry guard, SSRF blocking,
cache coalescing, `/healthz` latency under concurrent ingests, and MHTML re-render
determinism.
