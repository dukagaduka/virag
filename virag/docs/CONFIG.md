# Configuration

All application configuration is read from environment variables at startup by
`Settings.from_env()` (`virag/settings.py`). Three variables are **required** and abort
startup if unset — `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`, `VLM_MODEL_NAME`; everything
else has a default. A ready-to-edit template — including the Compose/deployment
variables — lives at [`deploy/service/.env.example`](../deploy/service/.env.example) and
[`deploy/retriever/.env.example`](../deploy/retriever/.env.example).

Every variable, its default and its bounds are declared once, as pydantic-settings
fields in `virag/settings.py`: each section reads its own prefix and a field's name is
the variable's suffix (`RetrieverSettings.batch_size` ← `RETRIEVER_BATCH_SIZE`). An
**empty** variable counts as unset (so it takes the default, or fails if required).
Invalid values fail fast with a `SettingsError` naming the variable (wrong type, out of
range, unknown enum, a concurrency cap above its pool) rather than surfacing later as a
runtime error.

The Qdrant image is pinned (`QDRANT_DEPLOY_IMAGE=qdrant/qdrant:v1.20.0`) together with
`qdrant-client` in `pyproject.toml`: the client refuses a server more than one minor
version apart, so bump both at once.

---

## Service

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_HOST` | no | `127.0.0.1` | Host in `Settings`. In the container the bind address comes from the `uvicorn` CLI (`--host 0.0.0.0`); see [DEPLOYMENT.md](DEPLOYMENT.md). |
| `VIRAG_PORT` | no | `8080` | Port (1–65535), same note as above. |
| `LOG_LEVEL` | no | `INFO` | Logging level (read in `configure_logging`). |
| `TILER_DB_SECRET` | no | `""` | The single binding secret shared by virag, the tiler and the Qdrant collection. Empty = an explicit, unbound slot. See [SECURITY.md](SECURITY.md). |

## RAG defaults

These are the fallbacks when a request sends `rag: true`; a `rag` **object** overrides
them per request (see [API.md](API.md)).

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_TOP_N` | no | `5` | Number of tiles to retrieve (≥ 1). |
| `VIRAG_SCORE_THRESHOLD` | no | *(none)* | Minimum Qdrant score; empty/unset = no threshold. Must parse as a float. |
| `VIRAG_FRAMING_TEXT` | no | built-in prompt | Text prepended to retrieved tiles. Empty falls back to the built-in default ("The following image tiles are retrieved document context. Use them to answer the question."). |

## Enroll

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_ENROLL_BATCH_SIZE` | no | `32` | Tiles gathered per embed+upsert step in `/enroll`, self-heal and `/revision` (≥ 1). Independent of how the retriever batches its requests. |

## Reconcile (enroll self-repair)

virag keeps Qdrant a consistent subset of the tiler by reconciling at startup and on an
interval — repairing partial/orphaned enrolls and propagating tiler deletes. See
[ARCHITECTURE.md](ARCHITECTURE.md).

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_SELF_HEAL` | no | `true` | On reconcile, re-embed a stale-pending doc from the tiler when the tiler doc is complete (else just drop its partial points and leave it for re-enroll). |
| `VIRAG_RECONCILE_INTERVAL_SEC` | no | `300` | Seconds between periodic reconcile passes (≥ 0). `0` disables the loop — the one-shot startup pass still runs. |
| `VIRAG_PENDING_TTL_SEC` | no | `3600` | A pending enroll older than this is treated as crashed/partial (≥ 1). Must exceed the longest enroll, or an in-flight enroll could be swept. Also the longest a revision/rollback waits after its swap for enrolls still writing into the old collection. |

## Serving gate

Chat and enroll answer `503` while the tiler revises or the collection was embedded from
another `render_version` than the tiler renders. See [ARCHITECTURE.md](ARCHITECTURE.md) §9.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_REVISION_ON_RENDER_MISMATCH` | no | `true` | When the tiler's `render_version` differs from the collection's, start `/revision` by itself (once per tiler `render_version`). `false` → stay not-ready until `POST /revision`. |
| `VIRAG_TILER_WATCH_INTERVAL_SEC` | no | `10` | How often the gate re-checks the tiler (`GET /info`) and the collection (≥ 1). |

## Qdrant

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `QDRANT_BASE_URL` | no | `http://qdrant-service:6333` | Qdrant host and REST port. virag's data path uses gRPC on this host (`QDRANT_GRPC_PORT`); REST serves the `/qdrant` proxy and the client's version check. |
| `QDRANT_GRPC_PORT` | no | `6334` | Qdrant gRPC port. |
| `QDRANT_TIMEOUT_SEC` | no | `60` | Per-call gRPC deadline (≥ 1). |
| `QDRANT_MAX_RETRIES` | no | `3` | Replays on `UNAVAILABLE` (0–4; gRPC caps a call at 5 attempts), with the shared `HTTP_RETRY_BACKOFF_*` backoff. Only idempotent calls: every point op and the collection reads/metadata update — never create/delete/alias changes. A still-unreachable Qdrant surfaces as `503`. |
| `QDRANT_COLLECTION_NAME` | no | `store-collection` | Collection to enroll into / query. |
| `QDRANT_VECTOR_SIZE` | **yes** | — | Embedding dimensionality (≥ 1). Must match the embedding model. |
| `QDRANT_DISTANCE_TYPE` | no | `Cosine` | One of `Cosine`, `Dot`, `Euclid`, `Manhattan`. |
| `QDRANT_CHUNK_SIZE` | no | `24` | Per-upsert message budget, in **megabytes** (≥ 1). `upsert` splits into sub-batches so no single request exceeds this. Over gRPC a vector float costs ~4 B, so `QDRANT_MAX_BATCH_CHUNKS` usually binds first (1024 × 2048-d ≈ 8 MB); Qdrant's 32 MB body limit is a REST limit (a 35 MB gRPC upsert was accepted in testing). |
| `QDRANT_MAX_BATCH_CHUNKS` | no | `1024` | Hard cap on points per upsert request (≥ 1), applied on top of the byte budget. |

The configured `QDRANT_COLLECTION_NAME` is the name of a stable serving **alias**; virag
serves through it while the physical collection it points to is versioned (`__v1`, `__v2`,
…) so a `/revision` can blue-green swap it. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Upstream HTTP retries

Every upstream client (retriever, tiler, VLM) holds one long-lived pooled connection
per upstream and replays transient failures — connection errors and `429`/`502`/`503`/`504`
replies — with full-jitter exponential backoff. Read timeouts are never replayed (a slow
upstream only gets slower under extra load), and neither are tiler ingests or the
passthrough proxies.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `HTTP_RETRY_BACKOFF_BASE_SEC` | no | `0.25` | Backoff base (≥ 0). Each wait is uniform over `[0, min(CAP, BASE · 2^attempt)]`. |
| `HTTP_RETRY_BACKOFF_CAP_SEC` | no | `8` | Backoff ceiling (≥ 0); also caps a numeric `Retry-After` from the upstream. |

## Retriever (OpenAI-compatible embeddings, e.g. vLLM)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `RETRIEVER_BASE_URL` | no | `http://retriever-vllm-virag:8000/v1` | Retriever endpoint. The `/v1` suffix is required. The retriever server runs in a separate stack, so the default container name only resolves if you attach both stacks to one Docker network; the shipped `deploy/service/.env.example` uses `http://host.docker.internal:48003/v1` instead. |
| `RETRIEVER_MODEL_NAME` | no | `retriever-model` | Retriever model name (must match `SERVED_MODEL_NAME` in `serve_retriever.sh`). |
| `RETRIEVER_API_KEY` | no | `dummy` | Bearer key for the retriever server (must match `API_KEY` in `serve_retriever.sh`; also injected on the `/retriever` proxy). |
| `RETRIEVER_BATCH_SIZE` | no | `32` | Items per embedding request (≥ 1): always for texts, for images only when image batching is on. |
| `RETRIEVER_MAX_CONCURRENCY` | no | `32` | Max concurrent **bulk** requests — enroll, self-heal, revision (≥ 0; `0` = unlimited; must be ≤ `RETRIEVER_MAX_CONNECTIONS`). Chat queries bypass it, so keep it below the pool size to leave them headroom. |
| `RETRIEVER_BULK_PRIORITY` | no | `0` | `0` = never send `priority` (portable to any OpenAI-compatible API). `> 0` tags bulk requests with this vLLM priority (lower = served earlier; queries stay at the default `0`), so the server schedules chat queries ahead of queued bulk work. Requires the retriever to run with `--scheduling-policy priority`; verified at startup. |
| `RETRIEVER_IMAGE_BATCHING` | no | `auto` | `auto` — probe at startup whether the upstream accepts several images per request (a vLLM batch of chat conversations) and fall back to one per request if not; `on` — require it (startup fails otherwise); `off` — always one image per request. |
| `RETRIEVER_TIMEOUT_SEC` | no | `60` | Per-read deadline: max silence between bytes (≥ 1). |
| `RETRIEVER_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `RETRIEVER_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the retriever server (≥ 1). |
| `RETRIEVER_MAX_RETRIES` | no | `3` | Retries for a transient failure (≥ 0). |

## Tiler (doc-tiler)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `TILER_BASE_URL` | no | `http://tiler-virag:8080` | doc-tiler endpoint. |
| `TILER_TIMEOUT_SEC` | no | `120` | Per-read deadline (≥ 1): for the tile stream, the max gap between events, not the whole render. |
| `TILER_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `TILER_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the tiler (≥ 1). |
| `TILER_MAX_RETRIES` | no | `3` | Retries for a transient failure on idempotent calls (≥ 0); an ingest is never replayed. |
| `TILER_MAX_CONCURRENCY` | no | `32` | Max concurrent **bulk** tile regenerations — self-heal, revision, rollback sync (≥ 0; `0` = unlimited; must be ≤ `TILER_MAX_CONNECTIONS`). Chat RAG tile fetches bypass it. Size it to the tiler's parallel render capacity; it also sizes how many re-embed batches are kept in flight. |

## Upstream VLM (OpenAI-compatible; external)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VLM_BASE_URL` | **yes** | — | Vision model endpoint. The `/v1` suffix is required. |
| `VLM_MODEL_NAME` | **yes** | — | Model name; used for the startup vision probe. Callers still pass their own `model` in the request body (passed through). |
| `VLM_API_KEY` | no | `dummy` | Bearer key for the VLM (injected on chat completions and the `/v1` proxy). |
| `VLM_TIMEOUT_SEC` | no | `480` | Per-read deadline on the VLM stream (max silence between chunks; covers a long prefill/TTFT on big pixel-RAG prompts), not total generation time (≥ 1). |
| `VLM_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `VLM_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the VLM (≥ 1); must be ≥ `VLM_MAX_CONCURRENCY`. |
| `VLM_MAX_CONCURRENCY` | no | `0` | Max concurrent upstream chat-completion calls; `0` = unlimited (≥ 0). Cap it to shield a fragile/remote VLM from concurrent multimodal load; excess calls queue. Held for the whole stream. |
| `VLM_MAX_RETRIES` | no | `3` | Retries for a transient failure (connection error or `429`/`502`/`503`/`504`) while *opening* an upstream call, before any bytes stream (≥ 0). Never replays a partially-streamed response; once retries run out the last upstream reply is passed through as-is. |

---

## Binding secret note

`TILER_DB_SECRET` is intentionally the **same** variable name used by doc-tiler and
carried on the Qdrant collection — one value, three consumers. It must be set (to the
identical value) on all of them *before* first init, and is immutable per volume/
collection thereafter. See [SECURITY.md](SECURITY.md) for the binding contract and
[DEPLOYMENT.md](DEPLOYMENT.md) for where it is wired in Compose.

For the Compose-only orchestration variables (`*_DEPLOY_*` image tags, host ports,
volume paths), see [`deploy/service/.env.example`](../deploy/service/.env.example) and
[`deploy/retriever/.env.example`](../deploy/retriever/.env.example). The vLLM serving
parameters are set in `deploy/retriever/serve_retriever.sh` (template:
`serve_retriever.example.sh`). See [DEPLOYMENT.md](DEPLOYMENT.md) — none of these are read
by `Settings.from_env()`.
