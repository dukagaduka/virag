# Configuration

All application configuration is read from environment variables at startup by
`Settings.from_env()` (`virag/settings.py`). Four variables are **required** and abort
startup if unset — `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`, `VLM_MODEL_NAME`, `VALKEY_URL`;
everything else has a default. A ready-to-edit template lives at
[`deploy/service/virag.env.example`](../deploy/service/virag.env.example); the Compose
variables are in [`deploy/service/.env.example`](../deploy/service/.env.example) (see
[DEPLOYMENT.md](DEPLOYMENT.md) for which file holds what).

Every variable, its default and its bounds are declared once, as pydantic-settings
fields in `virag/settings.py`: each section reads its own prefix and a field's name is
the variable's suffix (`RetrieverSettings.batch_size` ← `RETRIEVER_BATCH_SIZE`). An
**empty** variable counts as unset (so it takes the default, or fails if required).
Invalid values fail fast with a `SettingsError` naming the variable (wrong type, out of
range, unknown enum, a concurrency cap above its pool) rather than surfacing later as a
runtime error.

The Qdrant image is pinned (`QDRANT_DEPLOY_IMAGE=qdrant/qdrant:v1.20.0`) together with
`qdrant-client` in `pyproject.toml`: the client refuses a server more than one minor
version apart, so bump both at once. Valkey must be 9.0+ (`SET IFEQ`, `DELIFEQ`); virag
checks the version at startup.

---

## Service

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_API_KEY` | no | `""` | Bearer key for `/v1/*`; clients use it as if it were the LLM's key. Empty = open (a startup `WARNING`). See [SECURITY.md](SECURITY.md). |
| `VIRAG_ADMIN_API_KEY` | no | `""` | Bearer key for `/enroll` and `/revision*` (and `/v1/*` too). Empty = open (a startup `WARNING`). |
| `LOG_LEVEL` | no | `INFO` | Logging level (read in `configure_logging`). |

## RAG defaults

These are the fallbacks when a request sends `rag: true`; a `rag` **object** overrides
them per request (see [API.md](API.md)). How many tiles a query takes and how close
they must be are set with the retriever (`RETRIEVER_TOP_K`, `RETRIEVER_THRESHOLD`) and
the reranker (`RERANKER_TOP_N`, `RERANKER_THRESHOLD`) — see
[Retriever](#retriever-openai-compatible-embeddings-eg-vllm) and
[Reranker](#reranker-optional).

| Variable | Required | Default | Meaning |
|---|---|---|---|
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

## Render version

Chat answers `409 RENDER_VERSION_MISMATCH` for a tile the tiler renders by another
`render_version` than it was embedded from. See [ARCHITECTURE.md](ARCHITECTURE.md) §9.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_REVISION_ON_RENDER_MISMATCH` | no | `false` | At startup, when the collection was embedded from another `render_version` than the tiler's, start `/revision` right away. `false` → log a warning and wait for `POST /revision`. |

## Coordination (Valkey)

Any number of virag processes — replicas, uvicorn workers — share one Valkey for their
leases, the writers registry, the revision job and the reconcile lead. See
[ARCHITECTURE.md](ARCHITECTURE.md) §10.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VALKEY_URL` | **yes** | — | `redis://[:password@]host:port/db`. The password is Valkey's `requirepass` and a secret; a wrong one fails startup (`Valkey rejects the password in VALKEY_URL`). |
| `VALKEY_PREFIX` | no | `virag` | Prefix of every key. Deployments sharing one Valkey need distinct prefixes (or databases). |
| `VALKEY_TIMEOUT_SEC` | no | `2` | Connect and per-command deadline (> 0); past it the call is a `503`. |
| `VALKEY_MAX_CONNECTIONS` | no | `64` | Connection pool size (≥ 1). |
| `VIRAG_PROCESS_TTL_SEC` | no | `30` | Lease TTL (≥ 3), renewed every third of it. A process silent this long counts as dead: its revision job, reconcile lead and writers are taken over or ignored. Lower = faster takeover, less tolerance for a stalled event loop or a Valkey blip. |

The bulk concurrency caps (`RETRIEVER_MAX_CONCURRENCY`, `TILER_MAX_CONCURRENCY`),
`VLM_MAX_CONCURRENCY` and `RERANKER_MAX_CONCURRENCY` are per process: N processes allow
N times as much.

## Qdrant

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `QDRANT_BASE_URL` | no | `http://qdrant-service:6333` | Qdrant host and REST port. virag's data path uses gRPC on this host (`QDRANT_GRPC_PORT`); REST serves the client's version check. |
| `QDRANT_GRPC_PORT` | no | `6334` | Qdrant gRPC port. |
| `QDRANT_API_KEY` | no | `""` | Qdrant's `service.api_key`; empty = no key sent. A wrong one fails startup with `grpc_status: UNAUTHENTICATED`. |
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
upstream only gets slower under extra load), and neither are tiler ingests.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `HTTP_RETRY_BACKOFF_BASE_SEC` | no | `0.25` | Backoff base (≥ 0). Each wait is uniform over `[0, min(CAP, BASE · 2^attempt)]`. |
| `HTTP_RETRY_BACKOFF_CAP_SEC` | no | `8` | Backoff ceiling (≥ 0); also caps a numeric `Retry-After` from the upstream. |

## Retriever (OpenAI-compatible embeddings, e.g. vLLM)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `RETRIEVER_BASE_URL` | no | `http://retriever-vllm-virag:8000/v1` | Retriever endpoint. The `/v1` suffix is required. The retriever server runs in a separate stack, so the default container name only resolves if you attach both stacks to one Docker network; the shipped `deploy/service/virag.env.example` uses `http://host.docker.internal:48004/v1` instead. |
| `RETRIEVER_MODEL_NAME` | no | `retriever-model` | Retriever model name (must match `SERVED_MODEL_NAME` in `serve_retriever.sh`). |
| `RETRIEVER_API_KEY` | no | `dummy` | Bearer key for the retriever server (must match `RETRIEVER_DEPLOY_API_KEY` in `deploy/retriever/.env`). |
| `RETRIEVER_BATCH_SIZE` | no | `32` | Items per embedding request (≥ 1): always for texts, for images only when image batching is on. |
| `RETRIEVER_MAX_CONCURRENCY` | no | `32` | Max concurrent **bulk** requests — enroll, self-heal, revision (≥ 0; `0` = unlimited; must be ≤ `RETRIEVER_MAX_CONNECTIONS`). Chat queries bypass it, so keep it below the pool size to leave them headroom. |
| `RETRIEVER_BULK_PRIORITY` | no | `0` | `0` = never send `priority` (portable to any OpenAI-compatible API). `> 0` tags bulk requests with this vLLM priority (lower = served earlier; queries stay at the default `0`), so the server schedules chat queries ahead of queued bulk work. Requires the retriever to run with `--scheduling-policy priority`; verified at startup. |
| `RETRIEVER_IMAGE_BATCHING` | no | `auto` | `auto` — probe at startup whether the upstream accepts several images per request (a vLLM batch of chat conversations) and fall back to one per request if not; `on` — require it (startup fails otherwise); `off` — always one image per request. |
| `RETRIEVER_TIMEOUT_SEC` | no | `60` | Per-read deadline: max silence between bytes (≥ 1). |
| `RETRIEVER_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `RETRIEVER_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the retriever server (≥ 1). |
| `RETRIEVER_MAX_RETRIES` | no | `3` | Retries for a transient failure (≥ 0). |
| `RETRIEVER_TOP_K` | no | `5` | Candidate tiles a RAG query takes from Qdrant (≥ 1). Without a reranker all of them reach the VLM; with one, it picks `RERANKER_TOP_N` of them. Request override: `rag.top_k`. |
| `RETRIEVER_THRESHOLD` | no | *(none)* | Minimum Qdrant score of a candidate; empty/unset = no threshold. Must parse as a float. Passed to Qdrant as `score_threshold`, so it follows `QDRANT_DISTANCE_TYPE`: a floor for `Cosine`/`Dot`, a ceiling for `Euclid`/`Manhattan`. Request override: `rag.retriever_threshold`. |

## Reranker (optional)

Picks the tiles for the VLM out of the retriever's candidates. It scores each
(query, tile) pair on its own, so the `RETRIEVER_TOP_K` candidates go
`RERANKER_BATCH_SIZE` per request, all at once, and the cut to `RERANKER_TOP_N` is made
over all their scores. With `RERANKER_BASE_URL` unset it is off, and startup logs a
warning (naming any `RERANKER_*` variable set anyway, as ignored). It never blocks
startup or a chat: a reranker failing its startup check is logged; one failing a request
is logged — a warning for an outage, an error for a refusal or an unreadable reply (a
contract mismatch to fix) — and that request goes on with the retriever's order, cut to
`RERANKER_TOP_N`.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `RERANKER_BASE_URL` | no | *(none)* | Reranker server root, **without** `/v1` (e.g. `http://host.docker.internal:48005`); unset = no reranker. |
| `RERANKER_MODEL_NAME` | no | `reranker-model` | Must match `SERVED_MODEL_NAME` in `serve_reranker.sh`; checked at startup via `/v1/models`. |
| `RERANKER_API_KEY` | no | `dummy` | Bearer key (must match `RERANKER_DEPLOY_API_KEY` in `deploy/reranker/.env`). |
| `RERANKER_API` | no | `score` | Wire format. `score` — vLLM's `/score`: scores only, in input order. `rerank` — the Cohere/Jina-style `/v1/rerank` (`query`, `documents`, `return_documents: false`) that SGLang and others implement too; vLLM serves it but echoes every image back, so use `score` against vLLM. |
| `RERANKER_BATCH_SIZE` | no | `8` | Tiles per request (≥ 1). Not for speed — a vLLM reranker queues the pairs itself — but to keep each request short of `RERANKER_TIMEOUT_SEC`, to replay only one batch on a failure, and to let concurrent chats interleave on the server. |
| `RERANKER_TOP_N` | no | `5` | Tiles the VLM gets, best first (≥ 1; ≤ `RETRIEVER_TOP_K` while the reranker is on, else startup fails). Request override: `rag.top_n`. |
| `RERANKER_THRESHOLD` | no | *(none)* | Minimum relevance of a tile, `0`–`1`; empty/unset = no threshold. Request override: `rag.reranker_threshold`. |
| `RERANKER_TIMEOUT_SEC` | no | `30` | Per-read deadline: max silence between bytes (≥ 1). |
| `RERANKER_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `RERANKER_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the reranker (≥ 1). |
| `RERANKER_MAX_CONCURRENCY` | no | `0` | Max concurrent reranker requests from this process; `0` = unlimited (≥ 0; must be ≤ `RERANKER_MAX_CONNECTIONS`). |
| `RERANKER_MAX_RETRIES` | no | `3` | Retries for a transient failure (≥ 0). |

## Tiler (doc-tiler)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `TILER_BASE_URL` | no | `http://tiler-virag:8080` | doc-tiler endpoint. |
| `TILER_API_KEY` | no | `""` | The tiler's own `TILER_API_KEY`, sent as a bearer token; empty = none. A wrong one fails startup (`the tiler rejects TILER_API_KEY`). |
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
| `VLM_API_KEY` | no | `dummy` | Bearer key for the VLM (sent with chat completions and the `/v1/models` lookup). |
| `VLM_TIMEOUT_SEC` | no | `480` | Per-read deadline on the VLM stream (max silence between chunks; covers a long prefill/TTFT on big pixel-RAG prompts), not total generation time (≥ 1). |
| `VLM_CONNECT_TIMEOUT_SEC` | no | `10` | Connect deadline (≥ 1). |
| `VLM_MAX_CONNECTIONS` | no | `100` | Pooled keep-alive connections to the VLM (≥ 1); must be ≥ `VLM_MAX_CONCURRENCY`. |
| `VLM_MAX_CONCURRENCY` | no | `0` | Max concurrent upstream chat-completion calls; `0` = unlimited (≥ 0). Cap it to shield a fragile/remote VLM from concurrent multimodal load; excess calls queue. Held for the whole stream. |
| `VLM_MAX_RETRIES` | no | `3` | Retries for a transient failure (connection error or `429`/`502`/`503`/`504`) while *opening* an upstream call, before any bytes stream (≥ 0). Never replays a partially-streamed response; once retries run out the last upstream reply is passed through as-is. |

---

## Deploy variables

For the Compose-only orchestration variables (`*_DEPLOY_*` image tags, host ports,
volume paths), see [`deploy/service/.env.example`](../deploy/service/.env.example) and
[`deploy/retriever/.env.example`](../deploy/retriever/.env.example). The vLLM serving
parameters are set in `deploy/retriever/serve_retriever.sh` (template:
`serve_retriever.example.sh`). See [DEPLOYMENT.md](DEPLOYMENT.md) — none of these are read
by `Settings.from_env()`.
