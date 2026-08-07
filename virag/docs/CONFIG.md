# Configuration

All application configuration is read from environment variables at startup by
`Settings.from_env()` (`virag/settings.py`). Two variables are **required** and abort
startup if unset; everything else has a default. A ready-to-edit template — including
the Compose/deployment variables — lives at
[`deploy/.env.example`](../deploy/.env.example).

Invalid values fail fast with a `SettingsError` (wrong type, out of range, unknown
enum) rather than surfacing later as a runtime error.

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

## Reconcile (enroll self-repair)

virag keeps Qdrant a consistent subset of the tiler by reconciling at startup and on an
interval — repairing partial/orphaned enrolls and propagating tiler deletes. See
[ARCHITECTURE.md](ARCHITECTURE.md).

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VIRAG_SELF_HEAL` | no | `true` | On reconcile, re-embed a stale-pending doc from the tiler when the tiler doc is complete (else just drop its partial points and leave it for re-enroll). |
| `VIRAG_RECONCILE_INTERVAL_SEC` | no | `300` | Seconds between periodic reconcile passes (≥ 0). `0` disables the loop — the one-shot startup pass still runs. |
| `VIRAG_PENDING_TTL_SEC` | no | `3600` | A pending enroll older than this is treated as crashed/partial (≥ 1). Must exceed the longest enroll, or an in-flight enroll could be swept. |

## Qdrant

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `QDRANT_BASE_URL` | no | `http://qdrant-service:6333` | Qdrant endpoint. |
| `QDRANT_COLLECTION_NAME` | no | `store-collection` | Collection to enroll into / query. |
| `QDRANT_VECTOR_SIZE` | **yes** | — | Embedding dimensionality (≥ 1). Must match the embedding model. |
| `QDRANT_DISTANCE_TYPE` | no | `Cosine` | One of `Cosine`, `Dot`, `Euclid`, `Manhattan`. |
| `QDRANT_CHUNK_SIZE` | no | `24` | Per-upsert request body budget, in **megabytes** (≥ 1). `upsert` splits into sub-batches so no single request exceeds this; keep it under Qdrant's ~32 MB body limit. |
| `QDRANT_MAX_BATCH_CHUNKS` | no | `1024` | Hard cap on points per upsert request (≥ 1), applied on top of the byte budget. |

The configured `QDRANT_COLLECTION_NAME` is the name of a stable serving **alias**; virag
serves through it while the physical collection it points to is versioned (`__v1`, `__v2`,
…) so a `/revision` can blue-green swap it. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Embeddings (vLLM, OpenAI-compatible)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `EMBEDDINGS_BASE_URL` | no | `http://embeddings-vllm-virag:8000/v1` | Embeddings endpoint. The `/v1` suffix is required. |
| `VLLM_SERVED_MODEL_NAME` | no | `embeddings-model` | Embeddings model name (matches what the vLLM server advertises). |
| `VLLM_API_KEY` | no | `dummy` | Bearer key for the embeddings server (also injected on the `/embed` proxy). |
| `EMBEDDINGS_CHUNK_SIZE` | no | `16` | Max items per embedding request (≥ 1). |
| `EMBEDDINGS_TIMEOUT_SEC` | no | `60` | Per-request timeout (≥ 1). |

## Tiler (doc-tiler)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `TILER_BASE_URL` | no | `http://tiler-virag:8080` | doc-tiler endpoint. |
| `TILER_TIMEOUT_SEC` | no | `120` | Per-request timeout (≥ 1); tiling and rendering can be slow. |

## Upstream VLM (OpenAI-compatible; external)

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `VLM_BASE_URL` | **yes** | — | Vision model endpoint. The `/v1` suffix is required. |
| `VLM_MODEL_NAME` | **yes** | — | Model name; used for the startup vision probe. Callers still pass their own `model` in the request body (passed through). |
| `VLM_API_KEY` | no | `dummy` | Bearer key for the VLM (injected on chat completions and the `/v1` proxy). |
| `VLM_TIMEOUT_SEC` | no | `480` | Per-read deadline on the VLM stream (max silence between chunks; covers a long prefill/TTFT on big pixel-RAG prompts), not total generation time (≥ 480). |
| `VLM_MAX_CONCURRENCY` | no | `0` | Max concurrent upstream chat-completion calls; `0` = unlimited (≥ 0). Cap it to shield a fragile/remote VLM from concurrent multimodal load; excess calls queue. Held for the whole stream. |
| `VLM_MAX_RETRIES` | no | `3` | Retries for a transient connection error while *opening* an upstream call, before any bytes stream (≥ 0). Never replays a partially-streamed response. |

---

## Binding secret note

`TILER_DB_SECRET` is intentionally the **same** variable name used by doc-tiler and
carried on the Qdrant collection — one value, three consumers. It must be set (to the
identical value) on all of them *before* first init, and is immutable per volume/
collection thereafter. See [SECURITY.md](SECURITY.md) for the binding contract and
[DEPLOYMENT.md](DEPLOYMENT.md) for where it is wired in Compose.

For the Compose-only orchestration variables (`*_DEPLOY_*` image tags, host ports,
volume paths, the vLLM serving knobs), see [`deploy/.env.example`](../deploy/.env.example)
and [DEPLOYMENT.md](DEPLOYMENT.md) — those are read by Compose and the vLLM launch
script, not by `Settings.from_env()`.
