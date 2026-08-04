# Architecture

virag is an **OpenAI-compatible `/v1` proxy** that turns on pixel-RAG *on demand*. A
request that looks like an ordinary chat completion is forwarded untouched to an
upstream vision model (VLM); a request that carries a `rag` field is first augmented
with document-image tiles retrieved from a Qdrant + [doc-tiler](../../doc-tiler)
backend, then forwarded. The client keeps talking to one OpenAI-shaped endpoint and
opts into retrieval per request.

- **Stack:** Python 3.11+, FastAPI, async `httpx`, `qdrant-client` (async).
- **Backends (all external):** an embeddings server (vLLM, OpenAI-compatible), the
  doc-tiler service, a Qdrant instance, and the upstream VLM.
- **Scope:** the gateway only. Tiling/rendering lives in doc-tiler; embedding lives in
  the vLLM server; vector storage lives in Qdrant; generation lives in the VLM.

---

## 1. Request routing

Routers are registered in a deliberate order (`main.py`), because two of them both
match `/v1/*`:

1. `ops` — `/health`, `/ready`.
2. `enroll` — `POST /enroll`.
3. `chat_completions` — `POST /v1/chat/completions` (the RAG entry point).
4. `proxy` — catch-all reverse proxy: `/tiler/*`, `/embed/*`, `/qdrant/*`, and
   `/v1/*`.

Because `chat_completions` is registered **before** the `/v1/{path:path}` wildcard,
`/v1/chat/completions` is handled by the RAG-aware router and every *other* `/v1/*`
call (models, completions, embeddings, …) falls through to the passthrough proxy. See
[API.md](API.md) for the surface.

## 2. The RAG trigger

RAG is off by default: without a `rag` field, `/v1/chat/completions` is a plain
passthrough — functionally identical to calling the VLM directly. The trigger lives in
the request **body** (not a header or query param) so it rides through any
OpenAI-compatible SDK via `extra_body`, which flattens to top-level JSON.

`rag` is normalized (`_rag_config`) into a config or `None`:

- absent / `false` → RAG off.
- `true` → RAG on with server defaults (`VIRAG_TOP_N`, `VIRAG_SCORE_THRESHOLD`,
  `VIRAG_FRAMING_TEXT`), `include_sources=false`.
- an object → per-request overrides: `top_n`, `score_threshold`, `framing_text`,
  `include_sources`. Each is validated; a bad type is a `422`.

## 3. The retrieval pipeline

When RAG is on, `_run_rag` runs before the upstream call:

1. **Build the query.** Take the text of the *last user message* — concatenating the
   text parts of multimodal content. No user text → `422` (RAG was asked for but there
   is nothing to retrieve on).
2. **Embed.** One text embedding via the embeddings client (`embed_texts`).
3. **Search.** Qdrant top-`N`; if `score_threshold` is set, drop hits below it.
4. **Fetch tiles.** For each hit, call doc-tiler `GET /get_tile?doc_id&tile_num`
   (concurrently) and base64-encode the JPEG. A tile that fails to render is logged
   and **skipped**, not fatal.
5. **Integrity check.** Each hit's Qdrant payload carries the `image_hash` recorded at
   enroll time. The regenerated tile's sha256 must match it; a mismatch is logged and
   the tile **skipped**. This is what guarantees the model sees the exact bytes that
   were embedded, not a drifted re-render.
6. **Inject.** Prepend, into the last user message, a `framing_text` text part followed
   by one `image_url` part per surviving tile (`data:image/jpeg;base64,…`), then the
   caller's original content. Structure preserved for both string and multimodal
   messages.
7. **Passthrough.** The augmented body goes to the VLM unchanged otherwise.

If retrieval yields zero usable tiles, the request still goes upstream — just without
injected context (logged as `rag injected: 0 tiles`).

## 4. Sources

When `rag.include_sources` is true, the response carries a `sources` array — one entry
per injected tile: `{name, doc_id, tile_num, score}` (`name` from the tile's
`metadata.name`).

- **Non-streaming:** the upstream JSON is parsed and `sources` is added as a top-level
  key (only on a `200` with a JSON body; otherwise the upstream response passes
  through verbatim).
- **Streaming (SSE):** upstream chunks are relayed as-is, and a synthetic
  `chat.completion.chunk` carrying `sources` is spliced in **just before** the `data:
  [DONE]` sentinel — mirroring how the `usage` chunk is delivered. If `[DONE]` never
  appears, the sources chunk is appended at the end.

## 5. Passthrough proxy

Everything that isn't RAG is a thin reverse proxy (`_reverse_proxy`, streaming via
`httpx` `aiter_raw`):

| Prefix | Upstream | Auth injected |
|---|---|---|
| `/tiler/*` | doc-tiler | — |
| `/embed/*` | embeddings server | `Authorization: Bearer <VLLM_API_KEY>` |
| `/qdrant/*` | Qdrant | — |
| `/v1/*` (except chat/completions) | VLM | `Authorization: Bearer <VLM_API_KEY>` |

Hop-by-hop headers (RFC 7230) and `Host`/`Content-Length` are stripped in both
directions; the query string is preserved; an unreachable upstream is a `502`. This
lets an operator reach the whole backend through the one gateway (e.g. enroll via the
tiler, inspect the Qdrant collection) without exposing each service separately.

## 6. Clients

Four typed async clients, each built from its settings block and given a
`check_connection()` used at startup:

- **`EmbeddingsClient`** — vLLM OpenAI-compatible embeddings. Text goes through the
  `input` field (batchable); images go through chat-style `messages`/`image_url` with
  a `data:` URI, one request per image. (These are two different vLLM request shapes;
  conflating them is a common bug.)
- **`TilerService`** — doc-tiler client: `tile()`, `get_tile()`, `get_info()` (for
  binding), `doc_exists()`.
- **`QdrantStore`** — collection create, `upsert` (optional deterministic ids),
  `query`, and collection-level metadata (used by the binding).
- **`VLMClient`** — the upstream generator: `complete()` (buffered passthrough,
  any status) and `stream()` (an async context manager yielding the live SSE
  response). `check_connection()` also probes vision support with a 1×1 PNG.

## 7. Startup & integrity

`lifespan` wires everything and refuses to serve a half-working system:

- Build all four clients from `Settings.from_env()`.
- `create_collection()` on Qdrant; then `check_connection()` on embeddings, tiler and
  VLM. **Every dependency is fatal** — the service is useless without any one of them,
  so a failed check aborts startup.
- **Binding** (`enforce_binding`) — see [SECURITY.md](SECURITY.md): virag, the tiler,
  and the Qdrant collection must share one secret hash, or startup fails. A fresh
  collection is bound on first run; tiler `render_version` drift is a warning.
- `warn_doc_id_overlap` — best-effort: sample a few enrolled `doc_id`s and warn if the
  tiler doesn't have them (Qdrant and the tiler drifting out of sync).

`/ready` reports `503` until all four clients are attached to `app.state`; `/health`
is a bare liveness probe.

## 8. Module layout

```
virag/
  main.py            # app, lifespan (clients, dependency checks, binding), router order
  settings.py        # Settings.from_env() — every env var and its validation
  binding.py         # the TILER_DB_SECRET triple-binding + doc_id overlap warning
  errors.py          # ApiError + the uniform {"error": {...}} envelope
  utils.py           # non-domain helpers (error_body)
  api/
    chat_completions.py  # /v1/chat/completions: rag trigger, retrieval, injection, sources
    enroll.py            # /enroll: tile → embed → upsert
    proxy.py             # reverse proxy for /tiler, /embed, /qdrant, /v1/*
    ops.py               # /health, /ready
  clients/
    embeddings_client.py  qdrant_store.py  tiler_client.py  vlm_client.py
```
