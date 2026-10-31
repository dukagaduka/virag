# Architecture

virag is an **OpenAI-compatible `/v1` proxy** that turns on pixel-RAG *on demand*. A
request that looks like an ordinary chat completion is forwarded untouched to an
upstream vision model (VLM); a request that carries a `rag` field is first augmented
with document-image tiles retrieved from a Qdrant + [doc-tiler](../../doc-tiler)
backend, then forwarded. The client keeps talking to one OpenAI-shaped endpoint and
opts into retrieval per request.

- **Stack:** Python 3.11+, FastAPI, pydantic-settings (configuration), async `httpx2` (one pooled client per upstream, see `clients/http_service.py`), `qdrant-client` (async, over gRPC).
- **Backends (all external):** a retriever server (vLLM, OpenAI-compatible), the
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
4. `proxy` — catch-all reverse proxy: `/tiler/*`, `/retriever/*`, `/qdrant/*`, and
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
2. **Embed.** One text embedding via the retriever client (`embed_texts`).
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
`httpx2` `aiter_raw` over each upstream's shared pooled client, never retried):

| Prefix | Upstream | Auth injected |
|---|---|---|
| `/tiler/*` | doc-tiler | — |
| `/retriever/*` | retriever server | `Authorization: Bearer <RETRIEVER_API_KEY>` |
| `/qdrant/*` | Qdrant | — |
| `/v1/*` (except chat/completions) | VLM | `Authorization: Bearer <VLM_API_KEY>` |

Hop-by-hop headers (RFC 7230) and `Host`/`Content-Length` are stripped in both
directions; the query string is preserved; an unreachable upstream is a `502`. This
lets an operator reach the whole backend through the one gateway (e.g. enroll via the
tiler, inspect the Qdrant collection) without exposing each service separately.

## 6. Clients

Four typed async clients, each built from its settings block and given a
`check_connection()` used at startup:

- **`RetrieverClient`** — the retrieval embedding model behind any OpenAI-compatible
  `/embeddings` API. Text goes through the standard `input` field, `RETRIEVER_BATCH_SIZE`
  per request; images go through chat-style `messages`/`image_url` with a `data:` URI (a
  vLLM extension) — several per request as a batch of conversations when the upstream
  supports it, otherwise one per request (`RETRIEVER_IMAGE_BATCHING`). All batches of a
  call fire concurrently; a batch the upstream rejects on content (`400`/`413`/`422`) is
  retried once as single-item requests, while transient failures retry the batch as-is.
  *Bulk* calls (enroll, self-heal, revision) are capped by `RETRIEVER_MAX_CONCURRENCY`
  and may carry a vLLM `priority`; chat queries bypass both, so an ingest never queues a
  user's query. `check_connection()` also learns the vector size (checked against
  `QDRANT_VECTOR_SIZE`), resolves image batching and verifies the bulk priority.
- **`TilerService`** — doc-tiler client: `tile()`, `tile_stream()` (the NDJSON stream
  `/enroll` consumes), `get_tile()`, `get_info()` (for binding), `get_schema()` (metadata
  schema guard), `patch_metadata()` (the enroll pending markers), `search_docs()` /
  `iter_docs()` (reconcile), `doc_exists()`.
- **`QdrantStore`** — speaks **gRPC** to Qdrant: one multiplexed HTTP/2 channel (no pool
  of idle HTTP/1.1 connections to race the server's keep-alive), retries done by the
  channel's service config (`UNAVAILABLE`, idempotent methods only), an unreachable
  Qdrant surfaced as `503`. Collection lifecycle behind a serving alias over versioned physical
  collections, `upsert` (optional deterministic ids, byte-budget sub-batching), `query`,
  `delete_by_doc` / `iter_doc_ids` (reconcile), the revision blue-green swap
  (`create_revision_collection` / `swap_alias`), and collection-level metadata (used by
  the binding).
- **`VLMClient`** — the upstream generator: `complete()` (buffered passthrough,
  any status) and `stream()` (an async context manager yielding the live SSE
  response). `check_connection()` also probes vision support with a 1×1 PNG.

## 7. Startup & integrity

`lifespan` wires everything and refuses to serve a half-working system:

- Build all four clients from `Settings.from_env()`.
- `create_collection()` on Qdrant; then `check_connection()` on retriever, tiler and
  VLM. **Every dependency is fatal** — the service is useless without any one of them,
  so a failed check aborts startup. The retriever's vector size must also equal
  `QDRANT_VECTOR_SIZE`.
- **Binding** (`enforce_binding`) — see [SECURITY.md](SECURITY.md): virag, the tiler,
  and the Qdrant collection must share one secret hash, or startup fails. A fresh
  collection is bound on first run (secret + the tiler's `render_version`).
- **Metadata-schema guard** (`enforce_metadata_schema`) — reconcile searches the tiler's
  indexed metadata, so the tiler must declare the three fields virag depends on:
  `name` (text), `virag_started_at` (number), `virag_enrolled_at` (number). A missing or
  wrong-typed field is **fatal** (`BindingError`) — the tiler is misconfigured. See
  §9.
- **Serving gate** (`gate.py`) — checked once (it may start a revision right away),
  then re-checked every `VIRAG_TILER_WATCH_INTERVAL_SEC`. See §9.
- **Reconcile** — with the gate open, one `reconcile()` pass runs before serving
  (repairing any partial or orphaned enrolls left by an earlier crash); a background
  task repeats it on `VIRAG_RECONCILE_INTERVAL_SEC`, skipping passes while the gate is
  closed. See §9.

The tiler's `check_connection` hits `/info`, not `/readyz`: a tiler revising its corpus
is up but not ready, and virag waits for it behind the gate instead of refusing to
start. `/ready` reports `503` until all four clients are attached to `app.state` and
while the gate is closed; `/health` is a bare liveness probe (the `HEALTHCHECK`).

## 8. Enrollment & the Qdrant ⊆ Tiler invariant

`/enroll` (`api/enroll.py`) is internally **streaming**. It calls the tiler's
`POST /tile/stream` and consumes the NDJSON event stream, accumulating tiles into
batches of `VIRAG_ENROLL_BATCH_SIZE`, embedding each batch, and upserting its
vectors to Qdrant as it goes — bounded memory, never the whole document at once. The HTTP
contract is unchanged (see [API.md](API.md)); what changed is that it no longer buffers a
whole multi-MB manifest and no longer hits a render timeout, so arbitrarily large
documents enroll fully.

- **Deterministic point ids.** A point's id is `uuid5(namespace, "{doc_id}:{tile_num}")`,
  so re-enrolling a document overwrites its own points in place rather than duplicating.
- **Byte-budget batching.** `QdrantStore.upsert` splits into sub-batches so no single
  request exceeds Qdrant's request limit: `QDRANT_CHUNK_SIZE` (per-request budget in MB)
  and `QDRANT_MAX_BATCH_CHUNKS` (hard cap on points per request).
- **The invariant `Qdrant ⊆ Tiler`.** Every Qdrant point is backed by a tiler tile,
  enforced by write order (the tiler persists a tile, *then* virag writes its point) and
  by reconcile. A query therefore never returns a hit whose tile the tiler can't
  regenerate.
- **Pending markers.** At the start of an enroll virag stamps the tiler's indexed
  metadata with `virag_started_at` (epoch at start) and `virag_enrolled_at` (`0` while
  pending); on success it sets `virag_enrolled_at` to the epoch. On any mid-enroll failure
  the partial points are rolled back out of Qdrant; the tiler doc stays in place and
  pending for reconcile to heal or sweep.

## 9. Reconcile, the collection scheme & revision

**Collection scheme.** virag serves through a stable Qdrant **alias** whose name is the
configured `QDRANT_COLLECTION_NAME`. The alias points to a versioned physical collection
(`<name>__v1`, `__v2`, …); all point operations go through the alias. A plain
collection already sitting under `QDRANT_COLLECTION_NAME` (the pre-alias layout) is
refused at startup — the alias cannot take a name a collection holds.

**Reconcile** (`reconcile.py`) runs once at startup and then on
`VIRAG_RECONCILE_INTERVAL_SEC` (0 disables the loop; the startup pass still runs). Two
passes:

1. *Enforce `Qdrant ⊆ Tiler`.* List every tiler `doc_id` (`POST /docs/search`) and every
   Qdrant `doc_id` (scroll), and delete from Qdrant any `doc_id` the tiler no longer has.
   This is also how a document deleted from the tiler gets its stale points removed
   (delete-propagation) — on the next reconcile pass.
2. *Clear stale pending.* Find docs with `virag_enrolled_at == 0` **and**
   `virag_started_at < now − VIRAG_PENDING_TTL_SEC` (the TTL gate skips in-flight
   enrolls), delete their partial points, and — if `VIRAG_SELF_HEAL` is on and the tiler
   doc is complete and not quarantined — re-embed the doc from the tiler to restore it.
   Otherwise it is left for a re-enroll.

**Re-embed stream.** Self-heal, revision and rollback sync share one re-embed
(`reembed_docs`): every `(doc_id, tile_num)` of the docs is cut into batches of
`VIRAG_ENROLL_BATCH_SIZE` *across document boundaries*, each batch fetches its tiles
concurrently (bulk, capped by `TILER_MAX_CONCURRENCY`), embeds them in one bulk retriever
call and upserts once; a window of batches derived from the tiler cap stays in flight. A
failed batch fails the docs it touched; they are retried once in per-document batches.

**Revision** (`virag/revision.py`, `POST /revision`) is a background full re-embed with a
blue-green swap: refuse while the tiler is not serving, and note its `render_version`;
drop any leftover `__vN+1` and create it; start the store's writer journal; re-embed
every complete, non-quarantined tiler doc into it; abort without a swap (dropping the
build) if docs still fail; enforce `Qdrant ⊆ Tiler` on it; abort likewise if the tiler's
`render_version` changed meanwhile, else stamp it on the build; swap the alias, keeping
the replaced collection (stamped `retired_at`) as the rollback target; stop the journal; wait
up to `VIRAG_PENDING_TTL_SEC` for writers of the old collection to finish; sync the
journaled docs into the new collection (drop, then re-embed if the tiler holds them
complete and enrolled); enforce the invariant again. New writes are never mirrored into
the build — the journal of writers (enroll, self-heal; including those already in flight
at the start) plus the post-swap sync closes the gap. Use it after a retriever-model
change; after a tiler render change the gate starts it. The lock, journal and status are
in-process: correct for a single virag replica.

**Rollback** (`POST /revision/rollback`) repoints the alias to the retained collection,
drops the abandoned one and syncs the docs enrolled since `retired_at`, under the same
journal + drain. It is refused when the retained collection was embedded from another
`render_version` than the tiler's.

**Serving gate** (`gate.py`). The tiler binds `render_version` to its volume like geometry
and never serves a doc rendered by another one; virag binds each collection to the
`render_version` its tiles were embedded from (collection metadata, set at first bind and
by every revision). Serving across a mismatch would fail the integrity check on every
tile, so chat and enroll answer `503` (a router dependency, `require_serving`) unless the
tiler serves (`/info` → `revision.serving`) **and** its `render_version` equals the
collection's. A mismatch starts the revision by itself (`VIRAG_REVISION_ON_RENDER_MISMATCH`),
once per tiler `render_version` — a failed rebuild waits for `POST /revision` rather than
looping. A tiler bound to another secret closes the gate; an unreachable one leaves it as
it was (chat then degrades per tile, as before). The full upgrade path:
the tiler revises its docs (virag waits) → virag rebuilds its collection (still closed) →
both serve. The gate is in-process, like the journal.

## 10. Module layout

```
virag/
  main.py            # app, lifespan (clients, dependency checks, binding, gate, reconcile), router order
  settings.py        # pydantic-settings sections: every env var, its default and bounds
  binding.py         # the TILER_DB_SECRET triple-binding + tiler metadata-schema guard
  gate.py            # ServingGate: serve only while the collection matches the tiler's render_version
  reconcile.py       # Qdrant ⊆ Tiler self-repair (startup + periodic) and the shared re-embed stream
  revision.py        # RevisionManager: background revision / rollback job, journal sync
  errors.py          # ApiError + the uniform {"error": {...}} envelope
  responses.py       # OrjsonResponse, the app-wide default response class
  utils.py           # non-domain helpers (error_body, gather)
  api/
    chat_completions.py  # /v1/chat/completions: rag trigger, retrieval, injection, sources
    enroll.py            # /enroll: streaming tile → embed → upsert
    revision.py          # /revision (start, status) and /revision/rollback
    proxy.py             # reverse proxy for /tiler, /retriever, /qdrant, /v1/*
    ops.py               # /health, /ready
  clients/
    retriever_client.py  qdrant_store.py  tiler_client.py  vlm_client.py
```
