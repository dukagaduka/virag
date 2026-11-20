# Architecture

virag is an **OpenAI-compatible `/v1` proxy** that turns on pixel-RAG *on demand*. A
request that looks like an ordinary chat completion is forwarded untouched to an
upstream vision model (VLM); a request that carries a `rag` field is first augmented
with document-image tiles retrieved from a Qdrant + [doc-tiler](../../doc-tiler)
backend, then forwarded. The client keeps talking to one OpenAI-shaped endpoint and
opts into retrieval per request.

- **Stack:** Python 3.11+, FastAPI, pydantic-settings (configuration), async `httpx2` (one pooled client per upstream, see `clients/http_service.py`), `qdrant-client` (async, over gRPC).
- **Backends (all external):** a retriever server (vLLM, OpenAI-compatible), the
  doc-tiler service, a Qdrant instance, the upstream VLM, and a Valkey the virag
  processes coordinate through; optionally a reranker server (vLLM).
- **Scope:** the gateway only. Tiling/rendering lives in doc-tiler; embedding lives in
  the vLLM server; reranking in its own server; vector storage lives in Qdrant;
  generation lives in the VLM. virag
  itself keeps no state: any number of processes (replicas, uvicorn workers) serve
  side by side (§10).

---

## 1. Request routing

Routers are mounted in `include_routers` (`main.py`), each with the key it needs (§5):

1. `ops` — `/health`, `/ready` (open).
2. `enroll` — `POST /enroll` (admin key).
3. `revision` — `/revision`, `/revision/rollback` (admin key).
4. `chat_completions` — `POST /v1/chat/completions`, the RAG entry point (user key).
5. `models` — `GET /v1/models`: the upstream VLM's entry for `VLM_MODEL_NAME` (user key).

There is no passthrough: any other path is a `404`. See [API.md](API.md) for the surface.

## 2. The RAG trigger

RAG is off by default: without a `rag` field, `/v1/chat/completions` is a plain
passthrough — functionally identical to calling the VLM directly. The trigger lives in
the request **body** (not a header or query param) so it rides through any
OpenAI-compatible SDK via `extra_body`, which flattens to top-level JSON.

`rag` is normalized (`_rag_config`) into a config or `None`:

- absent / `false` → RAG off.
- `true` → RAG on with server defaults (`RETRIEVER_TOP_K`, `RETRIEVER_THRESHOLD`,
  `RERANKER_TOP_N`, `RERANKER_THRESHOLD`, `VIRAG_FRAMING_TEXT`), `include_sources=false`.
- an object → per-request overrides: `top_k`, `retriever_threshold`, `top_n`,
  `reranker_threshold`, `framing_text`, `include_sources`. Each is validated; a bad
  type, or a `top_n` above `top_k` while a reranker is on, is a `422`.

## 3. The retrieval pipeline

When RAG is on, `_run_rag` runs before the upstream call:

1. **Build the query.** Take the text of the *last user message* — concatenating the
   text parts of multimodal content. No user text → `422` (RAG was asked for but there
   is nothing to retrieve on).
2. **Embed.** One text embedding via the retriever client (`embed_texts`).
3. **Search.** Qdrant top-`K` (`top_k`); a `retriever_threshold` goes into the same
   query as Qdrant's `score_threshold`, so Qdrant applies it by the collection's
   distance (a floor for `Cosine`/`Dot`, a ceiling for `Euclid`/`Manhattan`).
4. **Fetch tiles.** For each hit, call doc-tiler `GET /get_tile?doc_id&tile_num`
   (concurrently) and base64-encode the JPEG. A tile the tiler refuses (`404`, `410`,
   its own hash `409`, …) is logged and **skipped**, not fatal. A tiler that is
   unreachable or not serving (`503`) fails the request with `503 NOT_READY` — virag
   does not answer without its context.
5. **Integrity check.** Each hit's Qdrant payload carries the `render_version` and the
   `image_hash` recorded at enroll time. A tile whose `X-Render-Version` differs from
   the payload's fails the whole request with `409 RENDER_VERSION_MISMATCH` (the first
   one cancels the other fetches): the collection is stale, and answering without its
   tiles would hide that. Otherwise the regenerated tile's sha256 must match the
   `image_hash`; a mismatch — or a point without `image_hash` — is logged and the tile
   **skipped**. This is what guarantees
   the model sees the exact bytes that were embedded, not a drifted re-render.
6. **Rerank** (only with `RERANKER_BASE_URL` set). The surviving tiles — the very bytes
   just fetched, no second render — are scored against the query text by the reranker
   (`RerankerClient.score`). It scores each (query, tile) pair on its own, so a score
   does not depend on the other tiles of its request: the tiles go
   `RERANKER_BATCH_SIZE` per request, all at once, and the scores of every batch are
   merged before the cut. Tiles below `reranker_threshold` are dropped, the rest sorted
   best first and cut to `top_n`. A failing reranker (down, overloaded, or refusing the
   request) does not fail the chat: it is logged — a refusal or an unreadable reply as
   an error, being a contract mismatch, an outage as a warning — and the tiles keep the
   retriever's order, cut to `top_n`. Without a reranker all `top_k` tiles go on.
7. **Inject.** Prepend, into the last user message, a `framing_text` text part followed
   by one `image_url` part per surviving tile (`data:image/jpeg;base64,…`), then the
   caller's original content. Structure preserved for both string and multimodal
   messages.
8. **Passthrough.** The augmented body goes to the VLM unchanged otherwise.

If retrieval yields zero usable tiles, the request still goes upstream — just without
injected context (logged as `rag injected: 0 tiles`).

## 4. Sources

When `rag.include_sources` is true, the response carries a `sources` array — one entry
per injected tile, in prompt order: `{name, doc_id, tile_num, score}` (`name` is read
from the tiler — the document's `metadata.name`, one request per distinct document, its
`doc_id` when it has none; `score` the Qdrant one), plus `rerank_score` for a tile the
reranker scored.

- **Non-streaming:** the upstream JSON is parsed and `sources` is added as a top-level
  key (only on a `200` with a JSON body; otherwise the upstream response passes
  through verbatim).
- **Streaming (SSE):** upstream chunks are relayed as-is, and a synthetic
  `chat.completion.chunk` carrying `sources` is spliced in **just before** the `data:
  [DONE]` sentinel — mirroring how the `usage` chunk is delivered. If `[DONE]` never
  appears, the sources chunk is appended at the end.

## 5. Authentication

virag's own clients present a bearer key (`api/auth.py`, a router dependency):
`VIRAG_API_KEY` for `/v1/*`, `VIRAG_ADMIN_API_KEY` for `/enroll` and `/revision*` (and
`/v1/*` too). A missing or wrong key is a `401`, the user key on an admin route a `403`;
an empty key leaves its routes open (a startup `WARNING`). virag in turn holds every
backend's key — `VLM_API_KEY`, `RETRIEVER_API_KEY`, `RERANKER_API_KEY`, `TILER_API_KEY`,
`QDRANT_API_KEY`, the password in `VALKEY_URL` — and never hands one out. See
[SECURITY.md](SECURITY.md).

## 6. Clients

Five typed async clients, each built from its settings block and given a
`check_connection()` used at startup. Each is an API to its backend and nothing more;
the rules that span virag processes live in the Coordinator (§10).

The HTTP ones share `HttpService`: one pooled client per upstream, retries with jittered
backoff, one mapping of transport and status errors to `ApiError`, and TCP keepalive on
every connection — an upstream that vanished without closing its connections is found
dead by the kernel in about 25 s, so a request waiting on it (an enroll reading the
tiler's stream, a chat waiting for the VLM) fails then and not at its read timeout,
which has to cover the upstream's longest silence.

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
- **`TilerService`** — doc-tiler client: `tile_stream()` (the NDJSON ingest stream
  `/enroll` consumes), `doc_stream()` (the same stream of a stored doc: self-heal, a
  revision's sync), `get_tile()`, `get_info()` (for binding), `get_schema()` (metadata
  schema guard), `get_metadata()` / `patch_metadata()` (document names; the enroll's
  pending marker), `search_docs()` / `iter_docs()` (reconcile), `doc_exists()`.
- **`QdrantStore`** — speaks **gRPC** to Qdrant: one multiplexed HTTP/2 channel (no pool
  of idle HTTP/1.1 connections to race the server's keep-alive), retries done by the
  channel's service config (`UNAVAILABLE`, idempotent methods only), an unreachable
  Qdrant surfaced as `503`. Collection lifecycle behind a serving alias over versioned physical
  collections, `upsert` (optional deterministic ids, byte-budget sub-batching), `query`,
  `delete_by_doc` / `iter_doc_ids` (reconcile), the revision blue-green swap
  (`create_revision_collection` / `swap_alias` / `rollback_alias`), and collection-level
  metadata (used by the binding). The alias is never cached: `resolve()` reads the
  serving collection and the rollback target from Qdrant, since another process may
  swap it at any time.
- **`ValkeyStore`** — plain and JSON values, conditional writes (`SET NX`, `SET XX` with a
  TTL, `SET IFEQ`, `DELIFEQ` — Valkey 9.0+, checked at startup), sets, names listed by
  pattern, every key under `VALKEY_PREFIX`; an unreachable Valkey surfaced as `503`. It knows nothing of
  what the keys mean.
- **`VLMClient`** — the upstream generator: `complete()` (buffered passthrough,
  any status) and `stream()` (an async context manager yielding the live SSE
  response). `check_connection()` also probes vision support with a 1×1 PNG.

## 7. Startup & integrity

`lifespan` wires everything and refuses to serve a half-working system:

- Build all five clients from `Settings.from_env()`.
- `create_collection()` on Qdrant; then `check_connection()` on retriever, tiler (which
  also proves `TILER_API_KEY`), VLM and Valkey. **Every dependency is fatal** — the service is useless without any one of
  them, so a failed check aborts startup. The retriever's vector size must also equal
  `QDRANT_VECTOR_SIZE`. Two processes starting on one empty Qdrant both try to create
  `<name>__v1`; the one that loses adopts the other's.
- **Binding** (`enforce_binding`) — see [SECURITY.md](SECURITY.md): the Qdrant
  collection must have been embedded from the tiler's volume (its `volume_id`), or
  startup fails. A fresh collection is bound on first run (`tiler_volume_id` + the
  tiler's `render_version`).
- **Metadata-schema guard** (`enforce_metadata_schema`) — reconcile searches the tiler's
  indexed metadata, so the tiler must declare the two fields virag depends on:
  `name` (text), `virag_enrolled_at` (number). A missing or
  wrong-typed field is **fatal** (`BindingError`) — the tiler is misconfigured. See
  §9.
- **Lease** — the process takes its lease in Valkey and starts the heartbeat that
  renews it (§10).
- **Upkeep** — before serving, a process finishes a revision / rollback job left after
  its alias move (`RevisionManager.resume_orphaned`, in the background) and runs one
  `reconcile()` pass (repairing any partial or orphaned enrolls left by an earlier
  crash; in the process leading reconcile); a background task in every process repeats
  both on `VIRAG_RECONCILE_INTERVAL_SEC`. See §9.
- **render_version check** (`RevisionManager.revise_if_stale`) — a collection embedded
  from another `render_version` than the tiler's is logged; with
  `VIRAG_REVISION_ON_RENDER_MISMATCH=true` the revision starts right away, unless
  another process already runs it. See §9.

The tiler is as essential as the VLM: its `check_connection` hits `/readyz`, so a tiler
that is down — or still revising its own corpus — refuses virag's startup (the
container restarts until the tiler is ready). `/ready` reports `503` until every
client and the Coordinator are attached to `app.state`; `/health` is a bare liveness
probe (the `HEALTHCHECK`).

Shutdown runs the same stack in reverse: the periodic reconcile and this process's
revision job are cancelled (the job publishes `failed: cancelled (shutdown)` and
releases the lock), the heartbeat stops, the reconcile lead and the lease are handed
back, then the clients close. uvicorn's `--timeout-graceful-shutdown` keeps a long
open stream from holding this past compose's `stop_grace_period`.

## 8. Enrollment & the Qdrant ⊆ Tiler invariant

`/enroll` (`api/enroll.py`) is internally **streaming**. It opens the tiler's
`POST /tile/stream` and hands the NDJSON event stream to the **`Enroller`**
(`enroll.py`) — the procedure that writes a document's points — which accumulates tiles into
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
  enforced by write order (the tiler inserts the doc's row before its first tile leaves,
  *then* virag writes points; the tiles are served once the ingest commits) and by
  reconcile. A query therefore never returns a hit whose tile the tiler can't
  regenerate.
- **One enroll of a document at a time.** At `begin` — the first moment the `doc_id` is
  known — the enroll takes the document: the key `enroll:<doc_id>` in Valkey, holding
  its process's token (`Coordinator.enrolling`, §10), released when the enroll ends. A
  second enroll of the same document answers `409 ENROLL_IN_PROGRESS` and touches
  nothing; reconcile and a revision's sync leave a held document to its enroll. So the
  points of a document are written, and rolled back, by one request only.
- **The pending marker.** Having taken the document, the enroll stamps the tiler's indexed
  metadata with `virag_enrolled_at = 0` (pending) and drops the document's earlier
  points; on success it sets `virag_enrolled_at` to the epoch. On any failure after the
  pending mark the partial points are rolled back out of Qdrant; a doc the tiler held
  before stays in place and pending — and, no enroll holding it any more, is enrolled
  again by the next reconcile pass; a new one the tiler drops with its ingest. An enroll
  that could not even mark the document pending leaves its points as they were. Before
  the rollback delete the enroll checks the document is still its own (fencing): a
  process that stalled past its lease may find another enroll writing it already.
- **One collection per write.** Taking the document also pins the enroll to the physical
  collection the alias resolves to then; every upsert and the rollback delete go there.
  A revision swapping the alias meanwhile never splits a document across two
  collections: the document is in its journal, and the revision syncs it into the new
  collection once the enroll is done. With Valkey unreachable the document cannot be
  taken, and the enroll answers `503 NOT_READY` before touching anything.

## 9. Reconcile, the collection scheme & revision

**Collection scheme.** virag serves through a stable Qdrant **alias** whose name is the
configured `QDRANT_COLLECTION_NAME`. The alias points to a versioned physical collection
(`<name>__v1`, `__v2`, …); all point operations go through the alias. A plain
collection already sitting under `QDRANT_COLLECTION_NAME` (the pre-alias layout) is
refused at startup — the alias cannot take a name a collection holds.

**Reconcile** (`reconcile.py`) runs once at startup and then on
`VIRAG_RECONCILE_INTERVAL_SEC` (0 disables the loop; the startup pass still runs), in
the one process leading it (§10). Two passes:

1. *Enforce `Qdrant ⊆ Tiler`.* List every tiler `doc_id` (`POST /docs/search`) and every
   Qdrant `doc_id` (scroll), and delete from Qdrant any `doc_id` the tiler no longer has.
   This is also how a document deleted from the tiler gets its stale points removed
   (delete-propagation) — on the next reconcile pass. Points of a doc the tiler
   quarantined go too: its tiles answer `410`, so they would only crowd live hits out
   of `top_k`; a re-enroll brings the points back. A doc being enrolled anywhere in the
   cluster is left to its enroll: the keys are read last, right before the delete.
2. *Finish abandoned enrolls.* A doc that is pending (`virag_enrolled_at == 0`) while
   nobody is enrolling it (no `enroll:<doc_id>` of a live process) was left by an
   enroll that failed or died — no age to wait out. If `VIRAG_SELF_HEAL` is on and the
   tiler doc is complete and not quarantined, it is **enrolled again** from the tiler's
   stored tiles: the same `Enroller`, fed the tiler's `GET /doc/{id}/tiles` stream
   instead of an ingest (`enroll_stored`). Otherwise its partial points are dropped
   (`Enroller.drop`) and it is left for a re-enroll. Docs are taken one at a time, each
   under its key; one that fails stays pending for the next pass.

**One writer.** A document's points in the serving collection are written and dropped
only through the `Enroller`, under the document's key: by `/enroll`, by reconcile's
second pass, and by a revision's sync after its swap. The exceptions hold no key and
only look at the keys: reconcile's first pass, which deletes points of documents the
tiler no longer serves, and a revision's build, which writes a collection nobody reads
yet.

**The flat re-embed** (`RevisionManager.reembed`) is how a revision builds its
collection: every `(doc_id, tile_num)` of the docs is cut into batches of
`VIRAG_ENROLL_BATCH_SIZE` *across document boundaries*, each batch fetches its tiles
concurrently (bulk, capped by `TILER_MAX_CONCURRENCY`), embeds them in one bulk retriever
call and upserts once; a window of batches derived from the tiler cap stays in flight. A
failed batch fails the docs it touched; they are retried once in per-document batches.

**Revision** (`virag/revision.py`, `POST /revision`) is a background full re-embed with a
blue-green swap: take the job (`revision:on`, one across the cluster, §10); refuse while
the tiler is not serving, and note its `render_version`; resolve the serving collection;
drop any leftover `__vN+1` and create it; start the journal; re-embed every
complete, non-quarantined tiler doc into it; abort without a swap (dropping the build)
if docs still fail; enforce `Qdrant ⊆ Tiler` on it; abort likewise if the tiler's
`render_version` changed meanwhile, else stamp it on the build; check the job is still
this process's (fencing: a process whose lease expired may have been replaced); swap the
alias, keeping the replaced collection (stamped `retired_at`) as the rollback target;
read the journal; wait up to `VIRAG_REVISION_DRAIN_TIMEOUT_SEC` for the enrolls in
flight at that moment (one of them may still be writing into the old collection); sync
the journaled docs into the new collection — each through the `Enroller`: enrolled
again from its stored tiles if the tiler holds it complete and enrolled, its points
dropped otherwise; a doc being enrolled is left to its enroll, which writes it there
whole; enforce the invariant again. New writes are never mirrored into the build — the journal
of documents enrolled meanwhile (by `/enroll` or self-heal, in any process; including
those already in flight at the start) plus the post-swap sync closes the gap. Use it after a retriever-model change and after a tiler render change. The
request may land on any process: that process runs the job, and every process answers
`GET /revision`.

**Rollback** (`POST /revision/rollback`) repoints the alias to the retained collection,
waits for the enrolls in flight (they may be writing into the abandoned one), drops it
and syncs the docs enrolled since `retired_at`, under the same job lock and journal. It is refused when
the retained collection was embedded from another `render_version` than the tiler's.

**A job past its alias move is always finished.** Before the swap a job that stops is
failed and costs nothing: the old collection keeps serving, the next request rebuilds.
After it the alias points at a collection that lacks the journaled docs, though the
tiler has them enrolled — invisible to reconcile. So from the alias move on a job that
stops (its process died or was shut down, a backend failed it) keeps the status
`running` and lets go of `revision:on`; `resume_orphaned` — every process, at startup
and on the reconcile interval — finds a `running` job nobody holds whose `collection`
is the serving one, takes it over and runs only its tail: the wait for the enrolls in
flight, the sync, the invariant. Everything the tail needs outlives the process: the
journal in Valkey, and in the status the collections' names and — published before a
rollback's repoint — its `retired_at`.

**render_version.** The tiler binds `render_version` to its volume like geometry and
never serves a doc rendered by another one; every tile it returns names its
`X-Render-Version`. virag records the `render_version` on each point (and on the
collection: set at first bind and by every revision). Chat compares the two per tile and
answers `409 RENDER_VERSION_MISMATCH` on a difference — there is no serving state to
track. At startup a collection embedded from another `render_version` than the tiler's
is logged, and with `VIRAG_REVISION_ON_RENDER_MISMATCH=true` (default `false`) rebuilt
right away. The upgrade path: the tiler revises its docs (virag cannot start meanwhile)
→ virag starts, rebuilds its collection (chat answers `RENDER_VERSION_MISMATCH` until the
swap) → both serve.

## 10. Coordination of virag processes

Chat needs nothing shared: it reads through the Qdrant alias and the tiler. Four things
do, and they live in Valkey, behind the **`Coordinator`** (`coordination.py`); Qdrant
stays the source of truth for the collections — the alias is the one atomic pointer to
the serving collection, resolved afresh wherever it matters:

1. the revision / rollback job — one at a time, its status readable from any process;
2. the journal of documents enrolled while the job runs;
3. the documents being enrolled — one enroll of a document at a time; they also seed
   the journal, are what the job waits for after a swap, and are left alone by
   reconcile and the sync;
4. the reconcile pass — one per interval across the cluster.

**Leases.** Every process holds one key with a TTL, `procs:<token>`
(`VIRAG_PROCESS_TTL_SEC`, 30 s), renewed every ttl/3 with `SET XX` — only while it still
exists, so an expired lease never comes back: a dead token stays dead.

**What a process holds** is one key each, and they are all alike: the *name* says what
is held, the *value* is the holder's token, and there is no TTL. It is taken with
`SET NX`; once the holder's lease is gone it is free and is taken over with
`SET IFEQ <dead token>`, so of two processes racing for it exactly one wins; it is
released with `DELIFEQ <own token>`, which never removes another holder's key. Before
an irreversible step the holder checks the key is still its own and its lease alive
(fencing). A process finding its own lease expired (its
event loop stalled, or Valkey was gone, longer than the TTL) stops itself with SIGTERM:
the others already count it dead. On a graceful shutdown it hands the lease back, so
nobody waits out the TTL.

| Key (under `VALKEY_PREFIX`) | Holds |
|---|---|
| `procs:<token>` | the lease: `{host, pid, started_at}`, with the TTL |
| `enroll:<doc_id>` | the token of the process enrolling this document |
| `revision:on` | the token of the process running the revision / rollback; while it exists, enrolls journal their documents |
| `reconcile:on` | the token of the reconcile leader |
| `revision:status` | the job status, with its `owner` |
| `revision:journal` | the doc_ids enrolled while the job ran |

**Rules.**

- *Job.* `SET revision:on <mine> NX`; if it is held, the holder's lease decides: alive
  → `409` with its status; dead → `SET IFEQ` takes it over (the new job starts from
  scratch; a leftover build is dropped). The status is published only while the job is
  still this process's; a `running` status whose owner's lease is gone reads as
  `failed: owner lost`. A job that reads `running` while no other live process holds
  `revision:on` is orphaned (`Coordinator.orphaned_job`): finished if it had moved the
  alias (§9), left for the next request otherwise. Before the swap (or the rollback's repoint) the job checks it
  still holds `revision:on` and its lease — the fencing that keeps a replaced process
  from swapping. On the way out it publishes the outcome and `DELIFEQ`s the lock.
- *Enrolls and journal.* An enroll takes its document (`enroll:<doc_id>`; held by a
  live process → `409 ENROLL_IN_PROGRESS`), then journals it if `revision:on` exists,
  then resolves the alias. The job sets `revision:on` first, then snapshots the
  documents being enrolled into the journal. Each enroll is therefore in the snapshot or
  sees the flag — no transaction needed; one that finished before the snapshot finished
  before the job read the tiler. The journal is read right after the swap: an enroll
  that resolved the old collection journaled its document before it resolved.
- *Drain.* Right after the swap the job notes the documents being enrolled and polls
  until those enrolls are over: any of them may have resolved the old collection, while
  whoever takes a document later resolves the new one. Which collection an enroll
  writes is therefore not kept in Valkey. Documents held by dead processes are ignored
  and their keys dropped; they stay pending in the tiler for reconcile.
- *Who only looks.* Reconcile's cleanup and the sync after a swap never take a
  document: they list the documents being enrolled (a scan of `enroll:*`) and skip
  them.
- *Reconcile lead.* Each tick, every process: lead already → run; `reconcile:on` free →
  `SET NX`; leader dead → `SET IFEQ`; else skip. The lead stays while the leader lives
  and moves within one TTL plus one interval after it dies.

**Failure modes.** A process killed outright loses its in-flight requests (the client
retries, on another process); `restart` brings it back under a new token. The documents it was
enrolling are free once its lease is gone; a job it ran reads as `owner lost`: with the
alias untouched the next `POST /revision` takes over, and with the alias moved a live
process finishes it at its next upkeep (§9). Valkey unreachable: chat keeps serving, enroll answers `503 NOT_READY`, a job
fails at its next step without swapping, reconcile skips its pass; past the TTL the
processes restart on their expired leases. Valkey runs with AOF, so a short restart
keeps the leases and the journal.

An upstream that vanishes without closing its connections (its host or the network went
down) is noticed in about 25 s (TCP keepalive, §6): the enroll reading its stream fails,
rolls back and releases its document. The reverse is slower, and a known limitation: a
virag process that vanishes the same way mid-enroll of a **new** document leaves the
tiler writing into a dead connection until its kernel gives up (15–30 min by default),
and until then that document cannot be enrolled again — the tiler answers
`409 INGEST_IN_PROGRESS`. It clears by itself; restarting the tiler clears it at once.

## 11. Module layout

```
virag/
  main.py            # app, lifespan (clients, dependency checks, binding, lease, reconcile, render_version check), router order
  settings.py        # pydantic-settings sections: every env var, its default and bounds
  binding.py         # the collection ↔ tiler volume binding + tiler metadata-schema guard
  coordination.py    # Coordinator: leases, documents being enrolled + journal, drain, the revision job, the reconcile lead
  enroll.py          # Enroller: a tile stream of the tiler (an ingest, or a stored doc) → a document's points; the one writer
  reconcile.py       # Qdrant ⊆ Tiler self-repair (startup + periodic): the invariant, abandoned enrolls
  revision.py        # RevisionManager: background revision / rollback job, the flat re-embed of its build, journal sync
  errors.py          # ApiError + the uniform {"error": {...}} envelope
  responses.py       # OrjsonResponse, the app-wide default response class
  utils.py           # non-domain helpers (error_body, gather, cancel, chunked, unavailable_as_503)
  api/
    chat_completions.py  # /v1/chat/completions: rag trigger, retrieval, injection, sources
    enroll.py            # /enroll: name policy, the tiler's ingest stream → Enroller
    revision.py          # /revision (start, status) and /revision/rollback
    models.py            # /v1/models: the VLM's entry for VLM_MODEL_NAME
    auth.py              # VIRAG_API_KEY / VIRAG_ADMIN_API_KEY router dependencies
    ops.py               # /health, /ready
  clients/
    retriever_client.py  reranker_client.py  qdrant_store.py  valkey_store.py  tiler_client.py  vlm_client.py
```
