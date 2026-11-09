# API guide

virag speaks the OpenAI `/v1` protocol. The one addition is a `rag` field on chat
completions that turns on pixel-RAG; everything else is either an ordinary OpenAI call
(passed through) or an operational endpoint.

A **live, always-accurate** OpenAPI document is served by the app itself:

- Interactive docs (Swagger UI): `GET /docs`
- Raw spec: `GET /openapi.json`

This page explains the semantics that a schema can't — especially the `rag` trigger and
how sources are returned. Base URL in examples: `http://localhost:48000`.

**Authentication.** `/v1/*` needs `Authorization: Bearer <VIRAG_API_KEY>` (or the admin
key); `/enroll` and `/revision*` need `Authorization: Bearer <VIRAG_ADMIN_API_KEY>`.
`/health` and `/ready` are open; an empty key leaves its routes open. The examples below
omit the header — add `-H "Authorization: Bearer $VIRAG_API_KEY"` (or the admin key). An
OpenAI SDK takes `VIRAG_API_KEY` as its `api_key`. See [SECURITY.md](SECURITY.md).

---

## `POST /v1/chat/completions`

An OpenAI-compatible chat completion. Without a `rag` field it is a **plain
passthrough** to the upstream VLM — same request, same response. With a `rag` field it
runs retrieval first and augments the last user message with document-image tiles (see
[ARCHITECTURE.md](ARCHITECTURE.md), "The retrieval pipeline").

### The `rag` field

Put it at the top level of the request body. With an OpenAI SDK, pass it via
`extra_body` (which flattens to top-level JSON).

| Form | Effect |
|---|---|
| absent / `false` | RAG off — plain passthrough. |
| `true` | RAG on with server defaults (`RETRIEVER_TOP_K`, `RETRIEVER_THRESHOLD`, `RERANKER_TOP_N`, `RERANKER_THRESHOLD`, `VIRAG_FRAMING_TEXT`), no sources. |
| object | RAG on with overrides. |

`rag` object fields (all optional):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `top_k` | positive int | `RETRIEVER_TOP_K` | Candidate tiles to take from Qdrant. Without a reranker they all reach the VLM. |
| `retriever_threshold` | number \| null | `RETRIEVER_THRESHOLD` | Minimum Qdrant score of a candidate, applied by Qdrant in the search (a floor for `Cosine`/`Dot`, a ceiling for `Euclid`/`Manhattan`); `null` = none. |
| `top_n` | positive int | `RERANKER_TOP_N` | With a reranker: tiles it passes to the VLM, best first; at most `top_k`. |
| `reranker_threshold` | number in `[0, 1]` \| null | `RERANKER_THRESHOLD` | With a reranker: minimum relevance of a tile; `null` = none. |
| `framing_text` | string | `VIRAG_FRAMING_TEXT` | Text prepended to the injected tiles. |
| `include_sources` | bool | `false` | Return the `sources` array (below). |

A wrong type on any of these is a `422`, and so is a `top_n` above `top_k` while the
reranker is on (without one, `top_n` and `reranker_threshold` are checked for type and
otherwise ignored). RAG on but no user text to build a query from is also a `422`.

With a reranker configured, the `top_k` candidates are scored against the query text and
the VLM gets those at or above `reranker_threshold`, best first, at most `top_n`. A
reranker that fails never fails the chat: that request takes the retriever's order, cut
to `top_n`, and virag logs it (see [CONFIG.md](CONFIG.md#reranker-optional)).

A retrieved tile the tiler refuses (gone, quarantined, or hashing differently than
enrolled) is skipped. Two failures fail the whole request instead of answering without
context: a tiler that is unreachable or not serving (`503 NOT_READY`), and a tile the
tiler now renders by another `render_version` than the one it was embedded from
(`409 RENDER_VERSION_MISMATCH` — the collection is stale and must be rebuilt with
[`POST /revision`](#post-revision)).

```bash
# RAG on with defaults
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "my-vlm",
  "messages": [{"role": "user", "content": "What was Q3 revenue?"}],
  "rag": true
}' | jq

# RAG with overrides + sources
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "my-vlm",
  "messages": [{"role": "user", "content": "What was Q3 revenue?"}],
  "rag": {"top_k": 16, "retriever_threshold": 0.2, "top_n": 4, "reranker_threshold": 0.1, "include_sources": true}
}' | jq
```

Python (OpenAI SDK):

```python
client.chat.completions.create(
    model="my-vlm",
    messages=[{"role": "user", "content": "What was Q3 revenue?"}],
    extra_body={"rag": {"top_k": 8, "include_sources": True}},
)
```

### Sources

When `include_sources` is true, each injected tile is reported, in prompt order, as
`{name, doc_id, tile_num, score}` (`name` from the tile's `metadata.name`, `score` the
Qdrant one), plus `rerank_score` for a tile the reranker scored (absent without a
reranker, or when the request fell back to the retriever's order).

- **Non-streaming:** added as a top-level `"sources"` key on the (200, JSON) response.
- **Streaming (`"stream": true`):** delivered as a synthetic
  `chat.completion.chunk` spliced in **immediately before** `data: [DONE]`:

  ```
  data: {"object":"chat.completion.chunk","choices":[],"sources":[ ... ]}

  data: [DONE]
  ```

Upstream errors and non-JSON bodies pass through unchanged; sources are only added on a
successful response.

---

## `POST /enroll`

Tile a document, embed its tiles, and upsert them into Qdrant so they become
retrievable. `multipart/form-data`.

| Field | In | Meaning |
|---|---|---|
| `doc_type` | form | `PDF \| URL \| MHTML \| IMG` (required). |
| `document` | file | The bytes (required unless `doc_type=URL`). |
| `url` | form | The URL (for `doc_type=URL`). |
| `metadata` | form | A JSON **object** as a string. Should include `name`. |
| `force` | query | Re-tile even if the doc already exists. |
| `allow_missing_name` | query | Opt out of the name requirement (see below). |

**Naming policy.** Every enrolled document is named: `metadata.name` must be a
non-empty string, else `422`. `allow_missing_name=true` is an explicit opt-out — the
document is then named by its content hash (`doc_id`). Point ids are deterministic
(`uuid5(doc_id:tile_num)`), so re-enrolling a document overwrites its own tiles rather
than duplicating them. A document that produces zero tiles enrolls nothing (reported as
`enrolled: 0`).

**Streaming ingest.** Internally, `/enroll` consumes the tiler's tile stream and embeds
and upserts tiles in batches as they arrive — it never buffers the whole document in
memory, so arbitrarily large documents (e.g. a 920-page PDF) enroll fully without hitting
a render timeout. The request/response contract above is unchanged. On any failure
mid-enroll the document's partial points are rolled back out of Qdrant; the tiler doc is
left in place and marked pending so [reconcile](ARCHITECTURE.md) heals or sweeps it later.

Response:

```json
{"doc_id": "…", "doc_type": "PDF", "render_version": "…", "tile_count": 12, "enrolled": 12}
```

```bash
curl -s -X POST localhost:8080/enroll \
     -F doc_type=PDF -F document=@report.pdf \
     -F 'metadata={"name":"2024 Annual Report","project":"atlas"}' | jq
```

---

## `POST /revision`

Re-embed the **whole corpus** into a fresh vector collection and swap it in. Use it
after a retriever-model change, with no downtime, and after a tiler render change: until
then chat answers `409 RENDER_VERSION_MISMATCH`. With `VIRAG_REVISION_ON_RENDER_MISMATCH=true`
virag starts it by itself when it starts against a collection embedded from another
`render_version` than the tiler's (default `false`: the mismatch is only logged). It is
not periodic maintenance — with nothing changed it would recompute the same vectors.

virag serves through a stable Qdrant **alias**; `/revision` builds a new physical
collection alongside the live one, re-embeds every *complete* tiler doc into it, then
**atomically** repoints the alias. The live collection keeps serving the entire time.
Enrolls that happen meanwhile are journaled and synced into the new collection right
after the swap. If any document still fails after a retry, the alias is **not** swapped
and the old collection keeps serving. The replaced collection is kept for
[`/revision/rollback`](#post-revisionrollback) until the next revision.

It runs in the **background**, in whichever virag process the request lands on: the
call returns `202` with the job status at once. One job runs at a time across every
process — a second call while one runs is `409 CONFLICT` (with that job's status), and
so is a call while the tiler is not serving (its own revision is pending or running). A
job whose process died is taken over by the next call, from scratch. Incomplete and quarantined tiler docs
are skipped (they have no servable tiles; re-enroll them). The new collection is stamped
with the tiler's `render_version`; if the tiler's changes before the swap, the job fails
without swapping.

```bash
curl -s -X POST localhost:8080/revision | jq
```

## `GET /revision`

Status of the current or last revision / rollback job, the same from every virag
process (it is kept in Valkey):

```json
{
  "kind": "revision",
  "state": "succeeded",
  "phase": "syncing",
  "owner": {"token": "…", "host": "virag-1", "pid": 1, "started_at": 1761129000},
  "started_at": 1761130000,
  "finished_at": 1761130950,
  "collection": "store-collection__v2",
  "render_version": "f04061fd38ea887a",
  "total": 130, "skipped": 2, "reembedded": 128, "synced": 3
}
```

`state` is `idle` (no job yet), `running` (`phase`: `building`, then `syncing` after the
swap), `succeeded`, `degraded` (swapped, but some journaled docs failed to sync — listed
in `sync_failed_doc_ids`; re-enroll them), or `failed` (`error`, and `failed_doc_ids`
when the alias was not swapped). A job whose process died reads as `failed` with
`error: "owner lost (<host>/<pid>)"`; `POST /revision` again.

## `POST /revision/rollback`

Repoint the alias back to the collection the last revision replaced, drop the abandoned
one once the enrolls still writing into it are done, and re-sync the documents enrolled
since the swap. Background, like `/revision`
(`202`; `409` while a job runs or when there is no previous collection). Revert the
retriever configuration first — the old collection holds the old model's vectors (a
vector-size mismatch is refused). A collection embedded from another `render_version`
than the tiler renders now is refused too: chat would answer `RENDER_VERSION_MISMATCH`.

---

## `GET /v1/models`

The OpenAI model list, with exactly one model: the upstream VLM's own entry for
`VLM_MODEL_NAME`, the model virag is set up (and vision-probed) for. A chat request's
`model` is forwarded to the upstream as is, so clients should send this one. When the
upstream lists no such entry or does not answer, a minimal
`{"id": VLM_MODEL_NAME, "object": "model", …}` stands in. No other `/v1/*` route exists.

---

## Operational

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — always `{"status":"ok"}`, touches nothing; the container `HEALTHCHECK`. |
| `GET /ready` | Readiness — `200 {"status":"ready"}` once all backends are wired; `503 {"status":"not ready","missing":[…]}` otherwise. |

---

## Error envelope

Every error uses one shape:

```json
{"error": {"code": "UNPROCESSABLE", "message": "…", "details": {"…": "…"}}}
```

| Status | `code` | Typical cause |
|---|---|---|
| `401` | `UNAUTHORIZED` | Missing or wrong `VIRAG_API_KEY` / `VIRAG_ADMIN_API_KEY` (with `WWW-Authenticate: Bearer`). |
| `403` | `FORBIDDEN` | The user key on an admin route (`/enroll`, `/revision*`). |
| `409` | `RENDER_VERSION_MISMATCH` | A retrieved tile is rendered by another `render_version` than it was embedded from — rebuild with `POST /revision`. |
| `422` | `UNPROCESSABLE` | Bad request body/JSON, invalid `rag` field, RAG on with no query text, bad `metadata`. |
| `502` | `UPSTREAM_FAILURE` | A backend (VLM, tiler, retriever, Qdrant) could not be reached. |
| `503` | `NOT_READY` | Queried before startup finished wiring backends; the tiler is unreachable / not serving during a RAG chat; Valkey is unreachable during an enroll or a revision call. |
| `500` | `INTERNAL_ERROR` | Unexpected server error. |

Backend services can surface their own statuses (e.g. the tiler's `409`/`413`) through
the enroll path; those pass through with the originating service's meaning.
