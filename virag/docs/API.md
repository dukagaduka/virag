# API guide

virag speaks the OpenAI `/v1` protocol. The one addition is a `rag` field on chat
completions that turns on pixel-RAG; everything else is either an ordinary OpenAI call
(passed through) or an operational endpoint.

A **live, always-accurate** OpenAPI document is served by the app itself:

- Interactive docs (Swagger UI): `GET /docs`
- Raw spec: `GET /openapi.json`

This page explains the semantics that a schema can't — especially the `rag` trigger and
how sources are returned. Base URL in examples: `http://localhost:8080`.

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
| `true` | RAG on with server defaults (`VIRAG_TOP_N`, `VIRAG_SCORE_THRESHOLD`, `VIRAG_FRAMING_TEXT`), no sources. |
| object | RAG on with overrides. |

`rag` object fields (all optional):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `top_n` | positive int | `VIRAG_TOP_N` | Tiles to retrieve. |
| `score_threshold` | number \| null | `VIRAG_SCORE_THRESHOLD` | Drop hits below this Qdrant score. |
| `framing_text` | string | `VIRAG_FRAMING_TEXT` | Text prepended to the injected tiles. |
| `include_sources` | bool | `false` | Return the `sources` array (below). |

A wrong type on any of these is a `422`. RAG on but no user text to build a query from
is also a `422`.

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
  "rag": {"top_n": 8, "score_threshold": 0.35, "include_sources": true}
}' | jq
```

Python (OpenAI SDK):

```python
client.chat.completions.create(
    model="my-vlm",
    messages=[{"role": "user", "content": "What was Q3 revenue?"}],
    extra_body={"rag": {"top_n": 8, "include_sources": True}},
)
```

### Sources

When `include_sources` is true, each injected tile is reported as
`{name, doc_id, tile_num, score}` (`name` from the tile's `metadata.name`).

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

Re-embed the **whole corpus** into a fresh vector collection and swap it in with no
downtime. Use it after an embedding-model change (or after the tiler's own `/revision`
re-render) so the vectors are rebuilt against the new model.

virag serves through a stable Qdrant **alias**; `/revision` builds a new physical
collection alongside the live one, re-embeds every *complete* tiler doc into it (fetching
tiles from the tiler), then **atomically** repoints the alias to the new collection and
drops the old one. The live collection keeps serving the entire time — no empty window.

This is long-running (it re-embeds the entire corpus). Incomplete tiler docs are skipped
(they aren't servable; leave them for a tiler `/revision`). Response:

```json
{"reembedded": 128, "skipped": 2, "collection": "store-collection__v2"}
```

```bash
curl -s -X POST localhost:8080/revision | jq
```

---

## Passthrough proxy

Reach the backend services through the gateway. Each prefix is stripped and the rest of
the path (plus query string) is forwarded upstream, streaming the response.

| Route | Upstream | Notes |
|---|---|---|
| `ANY /tiler/{path}` | doc-tiler | e.g. `GET /tiler/get_tile?doc_id=…&tile_num=0`. |
| `ANY /embed/{path}` | embeddings server | `Authorization` injected from `VLLM_API_KEY`. |
| `ANY /qdrant/{path}` | Qdrant | e.g. `GET /qdrant/collections`. |
| `ANY /v1/{path}` | VLM | Every `/v1/*` **except** `/v1/chat/completions`; `Authorization` injected from `VLM_API_KEY`. |

An unreachable upstream returns `502`.

---

## Operational

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness — always `{"status":"ok"}`, touches nothing. |
| `GET /ready` | Readiness — `200 {"status":"ready"}` once all backends are wired; `503 {"status":"not ready","missing":[…]}` otherwise. |

---

## Error envelope

Every error uses one shape:

```json
{"error": {"code": "UNPROCESSABLE", "message": "…", "details": {"…": "…"}}}
```

| Status | `code` | Typical cause |
|---|---|---|
| `422` | `UNPROCESSABLE` | Bad request body/JSON, invalid `rag` field, RAG on with no query text, bad `metadata`. |
| `502` | `UPSTREAM_FAILURE` | A backend (VLM, tiler, embeddings, Qdrant) could not be reached. |
| `503` | `NOT_READY` | Queried before startup finished wiring backends. |
| `500` | `INTERNAL_ERROR` | Unexpected server error. |

Backend services can surface their own statuses (e.g. the tiler's `409`/`413`) through
the proxy and enroll paths; those pass through with the originating service's meaning.
