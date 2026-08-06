# Streaming enroll — architecture design

Status: Phases 1 & 2 implemented (tiler streaming ingest + virag streaming
consumer + reconcile/self-heal). Phase 3 (revisions, delete-propagation) pending.
Scope: the `enroll` pipeline across **doc-tiler** (renders documents to tiles) and
**virag** (embeds tiles, indexes them in Qdrant).

---

## 1. Why

The current `enroll` renders a whole document, returns one manifest, embeds every
tile, then upserts every point — all in one shot. It collapses on large documents.
Observed on `milencdict.pdf` (141 MB, 920 pages, 2048-dim vectors):

- **A. Render exceeds the HTTP timeout.** `POST /tile` is one synchronous request;
  a 920-page render outruns `TILER_TIMEOUT_SEC` → `ReadTimeout` → 503.
- **B. The manifest is ~40 MB** of base64 tiles → exceeds Qdrant's 32 MB body limit
  on upsert (mitigated separately by byte-budgeted upsert chunking).
- **C. The Qdrant write is neither atomic nor self-cleaning.** A mid-write crash
  leaves partial points; re-enroll with fewer tiles orphans stale points; there is
  no rollback and no reconciliation.

`B` is patched. `A` and `C` are the architectural work here.

## 2. Invariant

**`Qdrant ⊆ Tiler`** — every Qdrant point's `doc_id` (and `tile_num`) must exist in
the tiler. The tiler MAY hold more (tiles/docs not yet, or no longer, in Qdrant);
Qdrant MUST NOT hold a point the tiler can't regenerate (else a query returns a hit
whose bytes `get_tile` answers with 404).

Two ordering rules follow and are enforced everywhere:

| Operation | Order | Reason |
|---|---|---|
| Write   | tiler tile-row → stream → Qdrant point | every point is backed by a tile |
| Delete  | Qdrant → tiler | never leave a point without its tile, even briefly |

Because the stream writes strictly "tile-row → send → point", the invariant holds
**continuously**, mid-stream included.

## 3. Identity & dedup

- `doc_id = sha256(document bytes)` — content-addressed, computed by the tiler,
  consumed verbatim by virag. Idempotent for uploaded files. (URL/MHTML captures are
  non-deterministic — a known duplication path, out of scope here.)
- Tile identity: `(doc_id, tile_num)`. `image_hash = sha256(jpeg)` is an integrity
  check, not an identity key.
- Qdrant point id: `uuid5(NAMESPACE, f"{doc_id}:{tile_num}")` — **deterministic**,
  no per-ingest generation. A re-enroll overwrites the same ids in place; the delta
  (shrunk tile range) is removed by `delete_by_doc` + reconcile.

Dedup gaps this design closes: stale-tile orphaning (via `delete_by_doc` on enroll
start + reconcile), non-atomic partial enroll (via pending marker + rollback +
reconcile), per-doc render_version staleness (whole-doc replace on re-enroll).

## 4. Two completion markers

Streaming introduces partial state at both levels; each level owns one marker,
cleared by its owner.

| Marker | Home | Means | Set by | Cleared by |
|---|---|---|---|---|
| `docs.complete` (bool) | Tiler | tiler render finished; gates `get_tile`/manifest visibility | tiler, on last tile | tiler (reset to false at the start of a `force` re-render) |
| pending (metadata) | Tiler docs metadata, **virag semantics** | Qdrant not yet confirmed for this enroll | virag, at enroll start | virag, after Qdrant write confirmed |

`complete` is a generic tiler concern (its own render lifecycle) — the tiler stays
self-sufficient. Pending is virag's, expressed through the tiler's generic,
already-existing indexed-metadata mechanism (no bespoke coupling, no extra table).

### Pending encoding (indexed metadata)

The tiler's metadata schema supports only `text` and `number` (no bool). Pending is
two **number** keys, declared in `deploy/tiler_metadata_schema.yaml`:

- `virag_started_at` — epoch seconds at enroll start (for the TTL / anti-race gate).
- `virag_enrolled_at` — `0` while pending, `>0` (epoch) once Qdrant is confirmed.

Together with the existing `name`, that is **three indexed fields**. virag verifies
at startup — via the tiler's `GET /schema/metadata` — that all three are declared;
a missing field is fatal (reconcile's metadata search would otherwise 400). The
tiler never interprets these keys; it only indexes them.

## 5. Flow (end-to-end streaming)

```
TILER  POST /tile (NDJSON stream):
  1. doc_id = sha256(blob); blobs.put(doc_id, blob)
  2. doc-row {tile_count=N (known up front for PDF), complete=false}   # + virag metadata
  3. render group-by-group (as_completed):
       per tile:  insert tile-row {tile_num, height, image_hash}       # WRITE
                  stream line  {tile_num, height, image_hash, data}    # then SEND
                  drop the JPEG (keep only the tiny row)
  4. last tile → complete=true
  5. final line {"event":"committed", doc_id, tile_count}

VIRAG  stream consumer:
  V1. mark pending: PATCH metadata {virag_started_at=now, virag_enrolled_at=0}
  V2. delete_by_doc(doc_id)                    # clear any prior partial for this doc
  V3. accumulate tiles up to EMBEDDINGS_CHUNK_SIZE
  V4. embed batch → drop JPEGs → upsert points (byte-budget chunked)
  V5. repeat to end of stream
  V6. on "committed": PATCH metadata {virag_enrolled_at=now}   # promote / clear pending

QDRANT point = vector + id(doc_id, tile_num) + payload{doc_id, tile_num,
               render_version, image_hash, metadata}
```

Memory is bounded at every hop: tiler holds tile *rows* (KB), virag holds ≤ one
embed-batch of JPEGs, Qdrant receives byte-bounded batches. The timeout dissolves —
bytes flow every few seconds, so httpx's inter-byte read timeout never trips.

## 6. Failure handling

| Failure | Left behind | Cleaned by |
|---|---|---|
| Tiler render crash | `complete=false` doc (blob + partial rows) | tiler incomplete-sweeper (§7) |
| Stream break / caught virag error | partial Qdrant points; pending still `0` | rollback: `delete_by_doc` (+ reconcile) |
| Hard virag crash mid-upsert | partial points; pending `0` | reconcile on startup / periodic |
| Between last tile and Qdrant confirm | partial points; pending `0` | reconcile |

Rollback deletes **only** from Qdrant and leaves the tiler doc in place (the
invariant allows `Tiler ⊇ Qdrant`); the parked doc is healed or swept. So the hot
rollback path never needs `DELETE /doc` on the tiler → `TILER_ALLOW_DELETE` is not
required for rollback (only for optional GC of truly-abandoned docs).

## 7. TTL / GC — two owners, two domains

- **Tiler-side (generic, self-sufficient):** an incomplete-ingest sweeper deletes
  `complete=false` docs older than `TILER_INCOMPLETE_TTL_SEC`, **excluding in-flight
  ingests** (tracked in memory). Deletes the doc **whole** — blob + doc-row + partial
  tile-rows in one `delete_doc` (FK cascade + blob unlink). The blob is the storage
  win; the rows go with it (a half-doc is unusable and keeping just the blob buys
  nothing but a disk-write dedup). TTL must exceed the longest render, so an active
  render is never killed. Reclaiming an incomplete doc means it must be re-uploaded to
  enroll again — correct for an abandoned ingest.
- **virag-side (virag semantics):** reconcile (§8). Never deletes tiler docs on the
  hot path; only Qdrant points. Optional deletion of truly-abandoned docs via
  `DELETE /doc` is operator-gated.

The two coordinate implicitly: when the tiler GCs an incomplete doc, its `doc_id`
leaves the tiler, and virag's `Qdrant ⊆ Tiler` pass removes any Qdrant remnants.

## 8. Reconcile (replaces `warn_doc_id_overlap`)

The old check sampled 5 doc_ids and only warned. Replaced by an authoritative,
repairing pass, run at startup and periodically (`VIRAG_RECONCILE_INTERVAL_SEC`,
0 = disabled):

1. **Enforce `Qdrant ⊆ Tiler`.** `tiler_ids` = all doc_ids via `POST /docs/search`
   (paginated); `qdrant_ids` = distinct doc_ids via Qdrant scroll;
   `dangling = qdrant_ids − tiler_ids` → `delete_by_doc` each. (Set diff, not N RPCs.)
2. **Clear stale pending.** `POST /docs/search {virag_enrolled_at == 0 AND
   virag_started_at < now − VIRAG_PENDING_TTL_SEC}` → for each: `delete_by_doc`
   (drop partial), then heal.
3. **Self-heal** (`VIRAG_SELF_HEAL`, default on): a pending doc whose tiler
   `complete=true` is re-embedded from the tiler (`get_tile` per tile) and its
   `virag_enrolled_at` set; a `complete=false` doc is left for `tiler_revision`.

The `VIRAG_PENDING_TTL_SEC` gate is the anti-race safeguard: an in-flight enroll has
`started_at ≈ now`, so it is skipped. Because the tiler is bound to this virag by
secret, **every** tiler doc carries virag metadata — there are no "foreign" docs, so
`tiler_ids − qdrant_ids` are all legitimate self-heal candidates.

`enforce_binding` (the db_secret / render_version cryptographic pairing) is a
separate, unchanged concern — identity, not consistency.

## 9. Revision — two operations

- **`tiler_revision`** — re-renders docs in the tiler (render_version change, or
  healing incomplete docs). Produces fresh tiles, sets `complete`. Does not touch
  pending (Qdrant not yet updated).
- **`virag_revision` (full)** — re-embeds all tiler docs into a **new** collection,
  then **blue-green swaps via a Qdrant alias** (build `v_next` → fill → repoint the
  serving alias atomically → drop the old collection). No empty window, no copy.
  virag always queries through the alias; the binding (secret / render_version)
  carries to the new collection. Clears pending as it embeds each doc.

`tiler_revision` (re-render) + `virag_revision` (re-embed) together fully reconcile
both stores. Distinguish the trigger: embedding-model change → `virag_revision`
only; render_version change → `tiler_revision` first, then `virag_revision`.

## 10. Phasing

- **Phase 1 — Qdrant correctness (virag) — DONE.**
  `delete_by_doc` + `doc_id` payload index; pending metadata markers + rollback in
  `enroll`; reconcile replacing `warn_doc_id_overlap` (`⊆` enforcement + pending
  cleanup); startup schema-fields check; new env + settings. Fixes gaps #1 (orphans)
  and #2 (atomicity).
- **Phase 2 — streaming — DONE.** Tiler `POST /tile/stream` NDJSON (`begin` → `tile`*
  → `committed`/`error`), `complete` flag + incremental store writes + incomplete
  sweeper; virag streaming consumer (batched embed→upsert with rollback); reconcile
  self-heal re-embeds complete pending docs via `get_tile`. Fixes the render timeout
  and bounds memory at every hop.
- **Phase 3 — revisions & remaining gaps — PENDING.** `tiler_revision`,
  `virag_revision` (alias blue-green), cross-service delete-propagation, URL-dedup,
  hash-gated re-embed.

## 11. New configuration

| Var | Home | Default | Purpose |
|---|---|---|---|
| `VIRAG_SELF_HEAL` | virag | `true` | reconcile re-embeds complete pending docs |
| `VIRAG_RECONCILE_INTERVAL_SEC` | virag | `300` (`0` = off) | periodic reconcile cadence |
| `VIRAG_PENDING_TTL_SEC` | virag | `3600` | pending age before reconcile acts (anti-race) |
| `TILER_INCOMPLETE_TTL_SEC` | tiler | `0` (off) | incomplete-ingest sweep age (> longest render) |

Plus `virag_started_at`, `virag_enrolled_at` (number) added to
`deploy/tiler_metadata_schema.yaml`.
