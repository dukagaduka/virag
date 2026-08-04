# Security

Four things worth understanding before exposing virag: the **binding secret** (which
guarantees virag only ever talks to *its* tiler and *its* Qdrant collection), the
**tile-integrity check** (the model only ever sees the exact bytes that were embedded),
how **upstream credentials** are handled on the proxy, and the **non-root** container.

---

## 1. The binding secret

A pixel-RAG answer is only meaningful if three things line up: the Qdrant vectors, the
tiler that can regenerate the tiles those vectors came from, and virag orchestrating
them. `render_version` proves two components were *configured* the same way; it cannot
prove they are the *same instance*. The binding secret closes that gap.

**Mechanism** (`binding.py`, enforced at startup in `lifespan`). `TILER_DB_SECRET` is
hashed once — `secret_hash = sha256(TILER_DB_SECRET)`, or `""` when unset — and must
agree across all three:

1. The **tiler** reports its `secret_hash` on `GET /info`. If it doesn't equal virag's,
   startup fails (`BindingError`) — the tiler is not bound to this virag (or was
   initialised without the secret).
2. The **Qdrant collection** carries a `secret_hash` in its collection metadata. On a
   fresh (unbound) collection, virag stamps it with the secret + the tiler's
   `render_version` on first run. On an already-bound collection, a mismatch is
   fatal — the collection belongs to a different virag/tiler pair.
3. `render_version` **drift** (collection bound to one version, tiler now reports
   another) is a **warning**, not fatal: stored `image_hash` values may no longer
   reproduce, which the integrity check (below) will then catch per tile.

**Operational rules:**

- One value, three consumers. Set `TILER_DB_SECRET` to the *identical* value on virag,
  the tiler, and (implicitly, via first-run stamping) the Qdrant collection — **before**
  first init. It is immutable per volume/collection thereafter.
- **Empty is an explicit, unbound slot.** An unset secret hashes to `""` and is stored
  as such; a stack initialised without one can't be bound later without a fresh
  collection/volume. "No secret" is a deliberate posture, not a default to silently
  upgrade.

There is also a best-effort **overlap check** at startup: virag samples a few enrolled
`doc_id`s from Qdrant and warns if the tiler doesn't have them — an early signal that
the two stores have drifted apart. It never blocks startup.

## 2. Tile-integrity check

At enroll time, each tile's `image_hash` (sha256 of its JPEG bytes) is stored in the
Qdrant payload alongside the vector. At query time, virag regenerates each retrieved
tile from the tiler and recomputes its hash; if it doesn't match the stored value, the
tile is **dropped** (logged as `integrity: … hash mismatch`) and never reaches the
model. So a pixel-tier drift on the tiler (a Chromium/JPEG/DPI change) degrades to
"fewer tiles", never "the model was shown different bytes than were embedded".

## 3. Upstream credentials on the proxy

The reverse proxy injects bearer credentials so backends can stay locked down behind the
gateway:

- `/embed/*` and calls to the embeddings server → `Authorization: Bearer <VLLM_API_KEY>`.
- `/v1/*` (passthrough) and chat-completions → `Authorization: Bearer <VLM_API_KEY>`.
- `/tiler/*` and `/qdrant/*` → no auth injected.

Notes:

- These keys live in the environment (`VLLM_API_KEY`, `VLM_API_KEY`) and default to
  `dummy`; set real values in any deployment where the backends enforce auth.
- virag itself does **not** authenticate its own callers — it is designed to sit behind
  your own gateway/network policy. Don't expose it directly to untrusted clients.
- Hop-by-hop headers (RFC 7230) and `Host`/`Content-Length` are stripped in both
  directions; the client's own `Authorization`, if any, is replaced (not merged) on the
  auth-injecting routes.

## 4. Non-root container

- Runs as uid/gid **10001**; nothing runs as root (`Dockerfile`).
- Stateless — no volumes to get ownership wrong; all persistent state is in the
  backends.
- All unexpected exceptions are caught and rendered as a generic `500` envelope, so an
  internal error never leaks a stack trace to the caller.

---

## Operator checklist

- [ ] `TILER_DB_SECRET` set to a strong value, identical on virag and the tiler, before
      the stack's first boot.
- [ ] Real `VLM_API_KEY` / `VLLM_API_KEY` if the upstreams enforce auth.
- [ ] virag reachable only from a trusted gateway/network — it has no built-in
      client authentication.
- [ ] Backends (tiler, Qdrant, embeddings) not separately exposed if the proxy routes
      are meant to be the only path to them.
