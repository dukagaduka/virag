# Security

Four things worth understanding before exposing virag: **virag's own keys** (who may
chat, who may enroll), the **keys virag holds** for its backends (its clients never see
them), the **volume binding** (virag only ever serves a collection embedded from *its*
tiler volume), and the **tile-integrity check** (the model only ever sees the exact bytes
that were embedded). Plus the **non-root** container.

---

## 1. virag's own keys

Both are sent as `Authorization: Bearer <key>` and compared in constant time; Swagger UI
(`GET /docs`) has an **Authorize** button for them.

| Key | Opens |
|---|---|
| `VIRAG_API_KEY` | `/v1/chat/completions`, `/v1/models` — a client uses it as if it were the LLM's key. |
| `VIRAG_ADMIN_API_KEY` | `/enroll`, `/revision`, `/revision/rollback` — and `/v1/*` too. |

- A missing or wrong key answers `401 UNAUTHORIZED` (`WWW-Authenticate: Bearer`); the
  user key on an admin route answers `403 FORBIDDEN`.
- `GET /health` and `GET /ready` stay open for orchestrators and balancers.
- **An empty key leaves its routes open**, logged as a `WARNING` at every start.
  Acceptable only where nobody untrusted reaches the port.
- virag has no other routes: the old `/qdrant`, `/tiler`, `/retriever` and `/v1/*`
  passthroughs are gone. Whoever needs a backend directly reaches it on its own port,
  with its own key.

## 2. Keys virag holds

virag authenticates to every backend itself; none of these keys ever leaves virag.

| Backend | Key | Must equal |
|---|---|---|
| VLM | `VLM_API_KEY` | the VLM's key |
| retriever | `RETRIEVER_API_KEY` | `RETRIEVER_DEPLOY_API_KEY` of `deploy/retriever` |
| reranker | `RERANKER_API_KEY` | `RERANKER_DEPLOY_API_KEY` of `deploy/reranker` |
| tiler | `TILER_API_KEY` | the tiler's `TILER_API_KEY` |
| Qdrant | `QDRANT_API_KEY` | Qdrant's `service.api_key` |
| Valkey | the password in `VALKEY_URL` (`redis://:<password>@host:6379/0`) | Valkey's `requirepass` |

A copy that does not match fails the startup check of that backend with a message
naming the key (`the tiler rejects TILER_API_KEY`, `grpc_status: UNAUTHENTICATED` for
Qdrant, `Valkey rejects the password in VALKEY_URL`). An empty `TILER_API_KEY` /
`QDRANT_API_KEY` sends no key.

vLLM's `--api-key` guards `/v1/*` only. A reranker's `/score` and `/rerank` answer
without the key, so the key alone does not lock the reranker down: keep its port private
to virag.

## 3. The volume binding

A pixel-RAG answer is only meaningful if the Qdrant vectors and the tiler that
regenerates their tiles belong together. `render_version` proves two components were
*configured* the same way; it cannot prove they are the *same instance*.

**Mechanism** (`binding.py`, enforced at startup in `lifespan`). The tiler gives each
volume a random `volume_id` at init and reports it on `GET /info`.

1. A **fresh (unbound) collection** is stamped with the tiler's `volume_id` (as
   `tiler_volume_id`) and its `render_version` on first run.
2. On an already-bound collection, another `volume_id` is **fatal** (`BindingError`):
   the tiler at `TILER_BASE_URL` serves another volume. Most importantly a tiler that
   came back on a fresh, empty volume is refused — otherwise reconcile would take every
   point for an orphan of a deleted document and delete it.
3. `render_version` **drift** (collection embedded from one version, the tiler renders
   another) is not fatal at startup: chat answers `409 RENDER_VERSION_MISMATCH` until
   the collection is rebuilt with `POST /revision` (or at startup with
   `VIRAG_REVISION_ON_RENDER_MISMATCH=true`).

The `volume_id` is not a secret — it identifies, the keys above authenticate. To start
over on a fresh tiler volume, drop the Qdrant collection too and re-enroll.

virag also enforces the **`Qdrant ⊆ Tiler` invariant**: every Qdrant point is backed by
a tiler tile. Reconcile (at startup and on an interval) deletes any Qdrant `doc_id` the
tiler no longer has or has quarantined, so a query never returns a hit whose tile the
tiler can't regenerate. See [ARCHITECTURE.md](ARCHITECTURE.md) for the reconcile passes.

## 4. Tile-integrity check

At enroll time, each tile's `image_hash` (sha256 of its JPEG bytes) is stored in the
Qdrant payload alongside the vector. At query time, virag regenerates each retrieved
tile from the tiler and recomputes its hash; a tile whose hash does not match — or whose
point carries no `image_hash` at all — is **dropped** (logged as `integrity: …`) and
never reaches the model.

## 5. Non-root container

- Runs as uid/gid **10001**; nothing runs as root (`Dockerfile`).
- Stateless — no volumes to get ownership wrong; all persistent state is in the
  backends (Qdrant, and the coordination state in Valkey).
- All unexpected exceptions are caught and rendered as a generic `500` envelope, so an
  internal error never leaks a stack trace to the caller.

---

## Operator checklist

- [ ] `VIRAG_API_KEY` and `VIRAG_ADMIN_API_KEY` set to strong random values
      (`openssl rand -hex 32`), different from each other.
- [ ] `TILER_API_KEY`, `QDRANT_API_KEY` and the Valkey password set, each identical on
      virag and on its backend.
- [ ] Real `VLM_API_KEY` / `RETRIEVER_API_KEY` / `RERANKER_API_KEY` if the upstreams enforce auth.
- [ ] Reranker port reachable by virag only: its `/score` is not covered by the vLLM key.
- [ ] TLS in front of virag (a reverse proxy) wherever its port is reachable beyond a
      trusted network: the keys travel as bearer tokens.
