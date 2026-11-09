# Deployment

virag is a small stateless gateway; the interesting part of deploying it is the
**backends it needs**: doc-tiler, Qdrant, Valkey, a retriever server, an upstream VLM,
and optionally a reranker server. Deployment is split into independent Compose stacks
under [`deploy/`](../deploy/): [`service/`](../deploy/service/) (virag, doc-tiler,
Qdrant, Valkey), [`retriever/`](../deploy/retriever/) (the vLLM retriever server) and
[`reranker/`](../deploy/reranker/) (the optional vLLM reranker server). The VLM is
external (you point `VLM_BASE_URL` at it).

---

## Quickstart (Compose)

```bash
# 1. retriever server (GPU)
cd deploy/retriever
cp .env.example .env                                   # image, port, model/adapter paths
cp serve_retriever.example.sh serve_retriever.sh     # tune for your hardware
docker compose up -d

# 2. virag + tiler + Qdrant + Valkey
cd ../service
cp .env.example .env                                  # images, ports, volume paths
cp virag.env.example virag.env                        # keys, VLM_*, RETRIEVER_*, QDRANT_VECTOR_SIZE, …
cp tiler.env.example tiler.env                        # TILER_API_KEY, geometry, pools
cp configs/valkey.example.conf configs/valkey.conf    # requirepass
cp configs/qdrant.example.yaml configs/qdrant.yaml    # service.api_key
# replace every `dummy` with its own `openssl rand -hex 32` (see "Secrets" below)
docker compose up --build
curl -s localhost:48000/ready | jq
curl -s localhost:48000/v1/models -H "Authorization: Bearer $VIRAG_API_KEY" | jq
```

The `service/` stack (`deploy/service/docker-compose.yaml`) is four services on one bridge
network. The retriever server (`retriever-vllm-virag`, GPU, runs `serve_retriever.sh`) is a
separate stack in `deploy/retriever/` that virag reaches over `RETRIEVER_BASE_URL`:

| Service | Role | Notes |
|---|---|---|
| `virag` | the gateway | built from this repo; waits for its backends to be healthy. |
| `tiler-virag` | doc-tiler | built from `../../doc-tiler`; mounts `tiler_metadata_schema.yaml` (see below). |
| `qdrant-virag` | Qdrant | vector store. |
| `valkey-virag` | Valkey 9 | coordination of virag processes (leases, writers, the revision job, the reconcile lead); AOF on, no port published. |

### Connecting virag to the retriever stack

The two stacks do not share a Docker network. virag reaches the retriever server through
the host: set `RETRIEVER_BASE_URL=http://host.docker.internal:<RETRIEVER_DEPLOY_PORT>/v1`
(`48004` in the example). On Linux `host.docker.internal` is not defined by default, so the
`virag` service declares `extra_hosts: ["host.docker.internal:host-gateway"]` in
`deploy/service/docker-compose.yaml`. If the retriever server runs on another machine, point
`RETRIEVER_BASE_URL` at its address instead.

In `deploy/service/virag.env` also set `RETRIEVER_MODEL_NAME` to `SERVED_MODEL_NAME` in
`serve_retriever.sh` and `RETRIEVER_API_KEY` to `RETRIEVER_DEPLOY_API_KEY` in
`deploy/retriever/.env`; virag checks at startup that the server advertises that model and
refuses to boot otherwise.

### Connecting virag to the reranker stack (optional)

The same way, through the host: `RERANKER_BASE_URL=http://host.docker.internal:<RERANKER_DEPLOY_PORT>`
— the server **root**, without `/v1` (virag calls `/score` and `/v1/models` under it).
Set `RERANKER_MODEL_NAME` to `SERVED_MODEL_NAME` in `serve_reranker.sh` and
`RERANKER_API_KEY` to `RERANKER_DEPLOY_API_KEY` in `deploy/reranker/.env`. Leave
`RERANKER_BASE_URL` empty to run without one. The reranker is never a startup
dependency: one that is down or misconfigured is logged at startup, and every chat
falls back to the retriever's order until it answers.

vLLM's `--api-key` guards `/v1/*` only — `/score` and `/rerank` answer without the key.
Keep the reranker's port reachable from virag alone (`RERANKER_DEPLOY_HOST=127.0.0.1`,
or a firewall), not on a public interface.

`virag` `depends_on` the tiler, Qdrant and Valkey only; the retriever server is outside the stack,
so virag has `restart: unless-stopped` and retries until its dependency checks (including
the retriever endpoint) pass — this also covers the long vLLM model load.

## Configuration

Each container reads only its own file; all of them live in `deploy/service/` and are
copied from a tracked `.example` (the copies are git-ignored):

| File | Read by | Holds |
|---|---|---|
| `.env` | Compose only | `*_DEPLOY_*`: images, container names, published hosts/ports, volume paths. No container sees it. |
| `virag.env` | virag | `VIRAG_*`, `QDRANT_*`, `VALKEY_*`, `RETRIEVER_*`, `RERANKER_*`, `TILER_*` (how virag reaches the tiler), `VLM_*` — see [CONFIG.md](CONFIG.md). Required: `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`, `VLM_MODEL_NAME`, `VALKEY_URL`. |
| `tiler.env` | doc-tiler | the tiler's own `TILER_*` (key, geometry, pixel tier, pools, sweeper) — see doc-tiler's CONFIG.md. |
| `configs/valkey.conf` | Valkey | `requirepass`, AOF. |
| `configs/qdrant.yaml` | Qdrant | `service.api_key`, merged over the image's defaults. |

`TILER_*` means two things, and the split keeps them apart: in `virag.env` it is the
client side (`TILER_BASE_URL`, timeouts, `TILER_API_KEY` to present), in `tiler.env` the
server side.

The retriever and reranker stacks keep their own `.env` (`deploy/retriever/.env`,
`deploy/reranker/.env`), with the server's bearer key `RETRIEVER_DEPLOY_API_KEY` /
`RERANKER_DEPLOY_API_KEY` (passed in as `VLLM_API_KEY`; the healthcheck sends it too).
vLLM serving parameters are *not* in `.env`: they are written out, with a comment per
parameter, in `serve_retriever.sh` / `serve_reranker.sh` (copied from the tracked
`.example.sh`, git-ignored). Keep `SERVED_MODEL_NAME` there in sync with virag's
`RETRIEVER_MODEL_NAME` / `RERANKER_MODEL_NAME`. The scripts print the full `vllm serve`
command at start with the key masked.

### Secrets

Every `dummy` in the examples is a secret to replace (`openssl rand -hex 32`). A secret
shared by two services is written in both places:

| Secret | In the service | In `virag.env` |
|---|---|---|
| virag's user key | — | `VIRAG_API_KEY` (clients use it; OWUI's `VIRAG_API_KEY` valve) |
| virag's admin key | — | `VIRAG_ADMIN_API_KEY` (`misc/enroll_dir.py --api-key`) |
| tiler key | `TILER_API_KEY` in `tiler.env` | `TILER_API_KEY` |
| Qdrant key | `service.api_key` in `configs/qdrant.yaml` | `QDRANT_API_KEY` |
| Valkey password | `requirepass` in `configs/valkey.conf` | `VALKEY_URL=redis://:<password>@valkey-virag:6379/0` |
| retriever key | `RETRIEVER_DEPLOY_API_KEY` in `deploy/retriever/.env` | `RETRIEVER_API_KEY` |
| reranker key | `RERANKER_DEPLOY_API_KEY` in `deploy/reranker/.env` | `RERANKER_API_KEY` |

A copy that does not match stops virag at startup with a message naming the key. See
[SECURITY.md](SECURITY.md).

### Ports

Published by default (`.env`): virag `48000` on `0.0.0.0` — it guards itself with its keys,
and other containers on the host (e.g. Open WebUI) reach it via `host.docker.internal`,
which is not loopback; the tiler `48001` and Qdrant `48002` (REST, dashboard) / `48003`
(gRPC) on `127.0.0.1`, for tools on the host, each behind its own key. Valkey publishes no
port. The retriever and reranker stacks default to `48004` and `48005`. Put TLS (a reverse
proxy) in front of virag wherever its port is reachable beyond a trusted network.

## Tiler metadata schema

virag stamps enroll-lifecycle markers on the tiler's indexed metadata and reconcile
searches them, so the tiler must declare **three** indexed fields:
`name` (text), `virag_started_at` (number), and `virag_enrolled_at` (number). The Compose
file mounts `deploy/service/tiler_metadata_schema.yaml` into the tiler at
`/config/metadata_schema.yaml` for exactly this. At startup virag reads the tiler's
`GET /schema/metadata` and **refuses to boot** (fatal) if any of the three is missing or
wrong-typed — so keep this schema in sync with the tiler.

## Image

Single-stage `Dockerfile` on `python:3.11-slim`:

- Dependencies extracted from `pyproject.toml` with stdlib `tomllib`, so the pip layer
  caches on `pyproject.toml` alone.
- Runs as an unprivileged user (uid/gid **10001**); nothing runs as root.
- Listens on `:8080` (`uvicorn --host 0.0.0.0 --port 8080 --workers 1
  --timeout-graceful-shutdown 20`); the Compose port mapping exposes it on
  `VIRAG_DEPLOY_PORT`. The graceful timeout stays under the service's
  `stop_grace_period: 30s`, so `docker stop` always reaches the shutdown that hands back
  the lease, the revision job and the reconcile lead — instead of a SIGKILL leaving them
  to expire.
- `HEALTHCHECK` curls `/health` — not `/ready`, which stays `503` for as long as a
  rebuild after a tiler render change takes.

```bash
docker build -t virag:0.1.0 .
```

## Scaling

virag is stateless — everything it shares lives in the backends: the collections in
Qdrant, the coordination in Valkey (ARCHITECTURE.md §10). Processes are interchangeable,
so scale either way, or both:

- **uvicorn workers** (`--workers N`) — N processes on one port, the kernel spreading
  connections among them; the uvicorn parent replaces a worker that dies.
- **replicas** — more containers behind one endpoint. Compose cannot publish one host
  port for several replicas, so put a balancer (nginx, traefik) in front; in Kubernetes a
  `Service` does it.

Any process takes any request: a `POST /revision` runs where it lands, `GET /revision`
answers the same everywhere, one process at a time leads reconcile. The per-process
caps (`*_MAX_CONCURRENCY`) multiply with the process count — size them accordingly.

## Health & observability

- **`GET /health`** — liveness; always `200`, touches nothing. The container
  `HEALTHCHECK` and the right liveness probe.
- **`GET /ready`** — readiness; `503` with a `missing` list until every backend client
  is wired, then `200`. The right signal for a readiness probe / load-balancer gate.
- **Logs** — structured lines to stdout at `LOG_LEVEL` (default `INFO`). virag has no
  metrics endpoint; operational visibility is the log stream. Notable lines: the
  `ready: …` summary at startup (all backends + `render_version`), `lease: <token>
  (<host>/<pid>)` naming the process in Valkey, `taking over from dead process` when a
  revision job or the reconcile lead moves, `lease: … expired` (critical) when a process
  stops itself, `reranker: off` (warning) when none is configured, per-request `rag
  query` / `rag retrieved` / `rag reranked` / `rag injected` counts, `rag rerank failed`
  (a warning for an outage, an error for a refusal) when a chat fell back to the
  retriever's order, `integrity:` warnings when a tile's
  hash doesn't match its Qdrant payload, and `reconcile:` lines when a pass deletes
  dangling or quarantined docs' Qdrant points, clears stale-pending enrolls, or self-heals
  a doc.

## Startup ordering & the retriever model

The vLLM retriever server can take minutes to load a model; the tiler build (Chromium)
can be slow too. Because `virag` retries until its fatal dependency checks pass, bring
the stack up together and let it settle — `/ready` flips to `200` once the retriever, the
tiler, the VLM and Valkey all answer and the tiler↔Qdrant binding is satisfied. For the tiler's
own build (pinned Chromium layer, geometry guard) see
[`../../doc-tiler/docs/DEPLOYMENT.md`](../../doc-tiler/docs/DEPLOYMENT.md).

## Upgrading the tiler's renderer

A new Chromium / Pillow / PyMuPDF pin, DPI, JPEG quality or PDF slicing changes the
tiler's `render_version`. Stop virag, redeploy the tiler, then start virag:

1. The tiler restarts not serving and re-renders its docs. virag cannot start against
   it meanwhile — the tiler is as essential as the VLM; with `restart: unless-stopped`
   it keeps retrying until the tiler is ready.
2. virag starts, sees the collection was embedded from the old `render_version` and logs
   it. Chat answers `409 RENDER_VERSION_MISMATCH` until the collection is rebuilt:
   `POST /revision` (or set `VIRAG_REVISION_ON_RENDER_MISMATCH=true` to start it at
   startup). Follow it with `GET /revision`.
3. The swap stamps the new `render_version` on the collection; chat serves again.

A virag left running across the tiler's redeploy behaves the same: chat answers
`RENDER_VERSION_MISMATCH` once the tiler serves the new render, until `POST /revision`.
The downtime is the tiler's re-render plus virag's re-embed. A rollback to the
pre-upgrade collection is only accepted once the tiler is rolled back too
(`render_version` must match). Changing the retriever model is a different case: there
`POST /revision` runs with no downtime, the old collection serving until the swap.
