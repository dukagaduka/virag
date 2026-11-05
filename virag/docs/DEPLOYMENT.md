# Deployment

virag is a small stateless gateway; the interesting part of deploying it is the
**backends it needs**: doc-tiler, Qdrant, Valkey, a retriever server, and an upstream VLM.
Deployment is split into two independent Compose stacks under [`deploy/`](../deploy/):
[`service/`](../deploy/service/) (virag, doc-tiler, Qdrant, Valkey) and
[`retriever/`](../deploy/retriever/) (the vLLM retriever server). The VLM is external
(you point `VLM_BASE_URL` at it).

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
cp .env.example .env     # set TILER_DB_SECRET, VLM_BASE_URL, VLM_MODEL_NAME, QDRANT_VECTOR_SIZE,
                         # RETRIEVER_BASE_URL, paths
docker compose up --build
curl -s localhost:${VIRAG_DEPLOY_PORT:-48000}/ready | jq
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
(`48003` in the example). On Linux `host.docker.internal` is not defined by default, so the
`virag` service declares `extra_hosts: ["host.docker.internal:host-gateway"]` in
`deploy/service/docker-compose.yaml`. If the retriever server runs on another machine, point
`RETRIEVER_BASE_URL` at its address instead.

In `deploy/service/.env` also set `RETRIEVER_MODEL_NAME` to `SERVED_MODEL_NAME` in
`serve_retriever.sh` and `RETRIEVER_API_KEY` to `RETRIEVER_DEPLOY_API_KEY` in
`deploy/retriever/.env`; virag checks at startup that the server advertises that model and
refuses to boot otherwise.

`virag` `depends_on` the tiler, Qdrant and Valkey only; the retriever server is outside the stack,
so virag has `restart: unless-stopped` and retries until its dependency checks (including
the retriever endpoint) pass — this also covers the long vLLM model load.

## Configuration

Variables live in three places:

- **Application settings** read by virag at startup — `VIRAG_*` (including the reconcile
  knobs `VIRAG_SELF_HEAL` / `VIRAG_RECONCILE_INTERVAL_SEC` / `VIRAG_PENDING_TTL_SEC`),
  `QDRANT_*` (including the upsert-batching `QDRANT_CHUNK_SIZE` / `QDRANT_MAX_BATCH_CHUNKS`),
  `VALKEY_*`, `RETRIEVER_*`, `TILER_*`, `VLM_*`, `TILER_DB_SECRET`. See
  [CONFIG.md](CONFIG.md). Required: `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`,
  `VLM_MODEL_NAME`, `VALKEY_URL`.
- **Compose/orchestration variables** — the `*_DEPLOY_*` image tags, host bind
  addresses/ports, and volume paths. Read by Compose, not by `Settings.from_env()`.
  `deploy/service/.env` (from [`.env.example`](../deploy/service/.env.example)) holds the
  application settings above together with the service-stack deploy variables;
  `deploy/retriever/.env` (from [`.env.example`](../deploy/retriever/.env.example)) holds
  the retriever deploy variables, among them the server's bearer key
  `RETRIEVER_DEPLOY_API_KEY` (passed in as `VLLM_API_KEY`; the healthcheck sends it too).
- **vLLM serving parameters** — *not* in `.env`. They are written out, with a comment per
  parameter, directly in `deploy/retriever/serve_retriever.sh`. The tracked
  [`serve_retriever.example.sh`](../deploy/retriever/serve_retriever.example.sh) holds
  baseline values; copy it to `serve_retriever.sh` (git-ignored) and tune it for your
  GPU. Keep `SERVED_MODEL_NAME` there in sync with virag's `RETRIEVER_MODEL_NAME`.

Set `TILER_DB_SECRET` to the **same** value it has on the tiler and (implicitly) the
Qdrant collection — one secret, three consumers — *before* first boot. See
[SECURITY.md](SECURITY.md).

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
  stops itself, per-request `rag
  query` / `rag retrieved` / `rag injected` counts, `integrity:` warnings when a tile's
  hash doesn't match its Qdrant payload, and `reconcile:` lines when a pass deletes
  dangling or quarantined docs' Qdrant points, clears stale-pending enrolls, or self-heals
  a doc.

## Startup ordering & the retriever model

The vLLM retriever server can take minutes to load a model; the tiler build (Chromium)
can be slow too. Because `virag` retries until its fatal dependency checks pass, bring
the stack up together and let it settle — `/ready` flips to `200` once the retriever, the
tiler, the VLM and Valkey all answer and the tiler↔Qdrant binding is satisfied. For the tiler's
own build (pinned Chromium layer, geometry/secret guard) see
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
