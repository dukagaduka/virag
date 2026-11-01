# Deployment

virag is a small stateless gateway; the interesting part of deploying it is the
**backends it needs**: doc-tiler, Qdrant, a retriever server, and an upstream VLM.
Deployment is split into two independent Compose stacks under [`deploy/`](../deploy/):
[`service/`](../deploy/service/) (virag, doc-tiler, Qdrant) and
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

# 2. virag + tiler + Qdrant
cd ../service
cp .env.example .env     # set TILER_DB_SECRET, VLM_BASE_URL, VLM_MODEL_NAME, QDRANT_VECTOR_SIZE,
                         # RETRIEVER_BASE_URL, paths
docker compose up --build
curl -s localhost:${VIRAG_DEPLOY_PORT:-48000}/ready | jq
```

The `service/` stack (`deploy/service/docker-compose.yaml`) is three services on one bridge
network. The retriever server (`retriever-vllm-virag`, GPU, runs `serve_retriever.sh`) is a
separate stack in `deploy/retriever/` that virag reaches over `RETRIEVER_BASE_URL`:

| Service | Role | Notes |
|---|---|---|
| `virag` | the gateway | built from this repo; waits for its backends to be healthy. |
| `tiler-virag` | doc-tiler | built from `../../doc-tiler`; mounts `tiler_metadata_schema.yaml` (see below). |
| `qdrant-virag` | Qdrant | vector store. |

### Connecting virag to the retriever stack

The two stacks do not share a Docker network. virag reaches the retriever server through
the host: set `RETRIEVER_BASE_URL=http://host.docker.internal:<RETRIEVER_DEPLOY_PORT>/v1`
(`48003` in the example). On Linux `host.docker.internal` is not defined by default, so the
`virag` service declares `extra_hosts: ["host.docker.internal:host-gateway"]` in
`deploy/service/docker-compose.yaml`. If the retriever server runs on another machine, point
`RETRIEVER_BASE_URL` at its address instead.

In `deploy/service/.env` also set `RETRIEVER_MODEL_NAME` and `RETRIEVER_API_KEY` to the values
of `SERVED_MODEL_NAME` and `API_KEY` in `serve_retriever.sh`; virag checks at startup that
the server advertises that model and refuses to boot otherwise.

`virag` `depends_on` the tiler and Qdrant only; the retriever server is outside the stack,
so virag has `restart: unless-stopped` and retries until its dependency checks (including
the retriever endpoint) pass — this also covers the long vLLM model load.

## Configuration

Variables live in three places:

- **Application settings** read by virag at startup — `VIRAG_*` (including the reconcile
  knobs `VIRAG_SELF_HEAL` / `VIRAG_RECONCILE_INTERVAL_SEC` / `VIRAG_PENDING_TTL_SEC`),
  `QDRANT_*` (including the upsert-batching `QDRANT_CHUNK_SIZE` / `QDRANT_MAX_BATCH_CHUNKS`),
  `RETRIEVER_*`, `TILER_*`, `VLM_*`, `TILER_DB_SECRET`. See
  [CONFIG.md](CONFIG.md). Required: `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`,
  `VLM_MODEL_NAME`.
- **Compose/orchestration variables** — the `*_DEPLOY_*` image tags, host bind
  addresses/ports, and volume paths. Read by Compose, not by `Settings.from_env()`.
  `deploy/service/.env` (from [`.env.example`](../deploy/service/.env.example)) holds the
  application settings above together with the service-stack deploy variables;
  `deploy/retriever/.env` (from [`.env.example`](../deploy/retriever/.env.example)) holds
  only the retriever deploy variables.
- **vLLM serving parameters** — *not* in `.env`. They are written out, with a comment per
  parameter, directly in `deploy/retriever/serve_retriever.sh`. The tracked
  [`serve_retriever.example.sh`](../deploy/retriever/serve_retriever.example.sh) holds
  baseline values; copy it to `serve_retriever.sh` (git-ignored) and tune it for your
  GPU. Keep `SERVED_MODEL_NAME` and `API_KEY` there in sync with virag's
  `RETRIEVER_MODEL_NAME` and `RETRIEVER_API_KEY`.

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
- Listens on `:8080` (`uvicorn --host 0.0.0.0 --port 8080 --workers 1`); the Compose
  port mapping exposes it on `VIRAG_DEPLOY_PORT`.
- `HEALTHCHECK` curls `/health` — not `/ready`, which stays `503` for as long as a
  rebuild after a tiler render change takes.

```bash
docker build -t virag:0.1.0 .
```

Run **one** uvicorn worker per container. virag is stateless (all state is in the
backends), so scale horizontally with more containers behind a load balancer if needed.

## Health & observability

- **`GET /health`** — liveness; always `200`, touches nothing. The container
  `HEALTHCHECK` and the right liveness probe.
- **`GET /ready`** — readiness; `503` with a `missing` list until every backend client
  is wired, then `200`. The right signal for a readiness probe / load-balancer gate.
- **Logs** — structured lines to stdout at `LOG_LEVEL` (default `INFO`). virag has no
  metrics endpoint; operational visibility is the log stream. Notable lines: the
  `ready: …` summary at startup (all backends + `render_version`), per-request `rag
  query` / `rag retrieved` / `rag injected` counts, `integrity:` warnings when a tile's
  hash doesn't match its Qdrant payload, and `reconcile:` lines when a pass deletes
  dangling or quarantined docs' Qdrant points, clears stale-pending enrolls, or self-heals
  a doc.

## Startup ordering & the retriever model

The vLLM retriever server can take minutes to load a model; the tiler build (Chromium)
can be slow too. Because `virag` retries until its fatal dependency checks pass, bring
the stack up together and let it settle — `/ready` flips to `200` once the retriever, the
tiler and the VLM all answer and the tiler↔Qdrant binding is satisfied. For the tiler's
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
