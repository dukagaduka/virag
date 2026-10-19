# Deployment

virag is a small stateless gateway; the interesting part of deploying it is the
**backends it needs**: doc-tiler, Qdrant, an embeddings server, and an upstream VLM.
Deployment is split into two independent Compose stacks under [`deploy/`](../deploy/):
[`service/`](../deploy/service/) (virag, doc-tiler, Qdrant) and
[`embeddings/`](../deploy/embeddings/) (the vLLM embeddings server). The VLM is external
(you point `VLM_BASE_URL` at it).

---

## Quickstart (Compose)

```bash
# 1. embeddings server (GPU)
cd deploy/embeddings
cp .env.example .env                                   # image, port, model/adapter paths
cp serve_embeddings.example.sh serve_embeddings.sh     # tune for your hardware
docker compose up -d

# 2. virag + tiler + Qdrant
cd ../service
cp .env.example .env     # set TILER_DB_SECRET, VLM_BASE_URL, VLM_MODEL_NAME, QDRANT_VECTOR_SIZE,
                         # EMBEDDINGS_BASE_URL, paths
docker compose up --build
curl -s localhost:${VIRAG_DEPLOY_PORT:-48000}/ready | jq
```

The `service/` stack (`deploy/service/docker-compose.yaml`) is three services on one bridge
network. The embeddings server (`embeddings-vllm-virag`, GPU, runs `serve_embeddings.sh`) is a
separate stack in `deploy/embeddings/` that virag reaches over `EMBEDDINGS_BASE_URL`:

| Service | Role | Notes |
|---|---|---|
| `virag` | the gateway | built from this repo; waits for its backends to be healthy. |
| `tiler-virag` | doc-tiler | built from `../../doc-tiler`; mounts `tiler_metadata_schema.yaml` (see below). |
| `qdrant-virag` | Qdrant | vector store. |

### Connecting virag to the embeddings stack

The two stacks do not share a Docker network. virag reaches the embeddings server through
the host: set `EMBEDDINGS_BASE_URL=http://host.docker.internal:<VLLM_DEPLOY_PORT>/v1`
(`48003` in the example). On Linux `host.docker.internal` is not defined by default, so the
`virag` service declares `extra_hosts: ["host.docker.internal:host-gateway"]` in
`deploy/service/docker-compose.yaml`. If the embeddings server runs on another machine, point
`EMBEDDINGS_BASE_URL` at its address instead.

In `deploy/service/.env` also set `VLLM_SERVED_MODEL_NAME` and `VLLM_API_KEY` to the values
of `SERVED_MODEL_NAME` and `API_KEY` in `serve_embeddings.sh`; virag checks at startup that
the server advertises that model and refuses to boot otherwise.

`virag` `depends_on` the tiler and Qdrant only; the embeddings server is outside the stack,
so virag has `restart: unless-stopped` and retries until its dependency checks (including
the embeddings endpoint) pass — this also covers the long vLLM model load.

## Configuration

Variables live in three places:

- **Application settings** read by virag at startup — `VIRAG_*` (including the reconcile
  knobs `VIRAG_SELF_HEAL` / `VIRAG_RECONCILE_INTERVAL_SEC` / `VIRAG_PENDING_TTL_SEC`),
  `QDRANT_*` (including the upsert-batching `QDRANT_CHUNK_SIZE` / `QDRANT_MAX_BATCH_CHUNKS`),
  `EMBEDDINGS_*`/`VLLM_*`, `TILER_*`, `VLM_*`, `TILER_DB_SECRET`. See
  [CONFIG.md](CONFIG.md). Required: `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`,
  `VLM_MODEL_NAME`.
- **Compose/orchestration variables** — the `*_DEPLOY_*` image tags, host bind
  addresses/ports, and volume paths. Read by Compose, not by `Settings.from_env()`.
  `deploy/service/.env` (from [`.env.example`](../deploy/service/.env.example)) holds the
  application settings above together with the service-stack deploy variables;
  `deploy/embeddings/.env` (from [`.env.example`](../deploy/embeddings/.env.example)) holds
  only the embeddings deploy variables.
- **vLLM serving parameters** — *not* in `.env`. They are written out, with a comment per
  parameter, directly in `deploy/embeddings/serve_embeddings.sh`. The tracked
  [`serve_embeddings.example.sh`](../deploy/embeddings/serve_embeddings.example.sh) holds
  baseline values; copy it to `serve_embeddings.sh` (git-ignored) and tune it for your
  GPU. Keep `SERVED_MODEL_NAME` and `API_KEY` there in sync with virag's
  `VLLM_SERVED_MODEL_NAME` and `VLLM_API_KEY`.

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
- `HEALTHCHECK` curls `/ready`.

```bash
docker build -t virag:0.1.0 .
```

Run **one** uvicorn worker per container. virag is stateless (all state is in the
backends), so scale horizontally with more containers behind a load balancer if needed.

## Health & observability

- **`GET /health`** — liveness; always `200`, touches nothing. Use for a liveness
  probe.
- **`GET /ready`** — readiness; `503` with a `missing` list until every backend client
  is wired, then `200`. This is the container `HEALTHCHECK` and the right signal for a
  readiness probe / load-balancer gate.
- **Logs** — structured lines to stdout at `LOG_LEVEL` (default `INFO`). virag has no
  metrics endpoint; operational visibility is the log stream. Notable lines: the
  `ready: …` summary at startup (all backends + `render_version`), per-request `rag
  query` / `rag retrieved` / `rag injected` counts, `integrity:` warnings when a tile's
  hash doesn't match its Qdrant payload, and `reconcile:` lines when a pass deletes
  dangling Qdrant points, clears stale-pending enrolls, or self-heals a doc.

## Startup ordering & the embeddings model

The vLLM embeddings server can take minutes to load a model; the tiler build (Chromium)
can be slow too. Because `virag` retries until its fatal dependency checks pass, bring
the stack up together and let it settle — `/ready` flips to `200` once embeddings, the
tiler and the VLM all answer and the tiler↔Qdrant binding is satisfied. For the tiler's
own build (vendoring Chromium, geometry/secret guard) see
[`../../doc-tiler/docs/DEPLOYMENT.md`](../../doc-tiler/docs/DEPLOYMENT.md).
