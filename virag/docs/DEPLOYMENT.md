# Deployment

virag is a small stateless gateway; the interesting part of deploying it is the
**backends it needs**: doc-tiler, Qdrant, an embeddings server, and an upstream VLM.
The Compose file in [`deploy/`](../deploy/) brings up the first three plus virag itself;
the VLM is external (you point `VLM_BASE_URL` at it).

---

## Quickstart (Compose)

```bash
cd deploy
cp .env.example .env     # set TILER_DB_SECRET, VLM_BASE_URL, VLM_MODEL_NAME, QDRANT_VECTOR_SIZE, paths
docker compose up --build
curl -s localhost:${VIRAG_DEPLOY_PORT:-48000}/ready | jq
```

The stack (`deploy/docker-compose.yaml`) is four services on one bridge network:

| Service | Role | Notes |
|---|---|---|
| `virag` | the gateway | built from this repo; waits for its backends to be healthy. |
| `tiler-virag` | doc-tiler | built from `../../doc-tiler`; mounts `tiler_metadata_schema.yaml` (see below). |
| `qdrant-virag` | Qdrant | vector store. |
| `embeddings-vllm-virag` | vLLM embeddings | GPU (`runtime: nvidia`); runs `serve_vllm_embeddings.sh`. |

`virag` `depends_on` the tiler and Qdrant being **healthy** and the embeddings server
being **started**, and has `restart: unless-stopped` so it survives the long vLLM model
load and retries until its dependency checks pass.

## Configuration

Two layers of variables live in `.env` (from [`.env.example`](../deploy/.env.example)):

- **Application settings** read by virag at startup — `VIRAG_*` (including the reconcile
  knobs `VIRAG_SELF_HEAL` / `VIRAG_RECONCILE_INTERVAL_SEC` / `VIRAG_PENDING_TTL_SEC`),
  `QDRANT_*` (including the upsert-batching `QDRANT_CHUNK_SIZE` / `QDRANT_MAX_BATCH_CHUNKS`),
  `EMBEDDINGS_*`/`VLLM_*`, `TILER_*`, `VLM_*`, `TILER_DB_SECRET`. See
  [CONFIG.md](CONFIG.md). Required: `QDRANT_VECTOR_SIZE`, `VLM_BASE_URL`,
  `VLM_MODEL_NAME`.
- **Compose/orchestration variables** — the `*_DEPLOY_*` image tags, host bind
  addresses/ports, and volume paths, plus the `VLLM_*` serving knobs consumed by
  `serve_vllm_embeddings.sh`. These are read by Compose and the launch script, not by
  `Settings.from_env()`.

Set `TILER_DB_SECRET` to the **same** value it has on the tiler and (implicitly) the
Qdrant collection — one secret, three consumers — *before* first boot. See
[SECURITY.md](SECURITY.md).

## Tiler metadata schema

virag stamps enroll-lifecycle markers on the tiler's indexed metadata and reconcile
searches them, so the tiler must declare **three** indexed fields:
`name` (text), `virag_started_at` (number), and `virag_enrolled_at` (number). The Compose
file mounts `deploy/tiler_metadata_schema.yaml` into the tiler at
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
