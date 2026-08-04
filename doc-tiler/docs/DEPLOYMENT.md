# Deployment

The Tiler ships as a **single container** with one named volume for state and one
mounted config file for the metadata schema. Everything below is driven from
[`deploy/`](../deploy/): a Compose file and an `.env.example` template.

---

## Quickstart (Compose)

```bash
# 0. (recommended) vendor Chromium first, so the build needs no CDN access — see below
python -m playwright install chromium-headless-shell
./scripts/vendor-chromium.sh

# 1. configure and start
cd deploy
cp .env.example .env          # set TILER_DB_SECRET and any geometry/pixel overrides
docker compose up --build     # builds the runtime image, starts the service
curl -s localhost:8080/readyz | jq
```

The service listens on **`:8080`** inside the container (`uvicorn --port 8080`), mapped
to the host by the Compose port binding. Interactive API docs: `http://localhost:8080/docs`.

Step 0 is optional — without it the build fetches Chromium from the Playwright CDN — but
it must run **before** the build, so do it first (see "Vendoring Chromium" below).

## Image

A two-stage build (`Dockerfile`):

- **`runtime`** (default) — the production image: FastAPI + the vendored
  `chromium-headless-shell`, deterministic fonts, no test dependencies. Dependencies are
  extracted from `pyproject.toml` with stdlib `tomllib` so the heavy pip layer caches on
  `pyproject.toml` alone. The build **fails** if Chromium can't actually launch, rather
  than deferring the failure to the first request.
- **`test`** — the same runtime plus the suite and its `dev` dependencies, for
  `docker compose --profile test run --rm tests`. Running the suite on the same image
  (same Chromium, same fonts) is what makes the rendering tests meaningful.

```bash
docker build --target runtime -t tiler:3.2.0 .
docker build --target test    -t tiler-test:3.2.0 .
```

Startup is wrapped in `tini` (PID 1, reaps Chromium's zombie renderers) →
`docker/entrypoint.sh`, which runs writability checks and the `python -m tiler.preflight`
geometry/secret guard **before** `exec`'ing uvicorn, so a misconfigured volume produces
one clear log line instead of a generic ASGI startup failure.

### Vendoring Chromium (run before the build)

The image ships only the lightweight `chromium-headless-shell` (what
`launch(headless=True)` uses on Playwright 1.49+ — the ~550 MB full bundle is dead weight
for a headless-only service). At build time the `Dockerfile` picks it up from one of two
places:

1. a **vendored copy** in `vendor/ms-playwright/` (if present), or
2. the **Playwright CDN**, downloaded with retries (the fallback).

For an air-gapped build — or just to stop a flaky CDN socket from failing `docker build`
— vendor it first, on a machine that *does* have network:

```bash
python -m playwright install chromium-headless-shell   # populate the local Playwright cache
./scripts/vendor-chromium.sh                           # copy that cache into ./vendor/ms-playwright
docker compose up --build                              # now the build consumes ./vendor, no CDN
```

- The vendored revision must match the `playwright` pin in `pyproject.toml`. It doesn't
  have to be checked by hand: the image runs a Chromium launch smoke test at build time,
  so a mismatch fails the **build**, not production.
- `scripts/vendor-chromium.sh [source-cache-dir]` defaults its source to
  `$PLAYWRIGHT_BROWSERS_PATH` or `~/.cache/ms-playwright`.
- `vendor/*` is gitignored (only `.keep` is tracked) — it is a per-machine build
  artifact, produced *before* the build, never committed.

## State & the named volume

All mutable state lives under `/data` (`TILER_DB_PATH`, `TILER_BLOB_DIR`,
`TILER_TMP_DIR`), backed by one volume. The metadata schema is mounted read-only at
`/config/metadata_schema.yaml`.

**Geometry and the binding secret are immutable per volume.** Booting an existing volume
with a different geometry — or a different `TILER_DB_SECRET` — is a fatal startup error
(see [ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md)). To change
either, deploy against a fresh volume and re-ingest.

## Running as a non-root user

Nothing in the container runs as root — blobs are written by uid **10001**.

- The uid/gid `10001` is baked in as a **hard constant** (an `ENV`, not a build `ARG`),
  and the entrypoint refuses to start under any other uid — defeating a `user:` /
  `--user` override, the one lever the image itself can't take away.
- `/data` is pre-created in the image *owned by that user*, so Docker seeds an empty
  named volume with correct ownership and it comes up writable with no privileged
  `chown` at runtime.
- The entrypoint checks writability **as the runtime user** before anything starts and,
  if the volume is root-owned (e.g. a bind mount created earlier by a root container),
  fails with the exact command to fix it.
- Compose sets `no-new-privileges`. Chromium runs with Playwright's sandbox disabled
  (standard for containers) — which is exactly why a non-root uid plus `no-new-privileges`
  matter.

Bind-mounting a host directory instead of a named volume? Pre-create it:

```bash
mkdir -p ./data && sudo chown -R 10001:10001 ./data
```

## Health & scaling

- **Liveness:** `GET /healthz` (touches nothing, stays fast under load).
- **Readiness:** `GET /readyz` — checks DB, blob dir, Chromium pool; `503` when not
  ready. The Compose healthcheck uses it.
- Run **one** uvicorn worker per container (`--workers 1`). Concurrency comes from the
  event loop plus the browser/CPU pools, not from workers — multiple workers would each
  spawn their own Chromium pool and SQLite writer and fight over the volume. **Scale
  horizontally with more containers**, sharing the blob store and DB behind their
  interfaces.

See [CONFIG.md](CONFIG.md) for every `TILER_*` variable and [METRICS.md](METRICS.md) for
the Prometheus endpoint.
