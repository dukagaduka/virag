# Deployment

The Tiler ships as a **single container** with one named volume for state and one
mounted config file for the metadata schema. Everything below is driven from
[`deploy/`](../deploy/): a Compose file and an `.env.example` template.

---

## Quickstart (Compose)

```bash
cd deploy
cp .env.example .env          # set TILER_DB_SECRET and any geometry/pixel overrides
docker compose up --build     # builds the runtime image, starts the service
curl -s localhost:8080/readyz | jq
```

The service listens on **`:8080`** inside the container (`uvicorn --port 8080`), mapped
to the host by the Compose port binding. Interactive API docs: `http://localhost:8080/docs`.

The build needs BuildKit (the `Dockerfile` uses cache mounts), which Docker runs through
the `buildx` plugin — install `docker-buildx-plugin` if `docker build` reports it
missing.

## Image

A two-stage build (`Dockerfile`):

- **`runtime`** (default) — the production image: FastAPI + Playwright's
  `chromium-headless-shell`, deterministic fonts, no test dependencies. The build
  **fails** if Chromium can't actually launch, rather than deferring the failure to the
  first request.
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

### Layers and what is pinned

The layers run from the most stable to the most volatile, so a rebuild only redoes what
changed:

| Layer | Pinned by | Rebuilt when |
|---|---|---|
| Base image | `BASE_IMAGE` digest in the `Dockerfile` | the digest is bumped |
| System packages (fonts, freetype, Chromium's libraries) | the dated `snapshot.debian.org` archive the base image was built from | the base digest is bumped |
| Playwright + `chromium-headless-shell` | the `playwright==X.Y.Z` pin in `pyproject.toml` | the pin changes |
| Remaining Python dependencies | their pins in `pyproject.toml` | `pyproject.toml` changes |
| Application | — | the source changes |

- **Chromium is downloaded once per machine and pin** (from the Playwright CDN, with
  retries) and reused from the layer cache afterwards: editing another dependency or the
  code does not touch it. A small build stage cuts the `playwright` line out of
  `pyproject.toml`, and only that line feeds the browser layer, so `pyproject.toml` stays
  the single source of the version.
- The image ships only the lightweight headless shell (what `launch(headless=True)` uses on
  Playwright 1.49+); the ~550 MB full browser is dead weight for a headless-only service.
- **System packages come from a dated Debian snapshot**, not the live mirrors: the same
  base digest gives byte-identical fonts and libraries on any machine at any later date.
  `--build-arg APT_SNAPSHOT=false` installs from the live mirrors instead (not
  reproducible — use only if snapshot.debian.org is unreachable).
- **To ship the image to a machine without CDN access**, push it to a registry, or
  `docker save tiler:3.2.0 | ssh host docker load`.

### Upgrading Chromium, fonts or the base

Every pinned layer above affects rendered pixels — the Chromium revision through
`render_version`, the fonts and libraries silently. Upgrading is a deliberate act:

1. Bump the `playwright` pin in `pyproject.toml` and/or the `BASE_IMAGE` digest.
2. Rebuild and redeploy.
3. Run the tiler's `POST /revision` (re-renders docs whose `render_version` changed; use
   `force=true` after a base-only bump, which `render_version` does not see), then the
   consumer's re-embed (virag's `POST /revision`).

Until step 3 finishes, regenerated web/MHTML tiles may not match their stored
`image_hash` (see `TILER_RENDER_MISMATCH_POLICY` in [CONFIG.md](CONFIG.md)).

## State & the named volume

All mutable state lives under `/data` (`TILER_DB_PATH`, `TILER_BLOB_DIR`,
`TILER_TMP_DIR`), backed by one volume. The metadata schema is mounted read-only at
`/config/metadata_schema.yaml`.

**Geometry and the binding secret are immutable per volume.** Booting an existing volume
with a different geometry — or a different `TILER_DB_SECRET` — is a fatal startup error
(see [ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md)). To change
either, deploy against a fresh volume and re-ingest.

**Reclaiming crashed streaming ingests.** `POST /tile/stream` persists a doc incrementally
and leaves it incomplete if the stream dies. Set `TILER_INCOMPLETE_TTL_SEC` (default `0` =
disabled) to have a background sweeper delete incomplete docs older than that TTL — set it
**above your longest render** so an in-flight ingest is never reaped — running every
`TILER_SWEEP_INTERVAL_SEC` (default `300`). See [CONFIG.md](CONFIG.md) and
[ARCHITECTURE.md](ARCHITECTURE.md).

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
  event loop plus the browser/render pools, not from workers — multiple workers would each
  spawn their own Chromium pool and SQLite writer and fight over the volume. **Scale
  horizontally with more containers**, sharing the blob store and DB behind their
  interfaces.

See [CONFIG.md](CONFIG.md) for every `TILER_*` variable and [METRICS.md](METRICS.md) for
the Prometheus endpoint.
