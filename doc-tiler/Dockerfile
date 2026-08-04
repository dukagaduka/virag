# syntax=docker/dockerfile:1
#
# Two targets:
#   --target runtime  (default for docker compose) — production image, no test deps
#   --target test     — adds pytest + the suite, for `docker compose run tests`
#
FROM python:3.11-slim AS runtime

# ---------------------------------------------------------------------------
# System deps: Chromium's runtime libraries + deterministic fonts (see
# docs/ARCHITECTURE.md, "Rendering determinism").
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-liberation \
      libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
      libcups2 libdrm2 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
      libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
      libxext6 libx11-6 libxcb1 libexpat1 libudev1 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

ENV TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# ---------------------------------------------------------------------------
# Unprivileged runtime user. Blobs written by root are a permanent operational
# headache on a bind mount, so nothing here ever runs as root.
#
# The uid/gid is baked in as a hard constant — NOT a build ARG — so it cannot be
# shifted with `--build-arg`. It is also exported as an ENV so the entrypoint can
# refuse to start under any other uid (defeats a `user:`/`--user` override at
# runtime, the one lever the image itself cannot take away). Do not turn this
# back into an ARG.
# ---------------------------------------------------------------------------
ENV APP_UID=10001 \
    APP_GID=10001
RUN groupadd --gid 10001 tiler \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin tiler

WORKDIR /app

# pyproject.toml is the single source of truth for dependencies (built with
# hatchling). We extract just the runtime dependency list with stdlib tomllib and
# pip-install it, so this heavy layer is cached on pyproject.toml alone and does
# not depend on the application source copied later.
COPY pyproject.toml .
RUN python -c "import tomllib; \
open('/tmp/reqs.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
 && pip install --no-cache-dir -r /tmp/reqs.txt \
 && rm /tmp/reqs.txt

# We run Chromium headless only, so we ship just the lightweight headless shell
# (launch(headless=True) uses it on Playwright 1.49+, not the full browser). It
# comes either from a vendored copy in the build context (offline / air-gapped /
# unreachable-CDN builds — see scripts/vendor-chromium.sh) or from the CDN with
# retries, since a single timed-out socket should not fail the build. `vendor/` is
# normally empty apart from .keep.
COPY vendor/ /tmp/vendor/
RUN set -eu; \
    if ls /tmp/vendor/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell >/dev/null 2>&1; then \
      echo "using vendored headless shell from the build context"; \
      mkdir -p ${PLAYWRIGHT_BROWSERS_PATH}; \
      cp -a /tmp/vendor/ms-playwright/. ${PLAYWRIGHT_BROWSERS_PATH}/; \
    else \
      for attempt in 1 2 3 4 5; do \
        python -m playwright install chromium-headless-shell && break; \
        echo "playwright install failed (attempt ${attempt}); retrying..."; \
        sleep 15; \
      done; \
    fi; \
    rm -rf /tmp/vendor; \
    ls ${PLAYWRIGHT_BROWSERS_PATH}/chromium_headless_shell-*/chrome-linux/headless_shell >/dev/null 2>&1 \
      || { echo "FATAL: Chromium headless shell was not installed"; exit 1; }; \
    chmod -R a+rX ${PLAYWRIGHT_BROWSERS_PATH}

# Fail the build — not the first request — if the browser cannot actually launch
# (missing system library, wrong revision, unreadable files).
RUN python -c "\
from playwright.sync_api import sync_playwright;\
p = sync_playwright().start();\
b = p.chromium.launch(headless=True);\
print('chromium ok:', b.version);\
b.close(); p.stop()"

COPY --chown=${APP_UID}:${APP_GID} tiler/ ./tiler/
COPY --chown=${APP_UID}:${APP_GID} docker/entrypoint.sh /usr/local/bin/tiler-entrypoint
COPY --chown=${APP_UID}:${APP_GID} deploy/metadata_schema.yaml /config/metadata_schema.yaml
RUN chmod +x /usr/local/bin/tiler-entrypoint

# /data is pre-created *owned by the app user*: Docker seeds an empty named volume
# from the image path, ownership included, so the volume comes up writable without
# any privileged chown at runtime.
RUN mkdir -p /data /data/blobs /data/tmp \
 && chown -R ${APP_UID}:${APP_GID} /data /config /app

ENV TILER_DB_PATH=/data/meta.sqlite \
    TILER_BLOB_DIR=/data/blobs \
    TILER_TMP_DIR=/data/tmp \
    TILER_METADATA_SCHEMA_PATH=/config/metadata_schema.yaml \
    HOME=/home/tiler

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/readyz || exit 1

# tini reaps the zombie Chromium/renderer processes a PID-1 Python would leak.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/tiler-entrypoint"]
CMD ["uvicorn", "tiler.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]


# ---------------------------------------------------------------------------
# Test image: the same runtime (same Chromium, same fonts — so rendering tests
# mean something) plus the suite and its dev dependencies.
# ---------------------------------------------------------------------------
FROM runtime AS test

USER root
# Dev/test dependencies come from pyproject's optional-dependencies.dev group
# (pyproject.toml is already at /app from the runtime stage).
RUN python -c "import tomllib; \
open('/tmp/dev.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['dev']))" \
 && pip install --no-cache-dir -r /tmp/dev.txt \
 && rm /tmp/dev.txt
COPY --chown=10001:10001 tests/ ./tests/
# pytest config lives in pyproject.toml ([tool.pytest.ini_options]), already at /app.
USER 10001:10001
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "-q"]
