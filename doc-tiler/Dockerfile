# syntax=docker/dockerfile:1
#
# Two targets:
#   --target runtime  (default for docker compose) — production image, no test deps
#   --target test     — adds pytest + the suite, for `docker compose run tests`
#
# Layers are ordered from the most stable to the most volatile, so a rebuild only
# redoes what actually changed:
#
#   pinned base → system packages from a dated Debian snapshot → Playwright +
#   Chromium → the remaining Python deps → the application
#
# Everything that affects rendered pixels sits in the first three and is pinned:
# the base by digest, the system packages (fonts, freetype, …) by the snapshot the
# base was built from, Chromium by the `playwright` pin in pyproject.toml. Bumping
# any of them is a deliberate act: the next start revises the stored docs before it
# serves (a base-only bump needs `POST /revision?force=true`; docs/DEPLOYMENT.md).

# python:3.11-slim (Debian 13). Bump the digest deliberately: it moves the OS and,
# through the snapshot below, the fonts and system libraries too.
ARG BASE_IMAGE=python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60


# ---------------------------------------------------------------------------
# The `playwright` pin, cut out of pyproject.toml into a file of its own. The
# runtime stage copies only this file, and a COPY is cached on content, so the
# browser layer is rebuilt when the pin changes — not on every pyproject edit.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS playwright-pin

COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "import re, tomllib; \
deps = tomllib.load(open('/tmp/pyproject.toml', 'rb'))['project']['dependencies']; \
pins = [d for d in deps if re.match(r'playwright\s*==', d)]; \
assert len(pins) == 1, 'pyproject.toml must pin playwright exactly once (playwright==X.Y.Z)'; \
open('/playwright.txt', 'w').write(pins[0] + chr(10))"


FROM ${BASE_IMAGE} AS runtime

# ---------------------------------------------------------------------------
# System deps: Chromium's runtime libraries + deterministic fonts (see
# docs/ARCHITECTURE.md, "Rendering determinism").
#
# apt reads the dated snapshot.debian.org archive the base image was built from
# (its sources carry that URL as a comment), so the same base digest always gets
# the same fonts and libraries — on any machine, at any later date. Set
# APT_SNAPSHOT=false to install from the live mirrors instead (not reproducible).
#
# apt's package lists and downloaded .debs live in BuildKit cache mounts — outside
# the image, kept between builds — so a rebuild does not re-download them. The
# slim image's docker-clean hook (which deletes .debs after install) is disabled
# for that reason; nothing of the cache ends up in a layer.
# ---------------------------------------------------------------------------
ARG APT_SNAPSHOT=true
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eu; \
    if [ "${APT_SNAPSHOT}" = "true" ]; then \
      sed -i -E '/^# http:\/\/snapshot\.debian\.org\//{N;s|^# (http://snapshot[^\n]*)\nURIs: [^\n]*|URIs: \1|}' \
        /etc/apt/sources.list.d/debian.sources; \
      grep -q '^URIs: http://snapshot.debian.org/' /etc/apt/sources.list.d/debian.sources \
        || { echo "base image carries no snapshot URL; build with APT_SNAPSHOT=false" >&2; exit 1; }; \
    fi; \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache; \
    apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      fonts-noto fonts-noto-cjk fonts-noto-color-emoji fonts-liberation \
      libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
      libcups2 libdrm2 libxkbcommon0 libatspi2.0-0 libxcomposite1 libxdamage1 \
      libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2 \
      libxext6 libx11-6 libxcb1 libexpat1 libudev1 curl tini; \
    fc-cache -f

ENV TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
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

# ---------------------------------------------------------------------------
# Playwright + its Chromium, in a layer of their own. We run Chromium headless
# only, so we ship just the lightweight headless shell (launch(headless=True) uses
# it on Playwright 1.49+, not the full browser). The download happens once per
# machine and pin; any other change reuses this layer. To move it to another
# machine, push the image to a registry (or `docker save | docker load`).
#
# pip is upgraded first for --resume-retries: on a flaky link a dropped download
# resumes instead of failing the whole build. The browser download is retried too.
# ---------------------------------------------------------------------------
COPY --from=playwright-pin /playwright.txt /tmp/playwright.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    pip install --upgrade "pip>=25.1"; \
    pip install --retries 10 --timeout 120 --resume-retries 10 -r /tmp/playwright.txt; \
    rm /tmp/playwright.txt; \
    ok=0; \
    for attempt in 1 2 3 4 5; do \
      if python -m playwright install chromium-headless-shell; then ok=1; break; fi; \
      echo "playwright install failed (attempt ${attempt}); retrying..."; \
      sleep 15; \
    done; \
    [ "${ok}" = 1 ] || { echo "playwright install failed after 5 attempts" >&2; exit 1; }; \
    chmod -R a+rX ${PLAYWRIGHT_BROWSERS_PATH}

# Fail the build — not the first request — if the browser cannot actually launch
# (missing system library, wrong revision, unreadable files).
RUN python -c "\
from playwright.sync_api import sync_playwright;\
p = sync_playwright().start();\
b = p.chromium.launch(headless=True);\
print('chromium ok:', b.version);\
b.close(); p.stop()"

# pyproject.toml is the single source of truth for dependencies (built with
# hatchling). We extract just the runtime dependency list with stdlib tomllib and
# pip-install it, so this layer is cached on pyproject.toml alone and does not
# depend on the application source copied later. pip's download cache is a
# BuildKit cache mount, so changing one dependency doesn't re-download the rest.
# Playwright is already installed at the same pin, so pip leaves it alone.
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; \
open('/tmp/reqs.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
 && pip install --retries 10 --timeout 120 --resume-retries 10 -r /tmp/reqs.txt \
 && rm /tmp/reqs.txt

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
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/healthz || exit 1

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
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; \
open('/tmp/dev.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['dev']))" \
 && pip install --retries 10 --timeout 120 --resume-retries 10 -r /tmp/dev.txt \
 && rm /tmp/dev.txt
COPY --chown=10001:10001 tests/ ./tests/
# pytest config lives in pyproject.toml ([tool.pytest.ini_options]), already at /app.
USER 10001:10001
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "-q"]
