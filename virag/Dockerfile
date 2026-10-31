# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

# curl for the container healthcheck; nothing else is needed at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Unprivileged runtime user — nothing here ever runs as root.
ENV APP_UID=10001 \
    APP_GID=10001
RUN groupadd --gid 10001 virag \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin virag

WORKDIR /app

# Dependencies come from pyproject.toml (hatchling build backend); the list is
# extracted with stdlib tomllib so this layer caches on pyproject.toml alone.
# pip's download cache lives in a BuildKit cache mount — outside the image, kept
# between builds — so changing one dependency doesn't re-download all of them.
# pip is upgraded first for --resume-retries: on a flaky link a dropped download
# resumes instead of failing the whole build.
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -c "import tomllib; open('/tmp/reqs.txt','w').write(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" \
 && pip install --upgrade "pip>=25.1" \
 && pip install --retries 10 --timeout 120 --resume-retries 10 -r /tmp/reqs.txt \
 && rm /tmp/reqs.txt

COPY --chown=${APP_UID}:${APP_GID} virag/ ./virag/

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "virag.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
