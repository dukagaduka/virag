#!/bin/sh
# Fail fast and loudly before uvicorn binds a port:
#   - config validation, metadata schema parse
#   - startup geometry guard against the existing volume
#   - blob dir writability as the *runtime user* (catches root-owned volumes)
set -eu

# --- Identity lock -----------------------------------------------------------
# The image bakes in uid/gid 10001 (see Dockerfile: ENV APP_UID/APP_GID). Docker
# still lets an operator override the runtime user via `user:` in compose or
# `docker run --user`, applied by the daemon before this script runs. We cannot
# forbid that at the image level, but we can refuse to run under any other
# identity — which makes the override pointless. Do not weaken this check.
# The expected identity is a literal here, NOT $APP_UID — an env var can itself be
# overridden in compose, so trusting it would reopen the very hole this closes.
_uid="$(id -u)"
_gid="$(id -g)"
if [ "$_uid" != "10001" ] || [ "$_gid" != "10001" ]; then
  echo "tiler: FATAL: refusing to run as uid=${_uid} gid=${_gid};" >&2
  echo "       this image is locked to uid=10001 gid=10001." >&2
  echo "       Remove any 'user:' override in compose / '--user' on docker run." >&2
  exit 1
fi

echo "tiler: running as uid=${_uid} gid=${_gid}"

for dir in "${TILER_BLOB_DIR:-/data/blobs}" "${TILER_TMP_DIR:-/data/tmp}" "$(dirname "${TILER_DB_PATH:-/data/meta.sqlite}")"; do
  if ! mkdir -p "$dir" 2>/dev/null || [ ! -w "$dir" ]; then
    echo "tiler: FATAL: $dir is not writable by uid=$(id -u)." >&2
    echo "tiler: the volume is probably root-owned; fix it with:" >&2
    echo "       docker run --rm -v <volume>:/data alpine chown -R $(id -u):$(id -g) /data" >&2
    exit 1
  fi
done

python -m tiler.preflight

exec "$@"
