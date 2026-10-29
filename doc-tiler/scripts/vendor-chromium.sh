#!/usr/bin/env bash
# Populate ./vendor/ms-playwright from a local Playwright cache so `docker build`
# needs no access to the Playwright CDN.
#
# Usage: scripts/vendor-chromium.sh [source-cache-dir]
# Default source: $PLAYWRIGHT_BROWSERS_PATH or ~/.cache/ms-playwright
#
# Vendor the revision matching the `playwright` pin in pyproject.toml: the build
# treats the copy as a seed and downloads the pinned revision when it differs (so a
# stale copy costs a download, not a broken image).
set -euo pipefail

SRC="${1:-${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor/ms-playwright"

# The executable's layout changed in newer Playwright (chrome-linux/headless_shell ->
# chrome-headless-shell-linux64/chrome-headless-shell); accept either.
found=0
for exe in "$SRC"/chromium_headless_shell-*/chrome-linux/headless_shell \
           "$SRC"/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell; do
  [ -e "$exe" ] && found=1
done

if [ "$found" = 0 ]; then
  echo "no Chromium headless shell found in $SRC" >&2
  echo "install it first:  python -m playwright install chromium-headless-shell" >&2
  exit 1
fi

# Headless shell only: launch(headless=True) uses it on Playwright 1.49+, so the
# ~550MB full Chromium bundle is dead weight for a headless-only service.
mkdir -p "$DEST"
for dir in "$SRC"/chromium_headless_shell-*; do
  [ -e "$dir" ] || continue
  echo "vendoring $(basename "$dir")"
  cp -a "$dir" "$DEST/"
done
echo "done: $DEST ($(du -sh "$DEST" | cut -f1))"
