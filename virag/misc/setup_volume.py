#!/usr/bin/env python3
"""Pre-create a deploy's bind-mount volume directories, world-writable.

Docker's daemon auto-creates any missing bind-mount source directory as root, so
a container that drops privileges (e.g. the tiler runs as uid 10001 and refuses a
root-owned /data) can't write to it. Creating the directories up front — owned by
you, mode 0777 — sidesteps that: whatever uid a container drops to can write.

It reads a deploy ``.env``, picks every ``*_VOLUME_PATH`` variable, and creates
``<path>/<basename-of-its-value>`` for each. For example:

    TILER_DEPLOY_VOLUME_PATH=./.smoke-data/tiler   ->  <path>/tiler
    QDRANT_DEPLOY_VOLUME_PATH=./.smoke-data/qdrant  ->  <path>/qdrant

Only the basename is used, so ``--path`` is simply the root that holds all the
volumes; where the .env happened to point them doesn't matter.

Usage:
    python misc/setup_volume.py --path ./.smoke-data --env deploy/.env
    python misc/setup_volume.py --path /srv/virag/data --env deploy/.env --mode 770

No third-party dependencies; nothing is imported from the virag package.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Every deploy variable naming a bind-mount host directory ends in this.
VOLUME_SUFFIX = "_VOLUME_PATH"


def parse_env(env_path: Path) -> dict[str, str]:
    """Parse a .env into a dict, ignoring blanks, comments, `export` and quotes."""
    values: dict[str, str] = {}

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        key, sep, val = line.partition("=")
        if not sep:
            continue

        values[key.strip()] = val.strip().strip('"').strip("'")

    return values


def volume_dirs(env: dict[str, str]) -> dict[str, str]:
    """Map each non-empty *_VOLUME_PATH variable to the basename of its value."""
    dirs: dict[str, str] = {}

    for key, val in env.items():
        if key.endswith(VOLUME_SUFFIX) and val:
            dirs[key] = Path(val).name

    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pre-create a deploy's volume directories under one root, world-writable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Reads every *_VOLUME_PATH from --env and creates <path>/<basename>\n"
            "with mode 0777, so containers that drop to a non-root uid can write."
        ),
    )
    ap.add_argument(
        "--path", type=Path, required=True,
        help="root directory that holds all volumes",
    )
    ap.add_argument(
        "--env", type=Path, required=True,
        help="deploy .env to read *_VOLUME_PATH variables from",
    )
    ap.add_argument(
        "--mode", default="777",
        help="octal permission bits applied to each dir (default: %(default)s)",
    )
    args = ap.parse_args()

    if not args.env.is_file():
        return f"env file not found: {args.env}"

    try:
        mode = int(args.mode, 8)

    except ValueError:
        return f"--mode must be octal digits, e.g. 777 (got {args.mode!r})"

    env = parse_env(args.env)
    dirs = volume_dirs(env)
    if not dirs:
        return f"no {VOLUME_SUFFIX} variables found in {args.env}"

    root = args.path
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, mode)

    print(f"creating {len(dirs)} volume dir(s) under {root} (mode {oct(mode)[2:]})")

    for key, name in sorted(dirs.items(), key=lambda kv: kv[1]):
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, mode)

        print(f"  + {name:<12} <- {key}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
