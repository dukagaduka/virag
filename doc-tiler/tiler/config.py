"""Environment configuration, geometry/pixel tiers and `render_version`."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Pixel-tier constants that are not env-tunable but do change output bytes.
RESIZE_FILTER = "LANCZOS"
JPEG_SUBSAMPLING = 2  # 4:2:0, pinned explicitly (Pillow's default is quality-dependent)
JPEG_PROGRESSIVE = False


class ConfigError(RuntimeError):
    """Raised for an env configuration that can never work; fatal at startup."""


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to `default` when unset/empty."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var, falling back to `default` when unset/empty."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def _env_str(name: str, default: str) -> str:
    """Read a string env var, falling back to `default` when unset/empty."""
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else raw


@lru_cache(maxsize=1)
def _cpu_count() -> int:
    """Available CPU count, honouring cgroup/cpuset pinning where possible."""
    try:
        return len(os.sched_getaffinity(0))  # respects cgroup/cpuset pinning
    except AttributeError:  # pragma: no cover - non-Linux
        return multiprocessing.cpu_count()


def chromium_build_id() -> str:
    """Chromium build stamp folded into `render_version`, resolved without launching.

    Reads Playwright's bundled `browsers.json` (the pinned revision), so the value is
    available even when the browser cannot be launched (tests, PDF-only deployments).
    """
    override = os.environ.get("TILER_CHROMIUM_REVISION")
    if override:
        return override
    try:
        import playwright  # noqa: PLC0415 - optional at import time

        root = Path(playwright.__file__).parent
        for candidate in (
            root / "driver" / "package" / "browsers.json",
            root / "driver" / "browsers.json",
        ):
            if candidate.exists():
                data = json.loads(candidate.read_text())
                for browser in data.get("browsers", []):
                    if browser.get("name") == "chromium":
                        return f"chromium-{browser.get('revision')}"
        from importlib.metadata import version  # noqa: PLC0415

        return f"playwright-{version('playwright')}"
    except Exception:  # pragma: no cover - playwright missing entirely
        return "chromium-unknown"


def _lib_version(module: str, dist: str) -> str:
    """Installed version of a distribution, or 'unknown' if unavailable."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version(dist)
    except Exception:  # pragma: no cover
        return "unknown"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings derived from environment variables."""

    # --- geometry tier (hard-enforced at startup) ---
    tile_width: int = 875
    tile_height: int = 1024
    tile_overlap: int = 0
    min_tile_height: int = 28

    # PDF strip slicing (off by default → each PDF page stays one tile). When on, each
    # page is sliced into strips of pdf_tile_height (with pdf_tile_overlap), never across
    # page boundaries — finer tiles improve retrieval recall on dense text pages.
    pdf_slice: bool = False
    pdf_tile_height: int = 1024
    pdf_tile_overlap: int = 0

    # --- pixel tier (folded into render_version, soft-checked) ---
    jpeg_quality: int = 85
    pdf_dpi: int = 200

    # --- runtime ---
    safe_fullpage_px: int = 30000
    n_browsers: int = 0  # 0 => min(cpu, 4)
    n_cpu_workers: int = 0  # 0 => cpu count
    recycle_after: int = 500
    render_cache_size: int = 64
    max_doc_bytes: int = 67_108_864
    nav_timeout_ms: int = 30000
    wait_network_idle: bool = True

    # Incomplete-ingest sweeper: reap streaming ingests left incomplete this long
    # (must exceed the longest render); 0 disables the sweeper.
    incomplete_ttl_sec: int = 0
    sweep_interval_sec: int = 300

    db_path: str = "/data/meta.sqlite"
    blob_dir: str = "/data/blobs"
    tmp_dir: str = ""  # defaults to <blob_dir>/../tmp
    metadata_schema_path: str = "/config/metadata_schema.yaml"

    url_fetch_enabled: bool = True
    url_allow_private: bool = False
    render_mismatch_policy: str = "strict"
    allow_delete: bool = False
    browser_enabled: bool = True

    _chromium_revision: str = field(default="", repr=False)

    # --- binding secret (optional; empty = an explicit, unbindable slot) ---
    db_secret: str = field(default="", repr=False)

    # ------------------------------------------------------------------ #
    @property
    def step(self) -> int:
        """Vertical advance between strips (tile height minus overlap)."""
        return self.tile_height - self.tile_overlap

    @property
    def browsers(self) -> int:
        """Concurrent Chromium render limit; defaults to min(cpu, 4)."""
        return self.n_browsers or min(_cpu_count(), 4)

    @property
    def cpu_workers(self) -> int:
        """Process-pool size; defaults to the CPU count."""
        return self.n_cpu_workers or _cpu_count()

    @property
    def chromium_revision(self) -> str:
        """Resolved Chromium build id (env override or pinned bundle)."""
        return self._chromium_revision or chromium_build_id()

    @property
    def temp_dir(self) -> str:
        """Scratch directory for MHTML archive temp files during rendering."""
        return self.tmp_dir or str(Path(self.blob_dir).parent / "tmp")

    @property
    def geometry(self) -> dict[str, int]:
        """The geometry tier as a dict."""
        return {
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "tile_overlap": self.tile_overlap,
            "min_tile_height": self.min_tile_height,
        }

    @property
    def pixel_tier(self) -> dict[str, object]:
        """The pixel tier (params that change output bytes) as a dict."""
        return {
            "jpeg_quality": self.jpeg_quality,
            "jpeg_subsampling": JPEG_SUBSAMPLING,
            "jpeg_progressive": JPEG_PROGRESSIVE,
            "pdf_dpi": self.pdf_dpi,
            "resize_filter": RESIZE_FILTER,
            "chromium": self.chromium_revision,
            "pillow": _lib_version("PIL", "pillow"),
            "pymupdf": _lib_version("fitz", "pymupdf"),
        }

    @property
    def pdf_tier(self) -> dict[str, object]:
        """PDF slicing params. Folded into render_version only when slicing is on, so
        deployments with slicing off keep their existing render_version untouched."""
        return {
            "pdf_tile_height": self.pdf_tile_height,
            "pdf_tile_overlap": self.pdf_tile_overlap,
        }

    @property
    def render_version(self) -> str:
        """Stable short hash over the geometry tier plus the pixel tier (and the PDF
        slicing tier only when PDF slicing is enabled)."""
        tiers: dict[str, object] = {"geometry": self.geometry, "pixel": self.pixel_tier}
        if self.pdf_slice:
            tiers["pdf"] = self.pdf_tier
        payload = json.dumps(tiers, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def secret_hash(self) -> str:
        """sha256 of TILER_DB_SECRET, or '' when unset — an explicit empty binding
        slot, bound to the volume for life (like geometry). A consumer (virag) uses
        it to refuse an unrelated store; the tiler itself stays usable standalone."""
        return hashlib.sha256(self.db_secret.encode()).hexdigest() if self.db_secret else ""

    def validate(self) -> None:
        """Raise ConfigError if any setting is out of its valid range."""
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ConfigError("TILER_TILE_WIDTH and TILER_TILE_HEIGHT must be positive")
        if not 0 <= self.tile_overlap < self.tile_height:
            raise ConfigError(
                "TILER_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_TILE_HEIGHT"
            )
        if not 0 < self.min_tile_height <= self.tile_height:
            raise ConfigError(
                "TILER_MIN_TILE_HEIGHT must satisfy 0 < MIN <= TILER_TILE_HEIGHT"
            )
        if self.pdf_slice:
            if self.pdf_tile_height <= 0:
                raise ConfigError(
                    "TILER_PDF_TILE_HEIGHT must be positive when TILER_PDF_SLICE_ENABLE is set"
                )
            if not 0 <= self.pdf_tile_overlap < self.pdf_tile_height:
                raise ConfigError(
                    "TILER_PDF_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_PDF_TILE_HEIGHT"
                )
            if self.min_tile_height > self.pdf_tile_height:
                raise ConfigError(
                    "TILER_MIN_TILE_HEIGHT must be <= TILER_PDF_TILE_HEIGHT when PDF slicing is on"
                )
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigError("TILER_JPEG_QUALITY must be within 1..100")
        if self.pdf_dpi <= 0:
            raise ConfigError("TILER_PDF_DPI must be positive")
        if self.render_mismatch_policy not in {"strict", "lenient"}:
            raise ConfigError(
                "TILER_RENDER_MISMATCH_POLICY must be 'strict' or 'lenient'"
            )
        if self.max_doc_bytes <= 0:
            raise ConfigError("TILER_MAX_DOC_BYTES must be positive")
        if self.safe_fullpage_px < self.tile_height:
            raise ConfigError("TILER_SAFE_FULLPAGE_PX must be >= TILER_TILE_HEIGHT")


def load_settings() -> Settings:
    """Build and validate a Settings instance from the environment."""
    settings = Settings(
        tile_width=_env_int("TILER_TILE_WIDTH", 875),
        tile_height=_env_int("TILER_TILE_HEIGHT", 1024),
        tile_overlap=_env_int("TILER_TILE_OVERLAP", 0),
        min_tile_height=_env_int("TILER_MIN_TILE_HEIGHT", 28),
        pdf_slice=_env_bool("TILER_PDF_SLICE_ENABLE", False),
        pdf_tile_height=_env_int("TILER_PDF_TILE_HEIGHT", 1024),
        pdf_tile_overlap=_env_int("TILER_PDF_TILE_OVERLAP", 0),
        jpeg_quality=_env_int("TILER_JPEG_QUALITY", 85),
        pdf_dpi=_env_int("TILER_PDF_DPI", 200),
        safe_fullpage_px=_env_int("TILER_SAFE_FULLPAGE_PX", 30000),
        n_browsers=_env_int("TILER_N_BROWSERS", 0),
        n_cpu_workers=_env_int("TILER_N_CPU_WORKERS", 0),
        recycle_after=_env_int("TILER_RECYCLE_AFTER", 500),
        render_cache_size=_env_int("TILER_RENDER_CACHE_SIZE", 64),
        max_doc_bytes=_env_int("TILER_MAX_DOC_BYTES", 67_108_864),
        incomplete_ttl_sec=_env_int("TILER_INCOMPLETE_TTL_SEC", 0),
        sweep_interval_sec=_env_int("TILER_SWEEP_INTERVAL_SEC", 300),
        nav_timeout_ms=_env_int("TILER_NAV_TIMEOUT_MS", 30000),
        wait_network_idle=_env_bool("TILER_WAIT_NETWORK_IDLE", True),
        db_path=_env_str("TILER_DB_PATH", "/data/meta.sqlite"),
        blob_dir=_env_str("TILER_BLOB_DIR", "/data/blobs"),
        tmp_dir=_env_str("TILER_TMP_DIR", ""),
        metadata_schema_path=_env_str(
            "TILER_METADATA_SCHEMA_PATH", "/config/metadata_schema.yaml"
        ),
        url_fetch_enabled=_env_bool("TILER_URL_FETCH_ENABLED", True),
        url_allow_private=_env_bool("TILER_URL_ALLOW_PRIVATE", False),
        render_mismatch_policy=_env_str("TILER_RENDER_MISMATCH_POLICY", "strict").lower(),
        allow_delete=_env_bool("TILER_ALLOW_DELETE", False),
        browser_enabled=_env_bool("TILER_BROWSER_ENABLED", True),
        _chromium_revision=_env_str("TILER_CHROMIUM_REVISION", ""),
        db_secret=_env_str("TILER_DB_SECRET", ""),
    )
    settings.validate()
    return settings
