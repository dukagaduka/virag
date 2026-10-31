"""Environment configuration, geometry/pixel tiers and `render_version`."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import orjson
from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pixel-tier constants that are not env-tunable but do change output bytes.
RESIZE_FILTER = "LANCZOS"
JPEG_SUBSAMPLING = 2  # 4:2:0, pinned explicitly (Pillow's default is quality-dependent)
JPEG_PROGRESSIVE = False


class ConfigError(RuntimeError):
    """Raised for an env configuration that can never work; fatal at startup."""


class Settings(BaseSettings):
    """Immutable runtime settings, read from `TILER_*` environment variables.

    A field's name is the variable's suffix (`tile_width` ← `TILER_TILE_WIDTH`), so the
    declarations below double as the reference; the two irregular names are aliased.
    An empty variable counts as unset.
    """

    model_config = SettingsConfigDict(
        env_prefix="TILER_",
        frozen=True,
        env_ignore_empty=True,
        extra="ignore",
        validate_by_name=True,
    )

    # --- geometry tier (hard-enforced at startup) ---
    tile_width: int = 875
    tile_height: int = 1024
    tile_overlap: int = 0
    min_tile_height: int = 28

    # PDF strip slicing (off by default → each PDF page stays one tile). When on, each
    # page is sliced into strips of pdf_tile_height (with pdf_tile_overlap), never across
    # page boundaries — finer tiles improve retrieval recall on dense text pages.
    pdf_slice: bool = Field(False, validation_alias="TILER_PDF_SLICE_ENABLE")
    pdf_tile_height: int = 1024
    pdf_tile_overlap: int = 0

    # --- pixel tier (folded into render_version, soft-checked) ---
    jpeg_quality: int = 85
    pdf_dpi: int = 200

    # --- runtime ---
    safe_fullpage_px: int = 30000
    n_browsers: int = 0  # 0 => min(cpu, 4)
    n_render_workers: int = 0  # 0 => cpu count
    # Slots bulk work (ingest) may hold, so interactive get_tile always finds a free one.
    n_bulk_render_workers: int = 0  # 0 => render_workers // 2 (at least 1)
    n_bulk_browsers: int = 0  # 0 => browsers // 2 (at least 1)
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
    allow_delete: bool = False
    browser_enabled: bool = True

    # Startup revision: docs rendered by another render_version are re-rendered before
    # the service turns ready; off → it stays not-ready until POST /revision.
    revision_on_render_mismatch: bool = True
    revision_concurrency: int = 4  # docs re-rendered at once

    # Overrides the Chromium build stamp folded into render_version (normally the
    # pinned Playwright bundle's revision).
    chromium_revision_override: str = Field("", validation_alias="TILER_CHROMIUM_REVISION", repr=False)

    # --- binding secret (optional; empty = an explicit, unbindable slot) ---
    db_secret: str = Field("", repr=False)


    @model_validator(mode="after")
    def _validate(self) -> Settings:
        """Reject settings outside their valid range (cross-field geometry included)."""
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("TILER_TILE_WIDTH and TILER_TILE_HEIGHT must be positive")

        if not 0 <= self.tile_overlap < self.tile_height:
            raise ValueError(
                "TILER_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_TILE_HEIGHT"
            )

        if not 0 < self.min_tile_height <= self.tile_height:
            raise ValueError(
                "TILER_MIN_TILE_HEIGHT must satisfy 0 < MIN <= TILER_TILE_HEIGHT"
            )

        if self.pdf_slice:
            if self.pdf_tile_height <= 0:
                raise ValueError(
                    "TILER_PDF_TILE_HEIGHT must be positive when TILER_PDF_SLICE_ENABLE is set"
                )

            if not 0 <= self.pdf_tile_overlap < self.pdf_tile_height:
                raise ValueError(
                    "TILER_PDF_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_PDF_TILE_HEIGHT"
                )

            if self.min_tile_height > self.pdf_tile_height:
                raise ValueError(
                    "TILER_MIN_TILE_HEIGHT must be <= TILER_PDF_TILE_HEIGHT when PDF slicing is on"
                )

        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("TILER_JPEG_QUALITY must be within 1..100")

        if self.pdf_dpi <= 0:
            raise ValueError("TILER_PDF_DPI must be positive")

        if self.max_doc_bytes <= 0:
            raise ValueError("TILER_MAX_DOC_BYTES must be positive")

        if self.safe_fullpage_px < self.tile_height:
            raise ValueError("TILER_SAFE_FULLPAGE_PX must be >= TILER_TILE_HEIGHT")

        if self.n_bulk_render_workers < 0 or self.n_bulk_browsers < 0:
            raise ValueError("TILER_N_BULK_RENDER_WORKERS and TILER_N_BULK_BROWSERS must be >= 0")

        if self.revision_concurrency <= 0:
            raise ValueError("TILER_REVISION_CONCURRENCY must be positive")

        return self


    # ------------------------------------------------------------------ #
    @property
    def step(self) -> int:
        """Vertical advance between strips (tile height minus overlap)."""
        return self.tile_height - self.tile_overlap


    @property
    def browsers(self) -> int:
        """Concurrent Chromium render limit; defaults to min(cpu, 4)."""
        return self.n_browsers or min(self._cpu_count(), 4)


    @property
    def render_workers(self) -> int:
        """Process-pool size; defaults to the CPU count."""
        return self.n_render_workers or self._cpu_count()


    @property
    def bulk_render_workers(self) -> int:
        """Render-pool slots bulk work may hold at once; defaults to half the pool."""
        return self.n_bulk_render_workers or max(1, self.render_workers // 2)


    @property
    def bulk_browsers(self) -> int:
        """Chromium renders bulk work may hold at once; defaults to half the pool."""
        return self.n_bulk_browsers or max(1, self.browsers // 2)


    @property
    def chromium_revision(self) -> str:
        """Resolved Chromium build id (env override or pinned bundle)."""
        return self.chromium_revision_override or self.chromium_build_id()


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
            "pillow": self._lib_version("pillow"),
            "pymupdf": self._lib_version("pymupdf"),
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

        # Byte-identical to json.dumps(sort_keys=True, separators=(",", ":")): changing
        # the serializer must never shift render_version (see test_guard).
        payload = orjson.dumps(tiers, option=orjson.OPT_SORT_KEYS)

        return hashlib.sha256(payload).hexdigest()[:16]


    @property
    def secret_hash(self) -> str:
        """sha256 of TILER_DB_SECRET, or '' when unset — an explicit empty binding
        slot, bound to the volume for life (like geometry). A consumer (virag) uses
        it to refuse an unrelated store; the tiler itself stays usable standalone."""
        return hashlib.sha256(self.db_secret.encode()).hexdigest() if self.db_secret else ""


    @staticmethod
    @lru_cache(maxsize=1)
    def _cpu_count() -> int:
        """Available CPU count, honouring cgroup/cpuset pinning where possible."""
        try:
            return len(os.sched_getaffinity(0))  # respects cgroup/cpuset pinning

        except AttributeError:  # pragma: no cover - non-Linux
            return multiprocessing.cpu_count()


    @staticmethod
    def chromium_build_id() -> str:
        """Chromium build stamp folded into `render_version`, resolved without launching.

        Reads Playwright's bundled `browsers.json` (the pinned revision), so the value is
        available even when the browser cannot be launched (tests, PDF-only deployments).
        """
        try:
            import playwright  # noqa: PLC0415 - optional at import time

            root = Path(playwright.__file__).parent

            for candidate in (
                root / "driver" / "package" / "browsers.json",
                root / "driver" / "browsers.json",
            ):
                if candidate.exists():
                    data = orjson.loads(candidate.read_bytes())

                    for browser in data.get("browsers", []):
                        if browser.get("name") == "chromium":
                            return f"chromium-{browser.get('revision')}"

            return f"playwright-{version('playwright')}"

        except Exception:  # pragma: no cover - playwright missing entirely
            return "chromium-unknown"


    @staticmethod
    def _lib_version(dist: str) -> str:
        """Installed version of a distribution, or 'unknown' if unavailable."""
        try:
            return version(dist)

        except Exception:  # pragma: no cover
            return "unknown"


def load_settings() -> Settings:
    """Build and validate a Settings instance from the environment; any invalid value
    becomes a ConfigError naming the variable."""
    try:
        return Settings()

    except ValidationError as exc:
        problems = []

        for item in exc.errors():
            loc = str(item["loc"][0]) if item["loc"] else ""
            message = item["msg"].removeprefix("Value error, ")

            problems.append(f"{_env_name(loc)}: {message}" if loc else message)

        raise ConfigError("; ".join(problems)) from exc


def _env_name(loc: str) -> str:
    """The variable behind an error location — a field name, or already an alias."""
    info = Settings.model_fields.get(loc)

    if info is None:
        return loc

    if isinstance(info.validation_alias, str):
        return info.validation_alias

    return f"TILER_{loc}".upper()
