"""Service settings: every `TILER_*` environment variable, its default and its bounds —
declared once — the geometry and pixel tiers, and `render_version`.

The settings come in sections, one per part of the service that reads them; all of them
share the `TILER_` prefix. A field's name is the variable's suffix
(`RenderSettings.render_timeout_sec` ← `TILER_RENDER_TIMEOUT_SEC`), so the declarations
below double as the reference; the irregular names are spelled out with
`validation_alias`. An empty variable counts as unset.
"""

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
# A PDF page is rasterized at pdf_dpi, but never wider than this many tile widths:
# supersampling beyond it costs time and memory and adds nothing to a 875 px tile.
MAX_SUPERSAMPLE = 4


class ConfigError(RuntimeError):
    """Raised for an env configuration that can never work; fatal at startup."""


class _Section(BaseSettings):
    """Shared config: immutable, empty variables count as unset, aliased fields can
    still be set by name (tests, code)."""

    model_config = SettingsConfigDict(
        env_prefix="TILER_",
        frozen=True,
        env_ignore_empty=True,
        extra="ignore",
        validate_by_name=True,
    )


class GeometrySettings(_Section):
    """How a document image is cut into tiles. A volume is bound to its geometry for
    life: another one is refused at startup."""

    tile_width: int = Field(875, gt=0)
    tile_height: int = Field(1024, gt=0)
    tile_overlap: int = Field(0, ge=0)
    min_tile_height: int = Field(28, gt=0)

    # PDF strip slicing (off by default → each PDF page stays one tile). When on, each
    # page is sliced into strips of pdf_tile_height (with pdf_tile_overlap), never across
    # page boundaries — finer tiles improve retrieval recall on dense text pages.
    pdf_slice: bool = Field(False, validation_alias="TILER_PDF_SLICE_ENABLE")
    pdf_tile_height: int = 1024
    pdf_tile_overlap: int = 0

    @model_validator(mode="after")
    def _consistent(self) -> GeometrySettings:
        if self.tile_overlap >= self.tile_height:
            raise ValueError("TILER_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_TILE_HEIGHT")

        if self.min_tile_height > self.tile_height:
            raise ValueError("TILER_MIN_TILE_HEIGHT must satisfy 0 < MIN <= TILER_TILE_HEIGHT")

        if self.pdf_slice:
            if self.pdf_tile_height <= 0:
                raise ValueError("TILER_PDF_TILE_HEIGHT must be positive when TILER_PDF_SLICE_ENABLE is set")

            if not 0 <= self.pdf_tile_overlap < self.pdf_tile_height:
                raise ValueError("TILER_PDF_TILE_OVERLAP must satisfy 0 <= OVERLAP < TILER_PDF_TILE_HEIGHT")

            if self.min_tile_height > self.pdf_tile_height:
                raise ValueError("TILER_MIN_TILE_HEIGHT must be <= TILER_PDF_TILE_HEIGHT when PDF slicing is on")

        return self


    @property
    def tier(self) -> dict[str, int]:
        """The geometry tier: what a volume is bound to."""
        return {
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "tile_overlap": self.tile_overlap,
            "min_tile_height": self.min_tile_height,
        }


    @property
    def pdf_tier(self) -> dict[str, object]:
        """PDF slicing params. Folded into render_version only when slicing is on, so
        deployments with slicing off keep their existing render_version untouched."""
        return {
            "pdf_tile_height": self.pdf_tile_height,
            "pdf_tile_overlap": self.pdf_tile_overlap,
        }


class PixelSettings(_Section):
    """What changes the bytes of a rendered tile besides the geometry."""

    jpeg_quality: int = Field(85, ge=1, le=100)
    pdf_dpi: int = Field(200, gt=0)

    # Overrides the Chromium build stamp folded into render_version (normally the
    # pinned Playwright bundle's revision).
    chromium_revision_override: str = Field("", validation_alias="TILER_CHROMIUM_REVISION", repr=False)


    @property
    def chromium_revision(self) -> str:
        """Resolved Chromium build id (env override or pinned bundle)."""
        return self.chromium_revision_override or self.chromium_build_id()


    @property
    def tier(self) -> dict[str, object]:
        """The pixel tier: the settings, constants and library versions that change
        output bytes."""
        return {
            "jpeg_quality": self.jpeg_quality,
            "jpeg_subsampling": JPEG_SUBSAMPLING,
            "jpeg_progressive": JPEG_PROGRESSIVE,
            "pdf_dpi": self.pdf_dpi,
            "pdf_max_supersample": MAX_SUPERSAMPLE,
            "resize_filter": RESIZE_FILTER,
            "chromium": self.chromium_revision,
            "pillow": self._lib_version("pillow"),
            "pymupdf": self._lib_version("pymupdf"),
        }


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


class RenderSettings(_Section):
    """Where rendering runs: the pools, their limits, the tile cache, Chromium's behaviour."""

    n_render_workers: int = Field(0, ge=0)      # 0 => cpu count
    n_browsers: int = Field(0, ge=0)            # 0 => min(cpu, 4)
    # Slots bulk work (ingest) may hold, so an interactive tile request always finds a free one.
    n_bulk_render_workers: int = Field(0, ge=0)  # 0 => render_workers // 2 (at least 1)
    n_bulk_browsers: int = Field(0, ge=0)        # 0 => browsers // 2 (at least 1)

    # Address space one render worker may take (bytes): a document that needs more fails
    # with 422 RENDER_TOO_LARGE instead of the kernel's OOM killer; 0 = no limit.
    render_worker_memory: int = Field(2 << 30, ge=0)
    # Execution time one render task may take; past it its worker is killed (and
    # replaced) and the document fails with 504 RENDER_TIMEOUT.
    render_timeout_sec: int = Field(120, gt=0)

    render_cache_size: int = 64                 # tiles kept by the tile cache

    browser_enabled: bool = True
    recycle_after: int = 500                    # renders after which Chromium is restarted
    nav_timeout_ms: int = 30000
    wait_network_idle: bool = True
    safe_fullpage_px: int = 30000               # a page taller than this is captured tile by tile


    @property
    def render_workers(self) -> int:
        """Process-pool size; defaults to the CPU count."""
        return self.n_render_workers or self._cpu_count()


    @property
    def browsers(self) -> int:
        """Concurrent Chromium render limit; defaults to min(cpu, 4)."""
        return self.n_browsers or min(self._cpu_count(), 4)


    @property
    def bulk_render_workers(self) -> int:
        """Render-pool slots bulk work may hold at once; defaults to half the pool."""
        return self.n_bulk_render_workers or max(1, self.render_workers // 2)


    @property
    def bulk_browsers(self) -> int:
        """Chromium renders bulk work may hold at once; defaults to half the pool."""
        return self.n_bulk_browsers or max(1, self.browsers // 2)


    @staticmethod
    @lru_cache(maxsize=1)
    def _cpu_count() -> int:
        """Available CPU count, honouring cgroup/cpuset pinning where possible."""
        try:
            return len(os.sched_getaffinity(0))  # respects cgroup/cpuset pinning

        except AttributeError:  # pragma: no cover - non-Linux
            return multiprocessing.cpu_count()


class StoreSettings(_Section):
    """Where the documents live, and how large one may be."""

    db_path: str = "/data/meta.sqlite"
    blob_dir: str = "/data/blobs"
    tmp_dir: str = ""                           # defaults to <blob_dir>/../tmp
    metadata_schema_path: str = "/config/metadata_schema.yaml"
    max_doc_bytes: int = Field(67_108_864, gt=0)


    @property
    def temp_dir(self) -> str:
        """Scratch directory for MHTML archive temp files during rendering."""
        return self.tmp_dir or str(Path(self.blob_dir).parent / "tmp")


class CaptureSettings(_Section):
    """What a URL ingest may fetch."""

    url_fetch_enabled: bool = True
    url_allow_private: bool = False


class RevisionSettings(_Section):
    """What the tiler does about docs rendered by another render_version."""

    # On: they are re-rendered at startup, before the service turns ready; off → it
    # stays not-ready until POST /revision.
    revision_on_render_mismatch: bool = True
    revision_concurrency: int = Field(4, gt=0)  # docs re-rendered at once


class ApiSettings(_Section):
    """Who may call the service, and what the callers may do."""

    # Bearer key for every route but liveness, readiness and metrics; empty = open.
    api_key: str = Field("", repr=False)
    allow_delete: bool = False


class Settings(_Section):
    """Every section, and what is computed across them."""

    geometry_settings: GeometrySettings = Field(default_factory=GeometrySettings)
    pixel_settings: PixelSettings = Field(default_factory=PixelSettings)
    render_settings: RenderSettings = Field(default_factory=RenderSettings)
    store_settings: StoreSettings = Field(default_factory=StoreSettings)
    capture_settings: CaptureSettings = Field(default_factory=CaptureSettings)
    revision_settings: RevisionSettings = Field(default_factory=RevisionSettings)
    api_settings: ApiSettings = Field(default_factory=ApiSettings)

    @model_validator(mode="after")
    def _fullpage_holds_a_tile(self) -> Settings:
        if self.render_settings.safe_fullpage_px < self.geometry_settings.tile_height:
            raise ValueError("TILER_SAFE_FULLPAGE_PX must be >= TILER_TILE_HEIGHT")

        return self


    @classmethod
    def from_env(cls) -> Settings:
        """Read every section from the environment; any invalid variable becomes one
        ConfigError naming the variables at fault."""
        try:
            return cls()

        except ValidationError as e:
            raise ConfigError(cls._describe(e)) from e


    @property
    def render_version(self) -> str:
        """Stable short hash over the geometry tier plus the pixel tier (and the PDF
        slicing tier only when PDF slicing is enabled)."""
        geometry = self.geometry_settings
        tiers: dict[str, object] = {"geometry": geometry.tier, "pixel": self.pixel_settings.tier}

        if geometry.pdf_slice:
            tiers["pdf"] = geometry.pdf_tier

        # Byte-identical to json.dumps(sort_keys=True, separators=(",", ":")): changing
        # the serializer must never shift render_version (see test_guard).
        payload = orjson.dumps(tiers, option=orjson.OPT_SORT_KEYS)

        return hashlib.sha256(payload).hexdigest()[:16]


    @staticmethod
    def _describe(error: ValidationError) -> str:
        """Render a validation error in terms of environment variable names."""
        model = _SECTIONS.get(error.title)
        problems = []

        for item in error.errors():
            message = item["msg"].removeprefix("Value error, ")

            if item["loc"] and model is not None:
                problems.append(f"{Settings._env_name(model, str(item['loc'][0]))}: {message}")
            else:
                problems.append(message)

        return "; ".join(problems)


    @staticmethod
    def _env_name(model: type[BaseSettings], loc: str) -> str:
        """The variable behind an error location — a field name, or already an alias."""
        info = model.model_fields.get(loc)

        if info is None:
            return loc

        if isinstance(info.validation_alias, str):
            return info.validation_alias

        return f"TILER_{loc}".upper()


_SECTIONS: dict[str, type[BaseSettings]] = {
    section.__name__: section
    for section in (
        Settings,
        GeometrySettings,
        PixelSettings,
        RenderSettings,
        StoreSettings,
        CaptureSettings,
        RevisionSettings,
        ApiSettings,
    )
}
