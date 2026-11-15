"""What runs inside a render worker process: the entry points the RenderPool submits.

A worker keeps the reader of its last task open (a PDF stays parsed between the tasks
of one document) and drops the per-task state after every task. Its address space is
capped at start, so a document that needs more memory fails with RenderTooLarge
instead of drawing the kernel's OOM killer onto the whole container.
"""

from __future__ import annotations

import resource

from PIL import Image

from .geometry import TileSpec, bands_for
from .readers import CpuReader
from .render import DecodeError, RenderTooLarge, to_jpeg
from .types import EncodedImage

# JPEG cannot hold an image taller than this.
MAX_JPEG_SIDE = 65_500

_current: CpuReader | None = None


def init_worker(memory_bytes: int) -> None:
    """Cap the worker's address space (0 = no cap)."""
    if memory_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def page_heights(reader: CpuReader) -> list[int]:
    """The document's page heights at tile width."""
    return _run(reader, lambda r: r.page_heights())


def render_tiles(reader: CpuReader, specs: list[TileSpec]) -> list[EncodedImage]:
    """Build and encode the given tiles of one document, in order."""
    return _run(reader, lambda r: [build_tile(r, spec) for spec in specs])


def build_tile(reader: CpuReader, spec: TileSpec) -> EncodedImage:
    """Assemble one tile from bands of at most `reader.band_rows` rows and encode it.

    While more than a band is missing the next read takes a full band; the last one
    takes exactly the rest, so no read crosses the tile's edges.
    """
    if spec.height > MAX_JPEG_SIDE:
        raise DecodeError(f"tile {spec.tile_num} is {spec.height} px tall; JPEG allows {MAX_JPEG_SIDE}")

    band = reader.band_rows(spec.page)

    if band >= spec.height:
        tile = reader.read(spec.page, spec.offset, spec.offset + spec.height)

    else:
        tile = Image.new("RGB", (reader.params.tile_width, spec.height), (255, 255, 255))

        for start, end in bands_for(spec.height, band):
            tile.paste(reader.read(spec.page, spec.offset + start, spec.offset + end), (0, start))

    return EncodedImage(height=tile.height, jpeg=to_jpeg(tile, reader.params.jpeg_quality))


def _run(reader: CpuReader, work):
    """Run `work` on the worker's reader for this document, mapping memory exhaustion."""
    global _current

    if (
        _current is None
        or type(_current) is not type(reader)
        or (_current.path, _current.params) != (reader.path, reader.params)
    ):
        if _current is not None:
            _current.close()

        _current = reader

    try:
        return work(_current)

    except MemoryError as exc:
        _forget()

        raise RenderTooLarge(str(exc) or "out of memory") from None

    except BaseException:
        _forget()
        raise

    finally:
        if _current is not None:
            _current.trim()


def _forget() -> None:
    """Close the cached reader after a failure: its state is no longer trustworthy."""
    global _current

    if _current is not None:
        try:
            _current.close()

        except Exception:  # noqa: BLE001 - it is being thrown away
            pass

    _current = None
