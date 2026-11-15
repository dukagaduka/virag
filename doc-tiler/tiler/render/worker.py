"""What runs inside a render worker process: the entry points the RenderPool submits.

A task names a document by its type and path; the worker opens the reader for it and
keeps it open (a PDF stays parsed between the tasks of one document), dropping the
per-task state after every task. The render params are the same for every task, so a
worker gets them once, at start. Its address space is capped there too, so a document
that needs more memory fails with RenderTooLarge instead of drawing the kernel's OOM
killer onto the whole container.
"""

from __future__ import annotations

import resource

from ..models import DocType
from .geometry import Tile
from .readers import READERS, DocumentReader, RenderParams
from .render import DecodeError, RenderTooLarge, to_jpeg
from .types import EncodedImage

# JPEG cannot hold an image taller than this.
MAX_JPEG_SIDE = 65_500

_params: RenderParams | None = None
_current: DocumentReader | None = None


def init_worker(memory_bytes: int, params: RenderParams) -> None:
    """Cap the worker's address space (0 = no cap) and take the render params."""
    global _params

    if memory_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    _params = params


def plan(doc_type: DocType, path: str) -> tuple[int, list[Tile]]:
    """The height of the document image and the tiles it is cut into."""
    return _run(doc_type, path, lambda reader: reader.layout())


def render(doc_type: DocType, path: str, tiles: list[Tile]) -> list[EncodedImage]:
    """Read and encode the given tiles of one document, in order."""
    return _run(doc_type, path, lambda reader: [encode_tile(reader, tile) for tile in tiles])


def encode_tile(reader: DocumentReader, tile: Tile) -> EncodedImage:
    """Read one tile's rows and encode them."""
    if tile.height > MAX_JPEG_SIDE:
        raise DecodeError(f"tile {tile.num} is {tile.height} px tall; JPEG allows {MAX_JPEG_SIDE}")

    image = reader.read(tile.offset, tile.offset + tile.height)

    return EncodedImage(height=image.height, jpeg=to_jpeg(image, reader.params.jpeg_quality))


def _run(doc_type: DocType, path: str, work):
    """Run `work` on the worker's reader of this document, mapping memory exhaustion."""
    global _current

    if type(_current) is not READERS[doc_type] or _current.path != path:
        _forget()
        _current = READERS[doc_type](path, _params)

    reader = _current

    try:
        return work(reader)

    except MemoryError as exc:
        _forget()

        raise RenderTooLarge(str(exc) or "out of memory") from None

    except BaseException:
        _forget()
        raise

    finally:
        reader.trim()


def _forget() -> None:
    """Close the reader the worker holds: another document's turn, or a failure left
    its state untrustworthy."""
    global _current

    if _current is not None:
        try:
            _current.close()

        except Exception:  # noqa: BLE001 - it is being thrown away
            pass

    _current = None
