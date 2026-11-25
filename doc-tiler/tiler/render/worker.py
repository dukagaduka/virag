"""Worker: what runs inside a render worker process.

The RenderPool gives a process its tasks by name — `Worker.cut`, `Worker.render` — and
a task names a document by its type and path. The process has one Worker: it opens the
reader for the document and keeps it open (a PDF stays parsed between the tasks of one
document), dropping the per-task state after every task. The render params are the same
for every task, so the worker gets them once, at start. Its address space is capped
there too, so a document that needs more memory fails with RenderTooLarge instead of
drawing the kernel's OOM killer onto the whole container.
"""

from __future__ import annotations

import resource
from typing import Any, Callable, ClassVar

from ..models import DocType
from .geometry import Tile
from .params import RenderParams
from .pixels import DecodeError, RenderTooLarge, to_jpeg
from .readers import READERS, DocumentReader
from .types import EncodedImage


class Worker:
    """One render worker process: its render params and the reader it keeps open."""

    # JPEG cannot hold an image taller than this.
    MAX_JPEG_SIDE = 65_500

    # This process's worker. A task reaches a process, not an object, so the process
    # keeps its one worker here (`start` puts it there).
    process: ClassVar[Worker | None] = None


    def __init__(self, params: RenderParams):
        self.params = params
        self.reader: DocumentReader | None = None


    # ------------------------------------------------------------------ #
    # what the RenderPool runs in a worker process
    # ------------------------------------------------------------------ #
    @classmethod
    def start(cls, memory_bytes: int, params: RenderParams) -> None:
        """The process's initializer: cap its address space (0 = no cap) and make its
        worker, with the render params."""
        if memory_bytes > 0:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        cls.process = cls(params)


    @classmethod
    def cut(cls, doc_type: DocType, path: str) -> tuple[int, list[Tile]]:
        """Task: the height of the document image and the tiles it is cut into."""
        return cls.process.on(doc_type, path, lambda reader: reader.cut())


    @classmethod
    def render(cls, doc_type: DocType, path: str, tiles: list[Tile]) -> list[EncodedImage]:
        """Task: read and encode the given tiles of one document, in order."""
        return cls.process.on(doc_type, path, lambda reader: [cls.encode(reader, tile) for tile in tiles])


    # ------------------------------------------------------------------ #
    # the worker
    # ------------------------------------------------------------------ #
    def on(self, doc_type: DocType, path: str, work: Callable[[DocumentReader], Any]) -> Any:
        """Run `work` on the worker's reader of this document — opened now, unless the
        last task left it open — mapping memory exhaustion."""
        if type(self.reader) is not READERS[doc_type] or self.reader.path != path:
            self.forget()
            self.reader = READERS[doc_type](path, self.params)

        reader = self.reader

        try:
            return work(reader)

        except MemoryError as exc:
            self.forget()

            raise RenderTooLarge(str(exc) or "out of memory") from None

        except BaseException:
            self.forget()
            raise

        finally:
            reader.trim()


    def forget(self) -> None:
        """Close the reader the worker holds: another document's turn, or a failure left
        its state untrustworthy."""
        if self.reader is not None:
            try:
                self.reader.close()

            except Exception:  # noqa: BLE001 - it is being thrown away
                pass

        self.reader = None


    @classmethod
    def encode(cls, reader: DocumentReader, tile: Tile) -> EncodedImage:
        """Read one tile's rows and encode them."""
        if tile.height > cls.MAX_JPEG_SIDE:
            raise DecodeError(f"tile {tile.num} is {tile.height} px tall; JPEG allows {cls.MAX_JPEG_SIDE}")

        image = reader.read(tile.offset, tile.offset + tile.height)

        return EncodedImage(height=image.height, jpeg=to_jpeg(image, reader.params.jpeg_quality))
