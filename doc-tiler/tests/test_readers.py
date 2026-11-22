"""Readers: a document read as one image — PDF pages stacked, rows rasterized on demand."""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageChops, ImageStat

from conftest import make_pdf, make_png

from tiler.models import DocType
from tiler.render import worker
from tiler.render.geometry import Tile, Tiling
from tiler.render.readers import ImageReader, PdfReader, RenderParams
from tiler.render.pixels import DecodeError

PARAMS = RenderParams(
    tile_width=875,
    jpeg_quality=85,
    pdf_dpi=200,
    tiling=Tiling(1024, 0, 28),
    pdf_tiling=Tiling(1024, 0, 28, whole=True),
)
SLICED = replace(PARAMS, pdf_tiling=Tiling(500, 100, 28))


def _file(tmp_path: Path, data: bytes, name: str = "doc.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _text_pdf(*sizes: tuple[float, float], lines: int = 40) -> bytes:
    doc = pymupdf.open()

    for width, height in sizes:
        page = doc.new_page(width=width, height=height)

        for i in range(lines):
            page.insert_text((20, 20 + i * height / lines), f"row {i} " * 12, fontsize=height / lines / 2)

    return doc.tobytes()


def _decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def test_pdf_pages_are_stacked_into_one_image(tmp_path):
    height, tiles = PdfReader(_file(tmp_path, make_pdf(pages=3)), PARAMS).layout()

    page = tiles[0].height
    assert abs(page - round(875 * 842 / 595)) <= 2                    # A4 at tile width
    assert tiles == [Tile(0, 0, page), Tile(1, page, page), Tile(2, 2 * page, page)]
    assert height == 3 * page


def test_pdf_pages_may_differ_in_size(tmp_path):
    """A landscape page among portrait ones is shorter at tile width; each tile starts
    where the page before it ends."""
    path = _file(tmp_path, _text_pdf((595, 842), (842, 595), (595, 842)))
    height, tiles = PdfReader(path, PARAMS).layout()

    portrait, landscape = tiles[0].height, tiles[1].height
    assert abs(landscape - round(875 * 595 / 842)) <= 2 and landscape < portrait
    assert [t.offset for t in tiles] == [0, portrait, portrait + landscape]
    assert height == 2 * portrait + landscape


def test_sliced_pdf_tiles_never_cross_a_page(tmp_path):
    reader = PdfReader(_file(tmp_path, make_pdf(pages=3)), SLICED)
    height, tiles = reader.layout()

    assert len(tiles) > 3
    assert [t.num for t in tiles] == list(range(len(tiles)))

    for tile in tiles:
        page = reader._page_at(tile.offset)
        assert tile.offset + tile.height <= page.top + page.height


def test_a_tile_is_deterministic(tmp_path):
    path = _file(tmp_path, make_pdf(pages=2))
    _, tiles = PdfReader(path, PARAMS).layout()

    first = worker.encode_tile(PdfReader(path, PARAMS), tiles[1])
    again = worker.encode_tile(PdfReader(path, PARAMS), tiles[1])     # a fresh worker's view

    assert first == again


def test_rows_read_apart_join_without_seams(tmp_path):
    """Rows read piece by piece match the page read whole up to resampling noise: a
    tile depends only on its own rows, which is what lets a tile request read just them."""
    path = _file(tmp_path, _text_pdf((595, 842)))
    height, _ = PdfReader(path, PARAMS).layout()

    whole = PdfReader(path, PARAMS).read(0, height)
    reader = PdfReader(path, PARAMS)
    pieces = Image.new("RGB", whole.size)

    for start in range(0, height, 300):
        pieces.paste(reader.read(start, min(height, start + 300)), (0, start))

    diff = ImageStat.Stat(ImageChops.difference(whole, pieces))
    assert max(diff.mean) < 0.5                                       # per channel, of 255
    assert max(hi for _, hi in diff.extrema) < 64


def test_a_read_must_stay_within_one_page(tmp_path):
    reader = PdfReader(_file(tmp_path, make_pdf(pages=2)), PARAMS)
    height, tiles = reader.layout()
    page = tiles[0].height

    with pytest.raises(DecodeError):
        reader.read(page - 10, page + 10)                             # across the boundary

    with pytest.raises(DecodeError):
        reader.read(height, height + 10)                              # past the document


def test_a_huge_page_renders_at_capped_resolution(tmp_path):
    """14400 pt square (PDF's maximum) at 200 dpi would be a 40000 px raster; capped at
    4 tile widths it is 3500 px and renders in moments."""
    reader = PdfReader(_file(tmp_path, _text_pdf((14400, 14400))), PARAMS)

    started = time.perf_counter()
    height, [tile] = reader.layout()
    encoded = worker.encode_tile(reader, tile)

    assert height == 875
    assert _decode(encoded.jpeg).size == (875, 875)
    assert time.perf_counter() - started < 20


def test_an_a4_page_keeps_the_full_dpi(tmp_path):
    reader = PdfReader(_file(tmp_path, make_pdf(pages=1)), PARAMS)

    [page] = reader._layout_pages()
    assert page.box.width == round(595 * 200 / 72)                    # not capped: 1653 < 3500


def test_image_reader_crops_its_rows(tmp_path):
    img = Image.new("RGB", (1750, 400), (255, 255, 255))
    img.paste((0, 0, 0), (0, 200, 1750, 400))                         # bottom half black
    path = tmp_path / "half.png"
    img.save(path)

    reader = ImageReader(str(path), PARAMS)
    assert reader.layout() == (200, [Tile(0, 0, 200)])                # 1750 → 875 wide

    top = ImageStat.Stat(reader.read(0, 90)).mean
    bottom = ImageStat.Stat(reader.read(110, 200)).mean
    assert min(top) > 250 and max(bottom) < 5


def test_trim_drops_the_pixels_and_close_the_document(tmp_path):
    reader = PdfReader(_file(tmp_path, make_pdf(pages=1)), PARAMS)
    reader.layout()
    reader.read(0, 10)
    assert reader._drawn is not None

    reader.trim()
    assert reader._drawn is None and reader._doc is not None and reader._pages is not None

    reader.close()
    assert reader._doc is None and reader._pages is None


def test_a_worker_keeps_its_reader_while_the_document_stays(tmp_path):
    pdf = _file(tmp_path, make_pdf(pages=2))
    png = _file(tmp_path, make_png(400, 300), "doc.png")
    worker.init_worker(0, PARAMS)

    try:
        _, tiles = worker.plan(DocType.PDF, pdf)
        reader = worker._current
        assert isinstance(reader, PdfReader) and reader._doc is not None

        worker.render(DocType.PDF, pdf, tiles[:1])
        assert worker._current is reader                              # the PDF stayed parsed
        assert reader._drawn is None                                  # its pixels did not

        assert worker.plan(DocType.IMG, png) == (656, [Tile(0, 0, 656)])
        assert isinstance(worker._current, ImageReader)
        assert reader._doc is None                                    # closed for the next document

    finally:
        worker._forget()
