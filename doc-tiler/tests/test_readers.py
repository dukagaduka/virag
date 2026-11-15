"""Readers: banded PDF rasterization is deterministic, seamless and memory-bound."""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops, ImageStat

from conftest import make_pdf

from tiler.render import worker
from tiler.render.geometry import TileSpec
from tiler.render.readers import ImageReader, PdfReader, RenderParams

PARAMS = RenderParams(tile_width=875, jpeg_quality=85, pdf_dpi=200)


def _pdf(tmp_path: Path, data: bytes, name: str = "doc.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def _text_pdf(width: float, height: float, lines: int = 40) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)

    for i in range(lines):
        page.insert_text((20, 20 + i * height / lines), f"row {i} " * 12, fontsize=height / lines / 2)

    return doc.tobytes()


def _decode(jpeg: bytes) -> Image.Image:
    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def test_a_banded_tile_is_deterministic(tmp_path):
    path = _pdf(tmp_path, make_pdf(pages=2))
    banded = replace(PARAMS, band_bytes=400_000)                     # several bands per page

    reader = PdfReader(path, banded)
    heights = reader.page_heights()
    spec = TileSpec(tile_num=1, page=1, offset=0, height=heights[1])
    assert reader.band_rows(1) < heights[1]

    first = worker.build_tile(reader, spec)
    again = worker.build_tile(PdfReader(path, banded), spec)          # a fresh worker's view

    assert first == again


def test_bands_join_without_seams(tmp_path):
    """A banded tile matches the single-band one up to resampling noise: no shifted or
    missing rows at the band edges."""
    path = _pdf(tmp_path, _text_pdf(595, 842))
    height = PdfReader(path, PARAMS).page_heights()[0]

    whole = PdfReader(path, PARAMS)
    banded = PdfReader(path, replace(PARAMS, band_bytes=300_000))
    assert whole.band_rows(0) >= height > banded.band_rows(0)

    a = whole.read(0, 0, height)
    b = Image.new("RGB", a.size)

    for start in range(0, height, banded.band_rows(0)):
        end = min(height, start + banded.band_rows(0))
        b.paste(banded.read(0, start, end), (0, start))

    diff = ImageStat.Stat(ImageChops.difference(a, b))
    assert max(diff.mean) < 0.5                                       # per channel, of 255
    assert max(hi for _, hi in diff.extrema) < 64


def test_a_huge_page_renders_at_capped_resolution(tmp_path):
    """14400 pt square (PDF's maximum) at 200 dpi would be a 40000 px raster; capped at
    4 tile widths it is 3500 px and renders in moments."""
    path = _pdf(tmp_path, _text_pdf(14400, 14400))
    reader = PdfReader(path, PARAMS)

    started = time.perf_counter()
    [height] = reader.page_heights()
    tile = worker.build_tile(reader, TileSpec(tile_num=0, page=0, offset=0, height=height))

    assert height == 875
    assert _decode(tile.jpeg).size == (875, 875)
    assert time.perf_counter() - started < 20


def test_an_a4_page_keeps_the_full_dpi(tmp_path):
    path = _pdf(tmp_path, make_pdf(pages=1))
    reader = PdfReader(path, PARAMS)
    reader.page_heights()

    raster = reader._page(0)
    assert raster.box.width == round(595 * 200 / 72)                 # not capped: 1653 < 3500


def test_image_reader_crops_its_rows(tmp_path):
    img = Image.new("RGB", (1750, 400), (255, 255, 255))
    img.paste((0, 0, 0), (0, 200, 1750, 400))                         # bottom half black
    path = tmp_path / "half.png"
    img.save(path)

    reader = ImageReader(str(path), PARAMS)
    assert reader.page_heights() == [200]                             # 1750 → 875 wide

    top = ImageStat.Stat(reader.read(0, 0, 90)).mean
    bottom = ImageStat.Stat(reader.read(0, 110, 200)).mean
    assert min(top) > 250 and max(bottom) < 5


def test_readers_pickle_without_open_state(tmp_path):
    import pickle

    reader = PdfReader(_pdf(tmp_path, make_pdf(pages=1)), PARAMS)
    reader.page_heights()
    reader.read(0, 0, 10)

    clone = pickle.loads(pickle.dumps(reader))
    assert (clone.path, clone.params) == (reader.path, reader.params)
    assert clone._doc is None and clone._raster is None
