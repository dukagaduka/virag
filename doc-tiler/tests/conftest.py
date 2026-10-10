from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

DEFAULT_SCHEMA = """
version: 1
fields:
  - key: project
    type: text
  - key: source.host
    type: text
  - key: year
    type: number
"""


def write_schema(tmp_path: Path, body: str = DEFAULT_SCHEMA) -> Path:
    path = tmp_path / "metadata_schema.yaml"
    path.write_text(body)
    return path


def base_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "TILER_DB_PATH": str(tmp_path / "meta.sqlite"),
        "TILER_BLOB_DIR": str(tmp_path / "blobs"),
        "TILER_TMP_DIR": str(tmp_path / "tmp"),
        "TILER_METADATA_SCHEMA_PATH": str(write_schema(tmp_path)),
        "TILER_N_RENDER_WORKERS": "2",
        "TILER_N_BROWSERS": "1",
        "TILER_BROWSER_ENABLED": "false",
        "TILER_URL_FETCH_ENABLED": "false",
        "TILER_ALLOW_DELETE": "true",
        "TILER_LOG_LEVEL": "WARNING",
    }
    env.update(overrides)
    return env


@contextlib.asynccontextmanager
async def running_app(env: dict[str, str]):
    """Start a fresh app (running its real lifespan) against the given environment."""
    from tiler.main import create_app

    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    app = create_app()
    try:
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://tiler") as client:
                yield client, app.state.service
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest_asyncio.fixture
async def client(tmp_path):
    async with running_app(base_env(tmp_path)) as (client, service):
        client.service = service
        yield client


# --------------------------------------------------------------------------- #
# Fixture documents
# --------------------------------------------------------------------------- #
def make_pdf(pages: int = 3, width: float = 595, height: float = 842) -> bytes:
    import fitz

    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 100 + i * 10), f"Tiler test page {i}", fontsize=24)
        page.draw_rect(fitz.Rect(40, 40, width - 40, height - 40))
    data = doc.tobytes()
    doc.close()
    return data


def make_png(width: int = 1000, height: int = 2600) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 100):
        draw.rectangle([10, y + 10, width - 10, y + 60], outline=(0, 0, 0), width=2)
        draw.text((24, y + 24), f"row {y // 100}", fill=(20, 20, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def upload(doc_type: str, data: bytes, name: str, mime: str, **fields: str) -> dict:
    """Kwargs for `client.post("/tile", **upload(...))`."""
    return {
        "files": {"document": (name, data, mime)},
        "data": {"doc_type": doc_type, **fields},
    }


def pdf_form(data: bytes, name: str = "doc.pdf", **fields: str) -> dict:
    return upload("PDF", data, name, "application/pdf", **fields)


def png_form(data: bytes, name: str = "doc.png", **fields: str) -> dict:
    return upload("IMG", data, name, "image/png", **fields)


def mhtml_form(data: bytes, name: str = "doc.mhtml", **fields: str) -> dict:
    return upload("MHTML", data, name, "multipart/related", **fields)


@pytest.fixture
def sample_pdf() -> bytes:
    return make_pdf()


@pytest.fixture
def sample_png() -> bytes:
    return make_png()
