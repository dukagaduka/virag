from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from virag.errors import ApiError
from virag.settings import TilerSettings
from virag.utils import error_body


@dataclass(slots=True)
class Tile:
    tile_num: int
    width: int
    height: int
    image_hash: str
    data: str | None = None   # base64 JPEG, present iff include_bytes=true


@dataclass(slots=True)
class TileResult:
    doc_id: str
    doc_type: str
    source_url: str | None
    render_version: str
    stored: bool
    page_height: int
    tile_count: int
    tiles: list[Tile]

    @classmethod
    def from_manifest(cls, data: dict) -> TileResult:
        return cls(
            doc_id=data["doc_id"],
            doc_type=data["doc_type"],
            source_url=data.get("source_url"),
            render_version=data["render_version"],
            stored=data["stored"],
            page_height=data["page_height"],
            tile_count=data["tile_count"],
            tiles=[
                Tile(
                    tile_num=t["tile_num"],
                    width=t["width"],
                    height=t["height"],
                    image_hash=t["image_hash"],
                    data=t.get("data"),
                )
                for t in data["tiles"]
            ],
        )


class TilerService:
    def __init__(
        self,
        base_url: str,
        timeout: float | httpx.Timeout = 120,
    ):

        self.base_url = base_url
        self.timeout  = timeout


    @property
    def tile_route(self):
        return f"{self.base_url}/tile"


    @property
    def get_tile_route(self):
        return f"{self.base_url}/get_tile"


    @property
    def readyz_route(self):
        return f"{self.base_url}/readyz"


    @property
    def info_route(self):
        return f"{self.base_url}/info"


    @classmethod
    def from_settings(cls, ts: TilerSettings) -> TilerService:
        return cls(
            base_url=ts.base_url,
            timeout=ts.timeout,
        )


    async def check_connection(self):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    self.readyz_route
                )

                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot establish connection to the {self.base_url} during the check",
                    details={"http_status_error": {"code": e.response.status_code}, "text": e.response.text},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.readyz_route} during the check",
                    details={"error": repr(e)},
                ) from e


    async def tile(
        self,
        doc_type: str,
        *,
        document: bytes | None = None,
        url: str | None = None,
        metadata: dict | None = None,
        store: bool = True,
        include_bytes: bool = True,
        force: bool = False,
    ) -> TileResult:

        params = {
            "store": store,
            "include_bytes": include_bytes,
            "force": force,
        }

        data: dict = {"doc_type": doc_type}
        if url is not None:
            data["url"] = url
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)

        files = (
            {"document": ("document", document, "application/octet-stream")}
            if document is not None
            else None
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self.tile_route,
                    params=params,
                    data=data,
                    files=files,
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Tiler rejected the ingest at {self.tile_route}",
                    details={"tiler_status_code": e.response.status_code, "tiler_error": error_body(e.response)},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.tile_route} during the ingest",
                    details={"error": repr(e)},
                ) from e

            try:
                return TileResult.from_manifest(response.json())

            except (KeyError, TypeError, ValueError) as e:
                raise ApiError(
                    status_code=422,
                    message=f"Cannot decode {self.tile_route} manifest or unexpected response structure",
                    details={"base_message": str(e)},
                ) from e


    async def get_tile(
        self,
        doc_id: str,
        tile_num: int,
        *,
        verify: bool = True,
    ) -> bytes:

        params = {
            "doc_id": doc_id,
            "tile_num": tile_num,
            "verify": verify,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    self.get_tile_route,
                    params=params,
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Tiler could not regenerate tile ({doc_id}, {tile_num}) at {self.get_tile_route}",
                    details={"tiler_status_code": e.response.status_code, "tiler_error": error_body(e.response)},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.get_tile_route} during tile regeneration",
                    details={"error": repr(e)},
                ) from e

            return response.content


    async def get_info(self) -> dict:
        """Fetch the tiler's runtime info (render_version, secret_hash, …) for binding."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(self.info_route)
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Tiler /info at {self.info_route} failed",
                    details={"tiler_status_code": e.response.status_code, "tiler_error": error_body(e.response)},
                ) from e

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.info_route}",
                    details={"error": repr(e)},
                ) from e

            try:
                return response.json()

            except ValueError as e:
                raise ApiError(
                    status_code=422,
                    message=f"Cannot decode {self.info_route} response",
                    details={"base_message": str(e)},
                ) from e


    async def doc_exists(self, doc_id: str) -> bool:
        """Cheaply check a document exists in the tiler (Range 0-0, no blob transfer)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/doc/{doc_id}",
                    headers={"Range": "bytes=0-0"},
                )

            except httpx.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Cannot reach {self.base_url}/doc/{doc_id}",
                    details={"error": repr(e)},
                ) from e

            if response.status_code in (200, 206):
                return True

            if response.status_code == 404:
                return False

            raise ApiError(
                status_code=503,
                message=f"Unexpected status {response.status_code} checking doc {doc_id}",
                details={"tiler_error": error_body(response)},
            )
