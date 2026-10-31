from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import httpx2
import orjson

from virag.clients.http_service import HttpService
from virag.errors import ApiError
from virag.settings import HttpSettings, TilerSettings
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


class TilerService(HttpService):
    def __init__(
        self,
        base_url: str,
        **http_options,
    ):

        super().__init__(base_url, **http_options)


    @property
    def tile_route(self):
        return f"{self.base_url}/tile"


    @property
    def tile_stream_route(self):
        return f"{self.base_url}/tile/stream"


    @property
    def get_tile_route(self):
        return f"{self.base_url}/get_tile"


    @property
    def info_route(self):
        return f"{self.base_url}/info"


    @property
    def schema_route(self):
        return f"{self.base_url}/schema/metadata"


    @property
    def docs_search_route(self):
        return f"{self.base_url}/docs/search"


    def doc_route(self, doc_id: str) -> str:
        return f"{self.base_url}/doc/{doc_id}"


    def metadata_route(self, doc_id: str) -> str:
        return f"{self.base_url}/doc/{doc_id}/metadata"


    @classmethod
    def from_settings(cls, ts: TilerSettings, hs: HttpSettings) -> TilerService:
        return cls(
            base_url=ts.base_url,
            timeout=ts.timeout_sec,
            max_concurrency=ts.max_concurrency,
            connect_timeout=ts.connect_timeout_sec,
            max_connections=ts.max_connections,
            max_retries=ts.max_retries,
            backoff_base=hs.backoff_base_sec,
            backoff_cap=hs.backoff_cap_sec,
        )


    async def check_connection(self) -> None:
        # /info, not /readyz: a tiler revising its corpus is up but not ready, and virag
        # waits for it behind its serving gate instead of refusing to start.
        await self._request("GET", self.info_route, action="the check")


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

        # An ingest is not replayed: a timed-out render may still be running upstream.
        response = await self._request(
            "POST",
            self.tile_route,
            action="the ingest",
            retry=False,
            params=params,
            data=self._ingest_form(doc_type, url, metadata),
            files=self._ingest_files(document),
        )

        return self._json(response, action="the ingest", parse=TileResult.from_manifest)


    async def tile_stream(
        self,
        doc_type: str,
        *,
        document: bytes | None = None,
        url: str | None = None,
        metadata: dict | None = None,
        force: bool = False,
    ):
        """Stream a stored ingest as NDJSON events (begin, tile*, committed | error).

        Because tiles arrive as they render, the per-read timeout — not a single
        whole-request deadline — bounds each hop, so an arbitrarily long render never
        trips `TILER_TIMEOUT_SEC`. Never replayed: events may already be consumed.
        """
        async with self._stream(
            "POST",
            self.tile_stream_route,
            action="the ingest",
            retry=False,
            params={"force": force},
            data=self._ingest_form(doc_type, url, metadata),
            files=self._ingest_files(document),
        ) as response:
            try:
                async for line in response.aiter_lines():
                    if line.strip():
                        yield orjson.loads(line)

            except httpx2.RequestError as e:
                raise ApiError(
                    status_code=503,
                    message=f"Lost {self.tile_stream_route} mid-stream during the ingest",
                    details={"error": repr(e)},
                ) from e


    async def get_tile(
        self,
        doc_id: str,
        tile_num: int,
        *,
        verify: bool = True,
        bulk: bool = False,
    ) -> bytes:
        """Regenerate one tile. `bulk` calls (self-heal, revision) are capped by
        `max_concurrency`; interactive ones (chat RAG) bypass it, so a corpus rebuild
        never starves a user's query of tiles."""
        params = {
            "doc_id": doc_id,
            "tile_num": tile_num,
            "verify": verify,
        }

        async with (self._guard() if bulk else nullcontext()):
            response = await self._request(
                "GET",
                self.get_tile_route,
                action=f"tile regeneration ({doc_id}, {tile_num})",
                params=params,
            )

        return response.content


    async def get_info(self) -> dict:
        """Fetch the tiler's runtime info (render_version, secret_hash, …) for binding."""
        response = await self._request("GET", self.info_route, action="the info fetch")

        return self._json(response, action="the info fetch")


    async def doc_exists(self, doc_id: str) -> bool:
        """Cheaply check a document exists in the tiler (Range 0-0, no blob transfer)."""
        route = self.doc_route(doc_id)

        try:
            response = await self.send("GET", route, headers={"Range": "bytes=0-0"})

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Cannot reach {route}",
                details={"error": repr(e)},
            ) from e

        if response.status_code in (200, 206):
            return True

        if response.status_code == 404:
            return False

        raise ApiError(
            status_code=503,
            message=f"Unexpected status {response.status_code} checking doc {doc_id}",
            details={"upstream_status": response.status_code, "upstream_error": error_body(response)},
        )


    async def get_schema(self) -> dict:
        """Fetch the tiler's declared metadata schema (which keys are indexed/searchable)."""
        response = await self._request("GET", self.schema_route, action="the schema fetch")

        return self._json(response, action="the schema fetch")


    async def patch_metadata(self, doc_id: str, patch: dict) -> None:
        """Shallow-merge a metadata patch onto a stored document (a null value deletes a key).

        Safe to replay: the patch sets absolute values, so applying it twice is a no-op.
        """
        await self._request(
            "PATCH",
            self.metadata_route(doc_id),
            action="the metadata patch",
            json=patch,
        )


    async def search_docs(
        self,
        *,
        filters: list[dict] | None = None,
        limit: int = 256,
        cursor: str | None = None,
    ) -> dict:
        """One page of the document search/listing: {items: [...], next_cursor}."""
        body: dict = {"limit": limit}
        if filters is not None:
            body["filters"] = filters
        if cursor is not None:
            body["cursor"] = cursor

        response = await self._request("POST", self.docs_search_route, action="search", json=body)

        return self._json(response, action="search")


    async def iter_docs(self, *, filters: list[dict] | None = None) -> list[dict]:
        """Follow the search cursor to the end and return every matching doc summary."""
        items: list[dict] = []
        cursor: str | None = None

        while True:
            page = await self.search_docs(filters=filters, cursor=cursor)
            items.extend(page.get("items", []))

            cursor = page.get("next_cursor")
            if not cursor:
                break

        return items


    @staticmethod
    def _ingest_form(doc_type: str, url: str | None, metadata: dict | None) -> dict:
        data: dict = {"doc_type": doc_type}
        if url is not None:
            data["url"] = url
        if metadata is not None:
            data["metadata"] = orjson.dumps(metadata).decode()

        return data


    @staticmethod
    def _ingest_files(document: bytes | None) -> dict | None:
        if document is None:
            return None

        return {"document": ("document", document, "application/octet-stream")}
