"""The tiler as virag sees it: the client of its API, and what that API hands out.

`TilerDoc` is a document as the tiler lists it; `Begin`, `Tile` and `Committed` are the
events of a document's tile stream — an ingest and the stream of a stored doc emit the
same ones, in this order. A stream the tiler fails after `Begin` ends with an `error`
event instead of `Committed`: the client raises it, so a consumer sees typed events or
an ApiError, never a dict.

The tiler keeps one thing for virag in a document's metadata — the enroll marker
(`TilerClient.ENROLLED_AT`): 0 while the document's enroll is pending, the time it
finished afterwards. Only the client reads and writes it.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx2
import orjson

from virag.clients.http_service import HttpService
from virag.errors import ApiError
from virag.settings import HttpSettings, TilerSettings


@dataclass(frozen=True, slots=True)
class TilerDoc:
    """A document as the tiler lists it."""

    doc_id: str
    doc_type: str | None = None
    tile_count: int = 0
    render_version: str = ""
    complete: bool = True
    quarantined: bool = False
    metadata: dict = field(default_factory=dict)


    @classmethod
    def from_wire(cls, item: dict) -> TilerDoc:
        return cls(
            doc_id=item["doc_id"],
            doc_type=item.get("doc_type"),
            tile_count=item.get("tile_count") or 0,
            render_version=item.get("render_version") or "",
            complete=bool(item.get("complete", True)),
            quarantined=bool(item.get("quarantined", False)),
            metadata=item.get("metadata") or {},
        )


    @property
    def embeddable(self) -> bool:
        """Its tiles can be fetched: it is complete and not quarantined by a tiler
        revision (a quarantined doc's tiles answer 410 until it is re-enrolled)."""
        return self.complete and not self.quarantined


    @property
    def enrolled_at(self) -> int:
        """When its enroll finished; 0 while one is pending."""
        return self.metadata.get(TilerClient.ENROLLED_AT, 0)


    @property
    def enrolled(self) -> bool:
        """An enroll has finished it (not pending)."""
        return self.enrolled_at > 0


@dataclass(frozen=True, slots=True)
class Begin:
    """The first event of a tile stream: the document's identity. The tiler knows the
    doc from here on."""

    doc_id: str
    doc_type: str
    render_version: str
    source_url: str | None = None


@dataclass(frozen=True, slots=True)
class Tile:
    """One rendered tile; `data` is its JPEG in base64, `image_hash` the sha256 of the JPEG."""

    tile_num: int
    image_hash: str
    data: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class Committed:
    """The last event: the doc is in the tiler, whole, and its tiles are served."""

    doc_id: str
    tile_count: int
    doc_type: str | None = None
    render_version: str | None = None
    source_url: str | None = None
    height: int | None = None
    result: str | None = None


Event = Begin | Tile | Committed


class TilerClient(HttpService):

    # The enroll marker in a document's metadata: 0 while its enroll is pending, the
    # time it finished afterwards.
    ENROLLED_AT = "virag_enrolled_at"

    # The indexed metadata fields virag relies on: `name` for search, and the enroll
    # marker the pending docs are searched by. Both must be declared in the tiler's
    # metadata schema (deploy/service/tiler_metadata_schema.yaml).
    REQUIRED_FIELDS = {
        "name": "text",
        ENROLLED_AT: "number",
    }

    # The events of a tile stream by their name on the wire.
    EVENTS = {"begin": Begin, "tile": Tile, "committed": Committed}

    # A tile rendered by another render_version than the one it was embedded from.
    RENDER_VERSION_MISMATCH = "RENDER_VERSION_MISMATCH"

    # What the tiler answers to a document it will not take: the caller's to fix, not an
    # outage, so it is relayed with its own status. 401 is not among them — a wrong
    # TILER_API_KEY is ours.
    REFUSALS = frozenset({400, 403, 409, 413, 415, 422})


    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        **http_options,
    ):

        super().__init__(
            base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            **http_options,
        )


    @property
    def ingest_route(self):
        return f"{self.base_url}/doc"


    @property
    def readyz_route(self):
        return f"{self.base_url}/readyz"


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


    def doc_tiles_route(self, doc_id: str) -> str:
        return f"{self.base_url}/doc/{doc_id}/tiles"


    def tile_route(self, doc_id: str, tile_num: int) -> str:
        return f"{self.base_url}/doc/{doc_id}/tiles/{tile_num}"


    def metadata_route(self, doc_id: str) -> str:
        return f"{self.base_url}/doc/{doc_id}/metadata"


    @classmethod
    def from_settings(cls, ts: TilerSettings, hs: HttpSettings) -> TilerClient:
        return cls(
            base_url=ts.base_url,
            api_key=ts.api_key,
            **cls.http_options(ts, hs),
        )


    @staticmethod
    def unavailable(e: ApiError) -> bool:
        """The tiler itself failed — unreachable, or not serving (503) — as opposed to
        one tile it refused (404, 409, 410, …)."""
        status = (e.details or {}).get("upstream_status")

        return e.code == "NOT_READY" and status in (None, 503)


    @staticmethod
    def unknown_doc(e: ApiError) -> bool:
        """The tiler does not hold the document a call named (404)."""
        return (e.details or {}).get("upstream_status") == 404


    async def check_connection(self) -> None:
        # /readyz: a tiler still revising its corpus is no tiler yet — virag refuses to start.
        await self._request("GET", self.readyz_route, action="the check")

        # /readyz is open; /info is the cheapest route that tells whether the key is right.
        response = await self._request("GET", self.info_route, action="the key check", passthrough_statuses=(401,))

        if response.status_code == 401:
            raise ApiError(
                status_code=503,
                message="the tiler rejects TILER_API_KEY: set the tiler's own value",
                details={"upstream_status": 401},
            )

        await self.check_schema()


    async def check_schema(self) -> None:
        """The tiler must declare every metadata field virag depends on (`REQUIRED_FIELDS`):
        the pending docs are searched by the enroll marker, and the tiler refuses a
        search by an undeclared key — so a missing field is a misconfiguration."""
        schema = await self.get_schema()
        declared = {f.get("key"): f.get("type") for f in schema.get("fields", [])}

        for key, expected_type in self.REQUIRED_FIELDS.items():
            actual_type = declared.get(key)

            if actual_type is None:
                raise ApiError(
                    status_code=503,
                    message=f"tiler metadata schema is missing required field {key!r}; "
                            f"declare it in tiler_metadata_schema.yaml",
                )

            if actual_type != expected_type:
                raise ApiError(
                    status_code=503,
                    message=f"tiler metadata field {key!r} is declared {actual_type!r} "
                            f"but virag needs {expected_type!r}",
                )


    async def ingest_stream(
        self,
        body: AsyncIterable[bytes],
        *,
        content_type: str,
        content_length: str | None = None,
    ) -> AsyncIterator[Event]:
        """Relay an ingest form to the tiler (`POST /doc?tiles=true`) and stream the
        ingest as its events: Begin, a Tile per tile, Committed.

        `body` is the multipart form as the tiler takes it (`doc_type`, `document` |
        `url`, `metadata`), sent on as it is read — never held here; `content_type`
        carries its boundary.

        Because tiles arrive as they render, the per-read timeout — not a single
        whole-request deadline — bounds each hop, so an arbitrarily long render never
        trips `TILER_TIMEOUT_SEC`. Never replayed: the body is gone once read.

        A document the tiler refuses before the stream begins (wrong type, too large,
        invalid metadata, already being ingested, …) raises with the tiler's own status;
        an ingest it fails after `Begin` raises 502.
        """
        headers = {"Content-Type": content_type}
        if content_length is not None:
            headers["Content-Length"] = content_length

        async with self._stream(
            "POST",
            self.ingest_route,
            action="the ingest",
            retry=False,
            passthrough_statuses=self.REFUSALS,
            params={"tiles": "true"},
            content=body,
            headers=headers,
        ) as response:
            if not response.is_success:
                raise await self._refusal(response)

            async for event in self._events(response, action="the ingest"):
                yield event


    async def doc_stream(self, doc_id: str) -> AsyncIterator[Event]:
        """Stream a document the tiler holds, as the ingest of its bytes would: the same
        events, its tiles rendered again. A doc that is unknown, still being ingested or
        quarantined fails before the stream begins."""
        async with self._stream(
            "GET",
            self.doc_tiles_route(doc_id),
            action="the tile stream",
            retry=False,
        ) as response:
            async for event in self._events(response, action="the tile stream"):
                yield event


    async def get_tile(
        self,
        doc_id: str,
        tile_num: int,
        *,
        render_version: str | None = None,
        verify: bool = True,
        bulk: bool = False,
    ) -> bytes:
        """Fetch one tile, rendered by the tiler from the stored original. `bulk` calls
        (a revision's re-embed) are capped by `max_concurrency`; interactive ones (chat
        RAG) bypass it, so a corpus rebuild never starves a user's query of tiles.

        `render_version` is the one the caller embedded the tile from: a tile the tiler
        rendered by another one is refused with 409 RENDER_VERSION_MISMATCH.
        """
        async with (self._guard() if bulk else nullcontext()):
            response = await self._request(
                "GET",
                self.tile_route(doc_id, tile_num),
                action=f"the tile fetch ({doc_id}, {tile_num})",
                params={"verify": verify},
            )

        rendered_by = response.headers.get("X-Render-Version")
        if render_version is not None and rendered_by != render_version:
            raise ApiError(
                status_code=409,
                code=self.RENDER_VERSION_MISMATCH,
                message=(
                    f"tile ({doc_id}, {tile_num}) was embedded from render_version "
                    f"{render_version} but the tiler renders {rendered_by}; rebuild the "
                    f"collection with POST /revision"
                ),
                details={
                    "doc_id": doc_id,
                    "tile_num": tile_num,
                    "embedded_render_version": render_version,
                    "tiler_render_version": rendered_by,
                },
            )

        return response.content


    async def get_info(self) -> dict:
        """Fetch the tiler's runtime info (render_version, volume_id, …) for binding."""
        response = await self._request("GET", self.info_route, action="the info fetch")

        return self._json(response, action="the info fetch")


    async def render_version(self) -> str:
        """The render_version the tiler renders by. Refused (409) while the tiler is not
        serving: its docs are being revised, so their tiles are not yet the ones it
        settles on."""
        info = await self.get_info()
        tiler_revision = info.get("revision") or {}

        if not tiler_revision.get("serving", True):
            raise ApiError(
                status_code=409,
                message=f"The tiler is not serving (its revision is {tiler_revision.get('state')}); "
                        f"retry once it is",
            )

        return info.get("render_version")


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
            details=self._upstream_details(response),
        )


    async def get_schema(self) -> dict:
        """Fetch the tiler's declared metadata schema (which keys are indexed/searchable)."""
        response = await self._request("GET", self.schema_route, action="the schema fetch")

        return self._json(response, action="the schema fetch")


    async def get_metadata(self, doc_id: str) -> dict:
        """A stored document's metadata object."""
        response = await self._request("GET", self.metadata_route(doc_id), action="the metadata fetch")

        return self._json(response, action="the metadata fetch")


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


    async def mark_pending(self, doc_id: str, metadata: dict | None = None) -> None:
        """Mark a document's enroll as pending, storing `metadata` along with the mark."""
        await self.patch_metadata(doc_id, {**(metadata or {}), self.ENROLLED_AT: 0})


    async def mark_enrolled(self, doc_id: str) -> None:
        """Mark a document's enroll as finished, now."""
        await self.patch_metadata(doc_id, {self.ENROLLED_AT: int(time.time())})


    async def docs(self, *, filters: list[dict] | None = None) -> list[TilerDoc]:
        """Every document the tiler lists, or those matching the metadata `filters`."""
        docs: list[TilerDoc] = []
        cursor: str | None = None

        while True:
            page = await self._docs_page(filters=filters, cursor=cursor)
            docs.extend(TilerDoc.from_wire(item) for item in page.get("items", []))

            cursor = page.get("next_cursor")
            if not cursor:
                break

        return docs


    async def pending_docs(self) -> list[TilerDoc]:
        """The documents whose enroll is pending: begun and never finished, or running now."""
        return await self.docs(filters=[{"key": self.ENROLLED_AT, "op": "eq", "value": 0}])


    async def _docs_page(
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


    @classmethod
    async def _events(cls, response: httpx2.Response, *, action: str) -> AsyncIterator[Event]:
        """The events of an NDJSON tile stream, one per line. An `error` event — the
        tiler failed the stream after `Begin` — is raised."""
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                fields = orjson.loads(line)
                name = fields.pop("event", None)

                if name == "error":
                    raise ApiError(
                        status_code=502,
                        message=f"tiler stream error: {fields.get('message')}",
                        details={"tiler_code": fields.get("code")},
                    )

                if name in cls.EVENTS:
                    yield cls._event(cls.EVENTS[name], fields, action=action)

        except httpx2.RequestError as e:
            raise ApiError(
                status_code=503,
                message=f"Lost {response.url} mid-stream during {action}",
                details={"error": repr(e)},
            ) from e


    @staticmethod
    def _event(kind: type, fields: dict, *, action: str) -> Event:
        """One event from its fields on the wire; fields this virag does not know are
        left out."""
        try:
            return kind(**{name: fields[name] for name in kind.__dataclass_fields__ if name in fields})

        except TypeError as e:
            raise ApiError(
                status_code=502,
                message=f"tiler stream sent a malformed `{kind.__name__.lower()}` event during {action}",
                details={"base_message": str(e)},
            ) from e


    @classmethod
    async def _refusal(cls, response: httpx2.Response) -> ApiError:
        """The tiler's refusal of a document as our own error of the same status."""
        await response.aread()
        body = cls._error_body(response)
        error = body.get("error", {}) if isinstance(body, dict) else {"message": body}
        details = {"tiler_code": error.get("code")}

        if error.get("details"):
            details["tiler_details"] = error["details"]

        return ApiError(
            status_code=response.status_code,
            message=f"The tiler refused the document: {error.get('message')}",
            details=details,
        )
