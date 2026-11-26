from __future__ import annotations

import logging
from typing import AsyncIterator

import orjson
from fastapi import APIRouter, Request

from virag.errors import unprocessable
from virag.utils import MultipartFields


log = logging.getLogger(__name__)

router = APIRouter()

# The form /enroll takes — the tiler's own ingest form, which virag relays as it is. The
# handler reads the body as a stream, so FastAPI cannot describe it by itself.
ENROLL_FORM = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["doc_type"],
                    "properties": {
                        "doc_type": {"type": "string", "enum": ["PDF", "URL", "MHTML", "IMG"]},
                        "document": {
                            "type": "string",
                            "format": "binary",
                            "description": "The document's bytes. Required unless doc_type=URL.",
                        },
                        "url": {"type": "string", "description": "The URL, for doc_type=URL."},
                        "metadata": {
                            "type": "string",
                            "description": "A JSON object as a string; should include `name`. "
                                           "Send it before `document`, so a missing name is refused "
                                           "before the document is uploaded.",
                        },
                    },
                },
            },
        },
    },
}


@router.post(
    "/enroll",
    summary="Tile a document, embed its tiles, and upsert them into the vector store",
    operation_id="enrollDocument",
    openapi_extra=ENROLL_FORM,
)
async def enroll(request: Request, allow_missing_name: bool = False):

    enroller = request.app.state.enroller
    tiler = request.app.state.tiler

    upload = EnrollUpload(request, allow_missing_name=allow_missing_name)
    log.info("enroll request: %s bytes", request.headers.get("content-length", "?"))

    events = tiler.ingest_stream(
        upload.chunks(),
        content_type=request.headers["content-type"],
        content_length=request.headers.get("content-length"),
    )
    summary = await enroller.enroll(events, metadata=upload.metadata)

    if not upload.has_name:
        log.warning("no metadata.name; doc_id=%s goes by its content hash", summary["doc_id"])

    return summary


class EnrollUpload:
    """The body of an /enroll request on its way to the tiler.

    The form is the tiler's own (`doc_type`, `document` | `url`, `metadata`), so the body
    is relayed as it arrives, chunk by chunk: virag keeps nothing of the document, in
    memory or on disk, and whatever is wrong with it is the tiler's to refuse. The one
    field read on the way is `metadata` — for the name policy, checked as soon as the
    field has gone by, and for the tiler's metadata patch at `begin`.
    """

    def __init__(self, request: Request, *, allow_missing_name: bool):

        self.request = request
        self.allow_missing_name = allow_missing_name

        try:
            self.form = MultipartFields(request.headers.get("content-type", ""))

        except ValueError as e:
            raise unprocessable(f"/enroll takes multipart/form-data: {e}") from e

        self._name_checked = False


    async def chunks(self) -> AsyncIterator[bytes]:
        """The body as it arrives. Ends the relay with 422 when the form is malformed or
        the document goes unnamed — before the chunk that showed it is sent on."""
        async for chunk in self.request.stream():
            try:
                self.form.feed(chunk)

            except ValueError as e:
                raise unprocessable(f"malformed multipart body: {e}") from e

            if "metadata" in self.form.fields:
                self.require_name()

            yield chunk

        self.require_name()                                     # a form without `metadata` at all


    def metadata(self) -> dict | None:
        """The `metadata` field as an object; None while it has not gone by, or is absent."""
        raw = self.form.fields.get("metadata")

        if raw is None:
            return None

        try:
            metadata = orjson.loads(raw)

        except (ValueError, TypeError) as e:
            raise unprocessable("`metadata` must be a valid JSON object", base_message=str(e)) from e

        if not isinstance(metadata, dict):
            raise unprocessable("`metadata` must be a JSON object")

        return metadata


    @property
    def has_name(self) -> bool:
        """Whether the metadata carries a non-empty string `name`."""
        name = (self.metadata() or {}).get("name")

        return isinstance(name, str) and name.strip() != ""


    def require_name(self) -> None:
        """Service policy: every enrolled document is named. The opt-out is explicit so
        the caller knowingly diverges — it names the document by its content hash."""
        if self._name_checked:
            return

        self._name_checked = True

        if not self.has_name and not self.allow_missing_name:
            raise unprocessable(
                "Each enrolled document must include metadata.name. "
                "Set allow_missing_name=true to fall back to the document content hash.",
            )
