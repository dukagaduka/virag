"""What the route modules share: the auth and serving dependencies, the error table of
the OpenAPI schema, the NDJSON response of a tile stream."""

from __future__ import annotations

import hmac
import logging
from contextlib import aclosing
from typing import Any, AsyncIterator

from fastapi import Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import events
from ..errors import ApiError, not_ready, unauthorized

log = logging.getLogger("tiler.api")

# auto_error=False: a missing header answers our own 401 envelope, not FastAPI's 403.
_BEARER = HTTPBearer(auto_error=False, description="TILER_API_KEY")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Bad request."},
    401: {"description": "Missing or invalid API key (TILER_API_KEY)."},
    403: {"description": "Blocked (e.g. SSRF)."},
    404: {"description": "Not found."},
    409: {"description": "Render non-determinism, a revision already running, or a doc being ingested."},
    410: {"description": "Document quarantined by a revision."},
    413: {"description": "Payload too large."},
    415: {"description": "Content-type does not match declared doc_type."},
    422: {"description": "Corrupt document or invalid metadata."},
    502: {"description": "Upstream fetch/render failure."},
    503: {"description": "Not ready: a revision is pending or running."},
    504: {"description": "Render timeout."},
}


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """Router dependency: a request must carry `Authorization: Bearer <TILER_API_KEY>`;
    an empty TILER_API_KEY leaves the routes open."""
    expected = request.app.state.settings.api_settings.api_key

    if not expected:
        return

    if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise unauthorized("missing or invalid API key")


def require_serving(request: Request) -> None:
    """Router dependency: data routes answer 503 while the revision gate is closed."""
    revision = request.app.state.revision

    if not revision.serving:
        raise not_ready(
            "not serving: stored docs are being brought to the current render_version "
            "(see GET /revision)",
            revision=revision.status,
        )


async def ndjson_response(stream: AsyncIterator[events.Event]) -> StreamingResponse:
    """A document's tile stream (`Begin`, a `Tile` per tile, `Committed`) as an NDJSON
    response.

    The first event is awaited before the response starts, so whatever fails up to it
    keeps its HTTP status. From then on the response is already 200 and a failure goes
    in-band, as an `Error` event; the absence of `committed` is the client's signal to
    roll back.
    """
    first = await anext(stream)

    async def lines() -> AsyncIterator[bytes]:
        try:
            # Closed with the response: a client that goes away ends the work at once.
            async with aclosing(stream):
                yield first.line()

                async for event in stream:
                    yield event.line()

        except ApiError as exc:
            yield events.Error(exc.code, exc.message).line()

        except Exception as exc:  # noqa: BLE001 - the response is already 200; report in-band
            log.exception("tile stream crashed")
            yield events.Error("INTERNAL_ERROR", repr(exc)).line()

    return StreamingResponse(lines(), media_type="application/x-ndjson")
