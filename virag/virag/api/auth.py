"""Bearer keys of virag's own clients.

`VIRAG_API_KEY` opens `/v1/*` — a client uses it as if it were the LLM's key; the
upstream keys stay inside virag. `VIRAG_ADMIN_API_KEY` opens `/enroll` and
`/revision*`, and `/v1/*` too. An empty key leaves its routes open (logged at startup).
"""

from __future__ import annotations

import hmac

from fastapi import Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from virag.errors import ApiError


# auto_error=False: a missing header answers our own 401 envelope, not FastAPI's 403.
_BEARER = HTTPBearer(auto_error=False, description="VIRAG_API_KEY, or VIRAG_ADMIN_API_KEY")


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """Router dependency for `/v1/*`: the user key or the admin key."""
    settings = request.app.state.settings

    if not settings.api_key:
        return

    if not _matches(credentials, settings.api_key, settings.admin_api_key):
        raise _unauthorized()


def require_admin_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> None:
    """Router dependency for `/enroll` and `/revision*`: the admin key only."""
    settings = request.app.state.settings

    if not settings.admin_api_key:
        return

    if _matches(credentials, settings.admin_api_key):
        return

    # The user key is valid, just not here.
    if _matches(credentials, settings.api_key):
        raise ApiError(status_code=403, message="this route needs VIRAG_ADMIN_API_KEY")

    raise _unauthorized()


def _matches(credentials: HTTPAuthorizationCredentials | None, *keys: str) -> bool:
    """Constant-time match of the presented key against the non-empty `keys`."""
    if credentials is None:
        return False

    presented = credentials.credentials.encode()

    return any(hmac.compare_digest(presented, key.encode()) for key in keys if key)


def _unauthorized() -> ApiError:
    return ApiError(
        status_code=401,
        message="missing or invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
