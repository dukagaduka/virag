"""Uniform error envelope matching the `Error` schema in docs/openapi.yaml."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """Domain error carrying the wire status + machine-readable code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    def body(self) -> dict[str, Any]:
        """Render this error as the JSON body sent to the client."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


def bad_request(msg: str, code: str = "BAD_REQUEST", **details: Any) -> ApiError:
    """Build a 400 error."""
    return ApiError(400, code, msg, details or None)


def forbidden(msg: str, code: str = "FORBIDDEN", **details: Any) -> ApiError:
    """Build a 403 error."""
    return ApiError(403, code, msg, details or None)


def not_found(msg: str, code: str = "NOT_FOUND", **details: Any) -> ApiError:
    """Build a 404 error."""
    return ApiError(404, code, msg, details or None)


def conflict(msg: str, code: str = "RENDER_MISMATCH", **details: Any) -> ApiError:
    """Build a 409 error."""
    return ApiError(409, code, msg, details or None)


def too_large(msg: str, code: str = "PAYLOAD_TOO_LARGE", **details: Any) -> ApiError:
    """Build a 413 error."""
    return ApiError(413, code, msg, details or None)


def unsupported_media(msg: str, code: str = "UNSUPPORTED_MEDIA", **details: Any) -> ApiError:
    """Build a 415 error."""
    return ApiError(415, code, msg, details or None)


def unprocessable(msg: str, code: str = "UNPROCESSABLE", **details: Any) -> ApiError:
    """Build a 422 error."""
    return ApiError(422, code, msg, details or None)


def bad_gateway(msg: str, code: str = "UPSTREAM_FAILURE", **details: Any) -> ApiError:
    """Build a 502 error."""
    return ApiError(502, code, msg, details or None)


def gateway_timeout(msg: str, code: str = "RENDER_TIMEOUT", **details: Any) -> ApiError:
    """Build a 504 error."""
    return ApiError(504, code, msg, details or None)


_STATUS_CODES = {
    400: "BAD_REQUEST",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA",
    416: "RANGE_NOT_SATISFIABLE",
    422: "UNPROCESSABLE",
    500: "INTERNAL_ERROR",
    502: "UPSTREAM_FAILURE",
    503: "NOT_READY",
    504: "RENDER_TIMEOUT",
}


def install_error_handlers(app: Any) -> None:
    """Register exception handlers that shape all errors into the envelope."""

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        """Serialise a domain ApiError."""
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Wrap a raw Starlette HTTP exception in the error envelope."""
        code = _STATUS_CODES.get(exc.status_code, "ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Reshape FastAPI's 422 validation error into the Error envelope."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "details": {"errors": _jsonable(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        """Turn any uncaught exception into a 500 instead of crashing the worker."""
        # Catch-all so no render failure can take down the worker.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            },
        )


def _jsonable(value: Any) -> Any:
    """Recursively coerce a value into JSON-serialisable primitives."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
