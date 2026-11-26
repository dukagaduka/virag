from typing import Any

from fastapi import FastAPI, Request

from virag.responses import OrjsonResponse


class ApiError(Exception):
    """Domain error carrying the wire status + machine-readable code."""

    # Wire status -> the machine-readable `code` in the error envelope.
    STATUS_CODES = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA",
        422: "UNPROCESSABLE",
        500: "INTERNAL_ERROR",
        502: "UPSTREAM_FAILURE",
        503: "NOT_READY",
    }


    def __init__(
        self,
        status_code: int,
        message: str,
        details: dict[str, Any] | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:

        super().__init__(message)

        self.status_code = status_code
        self.code = code or self.STATUS_CODES[status_code]
        self.message = message
        self.details = details
        self.headers = headers


    def body(self) -> dict[str, Any]:
        """Render this error as the JSON body sent to the client."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details

        return {"error": error}


def unauthorized(message: str, code: str = "UNAUTHORIZED", **details: Any) -> ApiError:
    """Build a 401 error with the Bearer challenge."""
    return ApiError(401, message, details or None, code, headers={"WWW-Authenticate": "Bearer"})


def forbidden(message: str, code: str = "FORBIDDEN", **details: Any) -> ApiError:
    """Build a 403 error."""
    return ApiError(403, message, details or None, code)


def not_found(message: str, code: str = "NOT_FOUND", **details: Any) -> ApiError:
    """Build a 404 error."""
    return ApiError(404, message, details or None, code)


def conflict(message: str, code: str = "CONFLICT", **details: Any) -> ApiError:
    """Build a 409 error."""
    return ApiError(409, message, details or None, code)


def unprocessable(message: str, code: str = "UNPROCESSABLE", **details: Any) -> ApiError:
    """Build a 422 error."""
    return ApiError(422, message, details or None, code)


def bad_gateway(message: str, code: str = "UPSTREAM_FAILURE", **details: Any) -> ApiError:
    """Build a 502 error."""
    return ApiError(502, message, details or None, code)


def not_ready(message: str, code: str = "NOT_READY", **details: Any) -> ApiError:
    """Build a 503 error."""
    return ApiError(503, message, details or None, code)


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers that render errors as the uniform `{"error": {...}}` body."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> OrjsonResponse:
        return OrjsonResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> OrjsonResponse:
        err = ApiError(status_code=500, message="Internal server error")
        return OrjsonResponse(status_code=err.status_code, content=err.body())

