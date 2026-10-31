"""JSON response rendered with orjson — the app-wide default response class.

FastAPI's own `ORJSONResponse` is deprecated; this keeps the same idea as a thin
`JSONResponse` subclass, so error handlers and routes share one serializer.
"""

from __future__ import annotations

from typing import Any

import orjson
from fastapi.responses import JSONResponse


class OrjsonResponse(JSONResponse):
    """`JSONResponse` whose body is serialized by orjson (compact, UTF-8)."""

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)
