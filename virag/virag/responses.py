from __future__ import annotations

from typing import Any

import orjson
from fastapi.responses import JSONResponse


class OrjsonResponse(JSONResponse):
    """`JSONResponse` serialized by orjson — the app-wide default response class.

    FastAPI's own `ORJSONResponse` is deprecated; this thin subclass keeps routes and
    error handlers on one serializer.
    """

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)
