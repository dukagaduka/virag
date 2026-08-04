from __future__ import annotations

import httpx


def error_body(response: httpx.Response):
    """Return an httpx response body as parsed JSON if possible, else the raw text."""
    try:
        return response.json()

    except ValueError:
        return response.text
