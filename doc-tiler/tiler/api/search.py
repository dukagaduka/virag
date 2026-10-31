"""List and/or search documents by declared metadata keys."""

from __future__ import annotations

from typing import Any, Sequence

from fastapi import APIRouter, Request

from ..errors import bad_request
from ..models import DocList, DocSummary, SearchRequest
from ..store.meta import DocRow, SearchError, SearchFilter
from .deps import service


router = APIRouter()


def summarize(rows: Sequence[DocRow]) -> list[dict[str, Any]]:
    """Project doc rows into the compact summaries returned by search/listing."""
    return [
        {
            "doc_id": r.doc_id,
            "doc_type": r.doc_type,
            "tile_count": r.tile_count,
            "complete": r.complete,
            "quarantined": r.quarantined,
            "render_version": r.render_version,
            "created_at": r.created_at,
            "source_url": r.source_url,
            "metadata": r.metadata,
        }
        for r in rows
    ]


@router.post(
    "/docs/search",
    response_model=DocList,
    responses={400: {"description": "A filter key is not declared in the metadata schema."}},
    summary="List and/or search documents by metadata.",
    operation_id="searchDocs",
)
async def search_docs(request: Request, body: SearchRequest) -> DocList:
    """List documents or filter them by declared metadata keys."""
    svc = service(request)
    filters = [SearchFilter(key=f.key, op=f.op, value=f.value) for f in body.filters]

    try:
        rows, next_cursor = await svc.store.search(filters, svc.schema, body.limit, body.cursor)

    except SearchError as exc:
        raise bad_request(str(exc), "UNDECLARED_FILTER_KEY", key=exc.key) from exc

    return DocList(
        items=[DocSummary(**item) for item in summarize(rows)], next_cursor=next_cursor
    )
