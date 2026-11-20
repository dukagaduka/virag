"""List and/or search documents by declared metadata keys."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..models import DocList, DocSummary, SearchRequest
from ..store.meta import SearchFilter

router = APIRouter()


@router.post(
    "/docs/search",
    response_model=DocList,
    responses={400: {"description": "A filter key is not declared in the metadata schema."}},
    summary="List and/or search documents by metadata.",
    operation_id="searchDocs",
)
async def search_docs(request: Request, body: SearchRequest) -> DocList:
    """List documents or filter them by declared metadata keys."""
    state = request.app.state
    filters = [SearchFilter(key=f.key, op=f.op, value=f.value) for f in body.filters]

    rows, next_cursor = await state.store.search(filters, state.schema, body.limit, body.cursor)

    return DocList(items=[DocSummary(**row.summary()) for row in rows], next_cursor=next_cursor)
