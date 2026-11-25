"""Pydantic models mirroring the schemas in docs/openapi.yaml."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field



class DocType(StrEnum):
    """Declared document type. A stored doc is never URL: a URL ingest is stored as MHTML."""

    PDF = "PDF"
    URL = "URL"
    MHTML = "MHTML"
    IMG = "IMG"


SearchOp = Literal["eq", "ne", "gt", "gte", "lt", "lte"]


class IngestResult(BaseModel):
    """Result of `POST /doc`: the stored doc's identity and size, without its tiles."""

    doc_id: str
    doc_type: DocType
    source_url: str | None = None
    render_version: str
    height: int
    tile_count: int
    result: Literal["rendered", "idempotent"]


class DocSummary(BaseModel):
    """A document as listed in search results."""

    doc_id: str
    doc_type: DocType
    tile_count: int
    complete: bool = True
    quarantined: bool = False
    render_version: str | None = None
    created_at: int
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocList(BaseModel):
    """A page of document summaries with an optional pagination cursor."""

    items: list[DocSummary]
    next_cursor: str | None = None


class SearchFilter(BaseModel):
    """A single metadata filter predicate."""

    key: str
    op: SearchOp = "eq"
    value: str | float | int


class SearchRequest(BaseModel):
    """A search/list request: AND-combined filters plus pagination."""

    filters: list[SearchFilter] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None


class MetadataFieldModel(BaseModel):
    """A declared, indexed metadata field."""

    key: str
    type: Literal["text", "number"]
    required: bool = False


class MetadataSchemaModel(BaseModel):
    """The declared metadata schema exposed to clients."""

    version: int
    fields: list[MetadataFieldModel]
