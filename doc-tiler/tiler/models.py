"""Pydantic models mirroring the schemas in docs/openapi.yaml."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocTypeIn = Literal["PDF", "URL", "MHTML", "IMG"]
StoredDocType = Literal["PDF", "MHTML", "IMG"]
SearchOp = Literal["eq", "ne", "gt", "gte", "lt", "lte"]


class TileMeta(BaseModel):
    """One tile's metadata, with optional base64 JPEG bytes."""

    tile_num: int
    width: int
    height: int
    image_hash: str
    data: str | None = None


class IngestManifest(BaseModel):
    """Result of an ingest: doc identity plus the produced tiles."""

    doc_id: str
    doc_type: StoredDocType
    source_url: str | None = None
    render_version: str
    stored: bool
    page_height: int
    tile_count: int
    tiles: list[TileMeta]


class DocSummary(BaseModel):
    """A document as listed in search results."""

    doc_id: str
    doc_type: StoredDocType
    tile_count: int
    created_at: int
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocList(BaseModel):
    """A page of document summaries with an optional pagination cursor."""

    items: list[DocSummary]
    next_cursor: str | None = None


class SearchFilterModel(BaseModel):
    """A single metadata filter predicate."""

    key: str
    op: SearchOp = "eq"
    value: str | float | int


class SearchRequest(BaseModel):
    """A search/list request: AND-combined filters plus pagination."""

    filters: list[SearchFilterModel] = Field(default_factory=list)
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


class ErrorDetail(BaseModel):
    """The inner error object of the error envelope."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Top-level error response wrapper."""

    error: ErrorDetail
