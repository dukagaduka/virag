"""Declared metadata schema, `meta_expr()` and write-time validation.

`meta_expr()` is the SINGLE source of the JSON extraction expression. Index
creation and query building must both go through it: SQLite only uses an
expression index when the query's expression is textually identical to the
index's.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

log = logging.getLogger("tiler.schema")

# Keys are interpolated into SQL/JSON-path text, so the grammar is deliberately tight.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
FIELD_TYPES = ("text", "number")


class SchemaError(RuntimeError):
    """Invalid metadata schema file; fatal at startup."""


class MetadataViolation(ValueError):
    """Metadata rejected at write time (maps to 422)."""

    def __init__(self, message: str, key: str | None = None):
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class Field:
    """A single declared, indexed metadata field."""

    key: str
    type: str
    required: bool = False

    @property
    def index_name(self) -> str:
        """The SQLite index name derived from this field's key."""
        return "idx_meta_" + self.key.replace(".", "_")


@dataclass(frozen=True)
class MetadataSchema:
    """The declared set of searchable metadata fields."""

    version: int
    fields: tuple[Field, ...]

    def get(self, key: str) -> Field | None:
        """Return the declared field for `key`, or None if undeclared."""
        return self._by_key.get(key)

    @property
    def _by_key(self) -> dict[str, Field]:
        """Fields indexed by key."""
        return {f.key: f for f in self.fields}

    def as_dict(self) -> dict[str, Any]:
        """Serialise the schema to a plain dict."""
        return {
            "version": self.version,
            "fields": [
                {"key": f.key, "type": f.type, "required": f.required} for f in self.fields
            ],
        }


def meta_expr(key: str) -> str:
    """The one and only JSON extraction expression for a metadata key."""
    if not _KEY_RE.match(key):
        raise SchemaError(f"invalid metadata key {key!r}")
    return f"json_extract(metadata, '$.{key}')"


EMPTY_SCHEMA = MetadataSchema(version=0, fields=())


def load_schema(path: str | Path) -> MetadataSchema:
    """Load and parse the schema file, returning EMPTY_SCHEMA if absent."""
    p = Path(path)
    if not p.exists():
        log.warning(
            "metadata schema %s not found — no metadata keys are searchable", p
        )
        return EMPTY_SCHEMA
    
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SchemaError(f"cannot parse {p}: {exc}") from exc
    
    return parse_schema(raw, source=str(p))


def parse_schema(raw: Any, source: str = "<inline>") -> MetadataSchema:
    """Validate a raw mapping into a MetadataSchema, raising SchemaError."""
    if not isinstance(raw, dict):
        raise SchemaError(f"{source}: top level must be a mapping")
    version = raw.get("version", 1)
    if not isinstance(version, int):
        raise SchemaError(f"{source}: 'version' must be an integer")
    entries = raw.get("fields") or []
    if not isinstance(entries, list):
        raise SchemaError(f"{source}: 'fields' must be a list")

    fields: list[Field] = []
    seen: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            raise SchemaError(f"{source}: each field must be a mapping, got {entry!r}")
        
        key = entry.get("key")
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise SchemaError(
                f"{source}: field key {key!r} must match {_KEY_RE.pattern} "
                "(dotted identifiers only)"
            )
        
        if key in seen:
            raise SchemaError(f"{source}: duplicate field key {key!r}")
        
        ftype = entry.get("type")
        if ftype not in FIELD_TYPES:
            raise SchemaError(
                f"{source}: field {key!r} has type {ftype!r}; expected one of {FIELD_TYPES}"
            )
        
        required = bool(entry.get("required", False))
        seen.add(key)
        fields.append(Field(key=key, type=ftype, required=required))

    return MetadataSchema(version=version, fields=tuple(fields))


# --------------------------------------------------------------------------- #
# Write-time validation
# --------------------------------------------------------------------------- #
_MISSING = object()


def _dig(obj: Any, key: str) -> Any:
    """Follow a dotted key into nested dicts; return _MISSING if absent."""
    cur: Any = obj
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]

    return cur


def _set_path(obj: dict[str, Any], key: str, value: Any) -> None:
    """Set the value at a dotted key path (parents must already exist)."""
    parts = key.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def coerce_text(value: Any, key: str) -> str:
    """Coerce a value to a 'text' field's string, or raise MetadataViolation."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetadataViolation(
            f"field {key!r} is declared 'text' but got {type(value).__name__}", key
        )
    
    return str(value)


def coerce_number(value: Any, key: str) -> float | int:
    """Coerce a value to a 'number' field, or raise MetadataViolation."""
    if isinstance(value, bool):
        raise MetadataViolation(f"field {key!r} is declared 'number' but got a boolean", key)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise MetadataViolation(
        f"field {key!r} is declared 'number' but got {type(value).__name__}", key
    )


def validate_metadata(metadata: Any, schema: MetadataSchema) -> dict[str, Any]:
    """Validate + coerce declared fields. Undeclared keys pass through untouched."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise MetadataViolation("metadata must be a JSON object")

    out = _deep_copy(metadata)

    for field in schema.fields:
        value = _dig(out, field.key)
        
        if value is _MISSING or value is None:
            if field.required:
                raise MetadataViolation(
                    f"required metadata field {field.key!r} is missing", field.key
                )
            continue

        coerced = (
            coerce_text(value, field.key)
            if field.type == "text"
            else coerce_number(value, field.key)
        )
        if coerced is not value:
            _set_path(out, field.key, coerced)

    return out


def coerce_filter_value(value: Any, field: Field) -> Any:
    """Coerce a search filter value to its declared field type."""
    if field.type == "text":
        return coerce_text(value, field.key)
    return coerce_number(value, field.key)


def _deep_copy(value: Any) -> Any:
    """Recursively copy nested dicts/lists so callers cannot mutate the input."""
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def index_statements(schema: MetadataSchema) -> Iterable[str]:
    """`CREATE INDEX IF NOT EXISTS` per declared field — the whole reindex story."""
    for field in schema.fields:
        yield f"CREATE INDEX IF NOT EXISTS {field.index_name} ON docs ({meta_expr(field.key)})"
