"""The declared metadata schema: which keys of a document's metadata are searchable,
their types, and what a metadata object must pass to be stored.

`Field.expr` is the SINGLE source of the JSON extraction expression. Index creation
and query building must both go through it: SQLite only uses an expression index
when the query's expression is textually identical to the index's.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable

import orjson
import yaml

from .errors import unprocessable

log = logging.getLogger("tiler.metadata")


class SchemaError(RuntimeError):
    """Invalid metadata schema file; fatal at startup."""


class MetadataViolation(ValueError):
    """A value that does not fit its declared field."""


    def __init__(self, message: str, key: str | None = None):
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class Field:
    """A single declared, indexed metadata field."""

    # Keys are interpolated into SQL/JSON-path text, so the grammar is deliberately tight.
    KEY_RE: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
    TYPES: ClassVar[tuple[str, ...]] = ("text", "number")
    MISSING: ClassVar[object] = object()  # what `read` returns for a key that is not there

    key: str
    type: str
    required: bool = False


    @classmethod
    def parse(cls, entry: Any, source: str) -> Field:
        """Validate one entry of a schema's `fields` into a Field, raising SchemaError."""
        if not isinstance(entry, dict):
            raise SchemaError(f"{source}: each field must be a mapping, got {entry!r}")

        key = entry.get("key")

        if not isinstance(key, str) or not cls.KEY_RE.match(key):
            raise SchemaError(
                f"{source}: field key {key!r} must match {cls.KEY_RE.pattern} "
                "(dotted identifiers only)"
            )

        ftype = entry.get("type")

        if ftype not in cls.TYPES:
            raise SchemaError(
                f"{source}: field {key!r} has type {ftype!r}; expected one of {cls.TYPES}"
            )

        return cls(key=key, type=ftype, required=bool(entry.get("required", False)))


    @property
    def expr(self) -> str:
        """The one and only JSON extraction expression for this field."""
        if not self.KEY_RE.match(self.key):
            raise SchemaError(f"invalid metadata key {self.key!r}")

        return f"json_extract(metadata, '$.{self.key}')"


    @property
    def index_name(self) -> str:
        """The SQLite index name derived from this field's key."""
        return "idx_meta_" + self.key.replace(".", "_")


    @property
    def index_statement(self) -> str:
        """`CREATE INDEX IF NOT EXISTS` over this field's expression."""
        return f"CREATE INDEX IF NOT EXISTS {self.index_name} ON docs ({self.expr})"


    def coerce(self, value: Any) -> Any:
        """The value as this field's type holds it, or MetadataViolation."""
        return self._text(value) if self.type == "text" else self._number(value)


    def read(self, metadata: Any) -> Any:
        """The value at this field's dotted key in a metadata object; MISSING if absent."""
        cur: Any = metadata

        for part in self.key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return self.MISSING

            cur = cur[part]

        return cur


    def write(self, metadata: dict[str, Any], value: Any) -> None:
        """Set the value at this field's dotted key (its parents must already exist)."""
        parts = self.key.split(".")
        cur = metadata

        for part in parts[:-1]:
            cur = cur[part]

        cur[parts[-1]] = value


    def _text(self, value: Any) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetadataViolation(
                f"field {self.key!r} is declared 'text' but got {type(value).__name__}", self.key
            )

        return str(value)


    def _number(self, value: Any) -> float | int:
        if isinstance(value, bool):
            raise MetadataViolation(f"field {self.key!r} is declared 'number' but got a boolean", self.key)

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            try:
                return float(value)

            except ValueError:
                pass

        raise MetadataViolation(
            f"field {self.key!r} is declared 'number' but got {type(value).__name__}", self.key
        )


@dataclass(frozen=True)
class MetadataSchema:
    """The declared set of searchable metadata fields."""

    version: int
    fields: tuple[Field, ...]


    @classmethod
    def load(cls, path: str | Path) -> MetadataSchema:
        """Load and parse the schema file; an absent one is a schema without fields."""
        p = Path(path)

        if not p.exists():
            log.warning(
                "metadata schema %s not found — no metadata keys are searchable", p
            )
            return cls(version=0, fields=())

        try:
            raw = yaml.safe_load(p.read_text()) or {}

        except yaml.YAMLError as exc:
            raise SchemaError(f"cannot parse {p}: {exc}") from exc

        return cls.parse(raw, source=str(p))


    @classmethod
    def parse(cls, raw: Any, source: str = "<inline>") -> MetadataSchema:
        """Validate a raw mapping into a MetadataSchema, raising SchemaError."""
        if not isinstance(raw, dict):
            raise SchemaError(f"{source}: top level must be a mapping")

        version = raw.get("version", 1)

        if not isinstance(version, int):
            raise SchemaError(f"{source}: 'version' must be an integer")

        entries = raw.get("fields") or []

        if not isinstance(entries, list):
            raise SchemaError(f"{source}: 'fields' must be a list")

        fields: dict[str, Field] = {}

        for entry in entries:
            field = Field.parse(entry, source)

            if field.key in fields:
                raise SchemaError(f"{source}: duplicate field key {field.key!r}")

            fields[field.key] = field

        return cls(version=version, fields=tuple(fields.values()))


    def get(self, key: str) -> Field | None:
        """Return the declared field for `key`, or None if undeclared."""
        return self._by_key.get(key)


    def validated(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """The metadata as it is stored: a copy with the declared fields validated and
        coerced; undeclared keys pass through untouched. Raises 422 on a violation, or
        when the object cannot be stored as JSON."""
        try:
            validated = self._coerced(metadata)

        except MetadataViolation as exc:
            raise unprocessable(str(exc), "METADATA_SCHEMA_VIOLATION", key=exc.key) from exc

        try:
            orjson.dumps(validated)

        except orjson.JSONEncodeError as exc:  # e.g. an integer beyond 64 bits
            raise unprocessable(f"metadata is not storable as JSON: {exc}", "INVALID_METADATA_JSON") from exc

        return validated


    def index_statements(self) -> Iterable[str]:
        """`CREATE INDEX IF NOT EXISTS` per declared field — the whole reindex story."""
        return [field.index_statement for field in self.fields]


    def as_dict(self) -> dict[str, Any]:
        """Serialise the schema to a plain dict."""
        return {
            "version": self.version,
            "fields": [
                {"key": f.key, "type": f.type, "required": f.required} for f in self.fields
            ],
        }


    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @property
    def _by_key(self) -> dict[str, Field]:
        """Fields indexed by key."""
        return {f.key: f for f in self.fields}


    def _coerced(self, metadata: Any) -> dict[str, Any]:
        """A copy of the metadata with every declared field coerced to its type."""
        if metadata is None:
            metadata = {}

        if not isinstance(metadata, dict):
            raise MetadataViolation("metadata must be a JSON object")

        out = copy.deepcopy(metadata)

        for field in self.fields:
            value = field.read(out)

            if value is Field.MISSING or value is None:
                if field.required:
                    raise MetadataViolation(
                        f"required metadata field {field.key!r} is missing", field.key
                    )

                continue

            coerced = field.coerce(value)

            if coerced is not value:
                field.write(out, coerced)

        return out
