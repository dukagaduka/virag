"""SQLite metadata store for documents, tiles, and indexed search.

Single connection, WAL, serialized through one lock — the workload is metadata
only (blobs live on disk), so this keeps transactional semantics trivial and
still leaves the event loop free. Everything the rest of the app needs from
persistence goes through `MetaStore`, so SQLite → Postgres stays a driver swap.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import aiosqlite

from ..meta.schema import Field, MetadataSchema, meta_expr

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
)

DDL = (
    """
    CREATE TABLE IF NOT EXISTS service_meta (
        id              INTEGER PRIMARY KEY CHECK (id = 1),
        tile_width      INTEGER NOT NULL,
        tile_height     INTEGER NOT NULL,
        tile_overlap    INTEGER NOT NULL,
        min_tile_height INTEGER NOT NULL,
        render_version  TEXT    NOT NULL,
        secret_hash     TEXT    NOT NULL DEFAULT '',
        initialized_at  INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS docs (
        doc_id         TEXT PRIMARY KEY,
        doc_type       TEXT NOT NULL,
        blob_rel_path  TEXT NOT NULL,
        source_url     TEXT,
        page_height    INTEGER NOT NULL,
        page_heights   TEXT,
        tile_count     INTEGER NOT NULL,
        mime_type      TEXT NOT NULL,
        metadata       TEXT,
        render_version TEXT NOT NULL,
        complete       INTEGER NOT NULL DEFAULT 1,
        created_at     INTEGER NOT NULL,
        updated_at     INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tiles (
        doc_id      TEXT NOT NULL,
        tile_num    INTEGER NOT NULL,
        height      INTEGER NOT NULL,
        image_hash  TEXT NOT NULL,
        PRIMARY KEY (doc_id, tile_num),
        FOREIGN KEY (doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_docs_created ON docs(created_at, doc_id)",
)

OPS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
NUMERIC_OPS = {"gt", "gte", "lt", "lte"}


class SearchError(ValueError):
    """Bad search request (maps to 400)."""

    def __init__(self, message: str, key: str | None = None):
        super().__init__(message)
        self.key = key


@dataclass(frozen=True)
class ServiceMeta:
    """The singleton service-config row: geometry plus render version."""

    tile_width: int
    tile_height: int
    tile_overlap: int
    min_tile_height: int
    render_version: str
    initialized_at: int
    secret_hash: str = ""


@dataclass(frozen=True)
class DocRow:
    """A row from the `docs` table with metadata parsed to a dict."""

    doc_id: str
    doc_type: str
    blob_rel_path: str
    source_url: str | None
    page_height: int
    tile_count: int
    mime_type: str
    metadata: dict[str, Any]
    render_version: str
    created_at: int
    updated_at: int
    complete: bool = True
    page_heights: list[int] | None = None

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "DocRow":
        """Build a DocRow from a SQLite row, decoding the metadata and page-heights JSON."""
        raw = row["metadata"]
        keys = row.keys()
        raw_ph = row["page_heights"] if "page_heights" in keys else None
        return cls(
            doc_id=row["doc_id"],
            doc_type=row["doc_type"],
            blob_rel_path=row["blob_rel_path"],
            source_url=row["source_url"],
            page_height=row["page_height"],
            tile_count=row["tile_count"],
            mime_type=row["mime_type"],
            metadata=json.loads(raw) if raw else {},
            render_version=row["render_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            complete=bool(row["complete"]),
            page_heights=json.loads(raw_ph) if raw_ph else None,
        )


@dataclass(frozen=True)
class TileRow:
    """A row from the `tiles` table: one tile's height and image hash."""

    doc_id: str
    tile_num: int
    height: int
    image_hash: str


@dataclass(frozen=True)
class SearchFilter:
    """One search predicate: a metadata key, comparison op, and value."""

    key: str
    op: str
    value: Any


def encode_cursor(created_at: int, doc_id: str) -> str:
    """Encode a `(created_at, doc_id)` pagination cursor to a token."""
    return base64.urlsafe_b64encode(f"{created_at}:{doc_id}".encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, str]:
    """Decode a pagination cursor back to `(created_at, doc_id)`."""
    try:
        pad = "=" * (-len(cursor) % 4)
        created_at, _, doc_id = base64.urlsafe_b64decode(cursor + pad).decode().partition(":")
        return int(created_at), doc_id
    except Exception as exc:
        raise SearchError(f"invalid cursor: {exc}") from exc


def build_search_query(
    filters: Sequence[SearchFilter],
    schema: MetadataSchema,
    limit: int,
    cursor: str | None,
) -> tuple[str, list[Any]]:
    """Build the search SQL. Every predicate goes through `meta_expr()` so the
    expression is textually identical to the index's, letting SQLite use it."""
    where: list[str] = []
    params: list[Any] = []

    for f in filters:
        field: Field | None = schema.get(f.key)
        if field is None:
            raise SearchError(
                f"filter key {f.key!r} is not declared in the metadata schema; "
                "only declared (indexed) keys are searchable",
                f.key,
            )
        op = (f.op or "eq").lower()
        if op not in OPS:
            raise SearchError(f"unsupported op {f.op!r}; expected one of {sorted(OPS)}", f.key)
        if op in NUMERIC_OPS and field.type != "number":
            raise SearchError(
                f"op {op!r} requires a 'number' field, but {f.key!r} is declared "
                f"{field.type!r}",
                f.key,
            )
        from ..meta.schema import MetadataViolation, coerce_filter_value  # noqa: PLC0415

        try:
            value = coerce_filter_value(f.value, field)
        except MetadataViolation as exc:
            raise SearchError(str(exc), f.key) from exc
        
        where.append(f"{meta_expr(field.key)} {OPS[op]} ?")
        params.append(value)

    if cursor:
        created_at, doc_id = decode_cursor(cursor)
        where.append("(created_at < ? OR (created_at = ? AND doc_id < ?))")
        params.extend([created_at, created_at, doc_id])

    sql = (
        "SELECT doc_id, doc_type, blob_rel_path, source_url, page_height, page_heights, "
        "tile_count, mime_type, metadata, render_version, complete, created_at, updated_at FROM docs"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC, doc_id DESC LIMIT ?"
    params.append(limit + 1)  # one extra row tells us whether a next page exists
    
    return sql, params


class MetaStore:
    """Async wrapper over the SQLite database backing docs, tiles, and search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock: Any = None

    @property
    def db(self) -> aiosqlite.Connection:
        """The open connection; raises if the store is not connected."""
        if self._db is None:
            raise RuntimeError("MetaStore is not connected")
        return self._db

    async def connect(self) -> None:
        """Open the connection and apply the startup pragmas."""
        import asyncio  # noqa: PLC0415

        self._lock = asyncio.Lock()
        connection = aiosqlite.connect(self.db_path)
        # aiosqlite runs SQLite on a plain (non-daemon) thread; if startup aborts
        # before close(), a non-daemon thread would wedge interpreter shutdown.
        connection.daemon = True
        self._db = await connection
        self._db.row_factory = aiosqlite.Row
        for pragma in PRAGMAS:
            await self._db.execute(pragma)
        await self._db.commit()

    async def close(self) -> None:
        """Close the connection if open."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_tables(self) -> None:
        """Create the base tables and indexes if they do not exist."""
        async with self._lock:
            for stmt in DDL:
                await self.db.execute(stmt)
            # Migrate volumes created before the binding secret existed: the column
            # backfills to '' (an empty slot), matching a tiler run without a secret.
            cur = await self.db.execute("PRAGMA table_info(service_meta)")
            columns = {row["name"] for row in await cur.fetchall()}
            if "secret_hash" not in columns:
                await self.db.execute(
                    "ALTER TABLE service_meta ADD COLUMN secret_hash TEXT NOT NULL DEFAULT ''"
                )
            # Migrate volumes created before streaming ingest: pre-existing docs were
            # written atomically, so they backfill to complete=1.
            cur = await self.db.execute("PRAGMA table_info(docs)")
            doc_columns = {row["name"] for row in await cur.fetchall()}
            if "complete" not in doc_columns:
                await self.db.execute(
                    "ALTER TABLE docs ADD COLUMN complete INTEGER NOT NULL DEFAULT 1"
                )
            # Migrate volumes created before PDF strip slicing: page_heights backfills to
            # NULL, i.e. "not sliced" — get_tile then falls back to page-per-tile PDF.
            if "page_heights" not in doc_columns:
                await self.db.execute("ALTER TABLE docs ADD COLUMN page_heights TEXT")
            await self.db.commit()

    async def create_metadata_indexes(self, statements: Iterable[str]) -> list[str]:
        """Create the given metadata expression indexes.

        SQLite backfills and then auto-maintains these; there is no separate
        reindex step.
        """
        created = []
        async with self._lock:
            for stmt in statements:
                await self.db.execute(stmt)
                created.append(stmt)
            await self.db.commit()
        return created

    async def drop_stale_metadata_indexes(self, keep: set[str]) -> list[str]:
        """Drop `idx_meta_*` indexes for fields that are no longer declared."""
        cur = await self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_meta_%'"
        )
        rows = await cur.fetchall()
        dropped = []
        async with self._lock:
            for row in rows:
                if row["name"] not in keep:
                    await self.db.execute(f"DROP INDEX IF EXISTS {row['name']}")
                    dropped.append(row["name"])
            if dropped:
                await self.db.commit()
        return dropped

    # ------------------------------------------------------------------ #
    # service_meta
    # ------------------------------------------------------------------ #
    async def get_service_meta(self) -> ServiceMeta | None:
        """Return the singleton service-config row, or None if uninitialized."""
        cur = await self.db.execute("SELECT * FROM service_meta WHERE id = 1")
        row = await cur.fetchone()
        if row is None:
            return None
        return ServiceMeta(
            tile_width=row["tile_width"],
            tile_height=row["tile_height"],
            tile_overlap=row["tile_overlap"],
            min_tile_height=row["min_tile_height"],
            render_version=row["render_version"],
            initialized_at=row["initialized_at"],
            secret_hash=row["secret_hash"],
        )

    async def init_service_meta(
        self,
        geometry: dict[str, int],
        render_version: str,
        secret_hash: str = "",
    ) -> ServiceMeta:
        """Insert the singleton service-config row for a fresh database."""
        now = int(time.time())
        async with self._lock:
            await self.db.execute(
                "INSERT INTO service_meta (id, tile_width, tile_height, tile_overlap, "
                "min_tile_height, render_version, secret_hash, initialized_at) VALUES (1,?,?,?,?,?,?,?)",
                (
                    geometry["tile_width"],
                    geometry["tile_height"],
                    geometry["tile_overlap"],
                    geometry["min_tile_height"],
                    render_version,
                    secret_hash,
                    now,
                ),
            )
            await self.db.commit()
        return ServiceMeta(
            render_version=render_version, initialized_at=now, secret_hash=secret_hash, **geometry
        )

    async def update_render_version(self, render_version: str) -> None:
        """Update the stored render version on the service-config row."""
        async with self._lock:
            await self.db.execute(
                "UPDATE service_meta SET render_version = ? WHERE id = 1", (render_version,)
            )
            await self.db.commit()

    # ------------------------------------------------------------------ #
    # docs / tiles
    # ------------------------------------------------------------------ #
    async def get_doc(self, doc_id: str) -> DocRow | None:
        """Return the doc row for `doc_id`, or None if absent."""
        cur = await self.db.execute("SELECT * FROM docs WHERE doc_id = ?", (doc_id,))
        row = await cur.fetchone()
        return DocRow.from_row(row) if row else None

    async def get_tiles(self, doc_id: str) -> list[TileRow]:
        """Return all tile rows for a doc, ordered by tile number."""
        cur = await self.db.execute(
            "SELECT doc_id, tile_num, height, image_hash FROM tiles WHERE doc_id = ? "
            "ORDER BY tile_num",
            (doc_id,),
        )
        return [TileRow(**dict(r)) for r in await cur.fetchall()]

    async def get_tile(self, doc_id: str, tile_num: int) -> TileRow | None:
        """Return a single tile row, or None if absent."""
        cur = await self.db.execute(
            "SELECT doc_id, tile_num, height, image_hash FROM tiles "
            "WHERE doc_id = ? AND tile_num = ?",
            (doc_id, tile_num),
        )
        row = await cur.fetchone()
        return TileRow(**dict(row)) if row else None

    async def upsert_doc(
        self,
        *,
        doc_id: str,
        doc_type: str,
        blob_rel_path: str,
        source_url: str | None,
        page_height: int,
        tile_count: int,
        mime_type: str,
        metadata: dict[str, Any] | None,
        render_version: str,
        tiles: Sequence[tuple[int, int, str]],
        page_heights: list[int] | None = None,
    ) -> DocRow:
        """Single transaction: upsert the doc row and replace its tile rows."""
        now = int(time.time())
        meta_json = json.dumps(metadata, sort_keys=True) if metadata else None
        page_heights_json = json.dumps(page_heights) if page_heights else None
        async with self._lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                await self.db.execute(
                    """
                    INSERT INTO docs (doc_id, doc_type, blob_rel_path, source_url, page_height,
                                      page_heights, tile_count, mime_type, metadata, render_version,
                                      complete, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        doc_type=excluded.doc_type,
                        blob_rel_path=excluded.blob_rel_path,
                        source_url=COALESCE(excluded.source_url, docs.source_url),
                        page_height=excluded.page_height,
                        page_heights=excluded.page_heights,
                        tile_count=excluded.tile_count,
                        mime_type=excluded.mime_type,
                        metadata=COALESCE(excluded.metadata, docs.metadata),
                        render_version=excluded.render_version,
                        complete=1,
                        updated_at=excluded.updated_at
                    """,
                    (
                        doc_id,
                        doc_type,
                        blob_rel_path,
                        source_url,
                        page_height,
                        page_heights_json,
                        tile_count,
                        mime_type,
                        meta_json,
                        render_version,
                        now,
                        now,
                    ),
                )
                await self.db.execute("DELETE FROM tiles WHERE doc_id = ?", (doc_id,))
                await self.db.executemany(
                    "INSERT INTO tiles (doc_id, tile_num, height, image_hash) VALUES (?,?,?,?)",
                    [(doc_id, num, height, image_hash) for num, height, image_hash in tiles],
                )
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise
        row = await self.get_doc(doc_id)
        assert row is not None
        return row

    async def begin_doc(
        self,
        *,
        doc_id: str,
        doc_type: str,
        blob_rel_path: str,
        source_url: str | None,
        page_height: int,
        tile_count: int,
        mime_type: str,
        metadata: dict[str, Any] | None,
        render_version: str,
    ) -> None:
        """Open a streaming ingest: (re)create the doc row as incomplete, clear old tiles.

        `complete=0` hides the doc from regeneration until `finish_doc`, so a stream
        that dies mid-way leaves a swept-able partial rather than a servable half-doc.
        """
        now = int(time.time())
        meta_json = json.dumps(metadata, sort_keys=True) if metadata else None
        async with self._lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                await self.db.execute(
                    """
                    INSERT INTO docs (doc_id, doc_type, blob_rel_path, source_url, page_height,
                                      tile_count, mime_type, metadata, render_version, complete,
                                      created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,0,?,?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        doc_type=excluded.doc_type,
                        blob_rel_path=excluded.blob_rel_path,
                        source_url=COALESCE(excluded.source_url, docs.source_url),
                        page_height=excluded.page_height,
                        tile_count=excluded.tile_count,
                        mime_type=excluded.mime_type,
                        metadata=COALESCE(excluded.metadata, docs.metadata),
                        render_version=excluded.render_version,
                        complete=0,
                        updated_at=excluded.updated_at
                    """,
                    (
                        doc_id,
                        doc_type,
                        blob_rel_path,
                        source_url,
                        page_height,
                        tile_count,
                        mime_type,
                        meta_json,
                        render_version,
                        now,
                        now,
                    ),
                )
                await self.db.execute("DELETE FROM tiles WHERE doc_id = ?", (doc_id,))
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise


    async def append_tiles(self, doc_id: str, tiles: Sequence[tuple[int, int, str]]) -> None:
        """Insert a batch of tile rows for an in-progress streaming ingest."""
        if not tiles:
            return

        async with self._lock:
            await self.db.executemany(
                "INSERT OR REPLACE INTO tiles (doc_id, tile_num, height, image_hash) VALUES (?,?,?,?)",
                [(doc_id, num, height, image_hash) for num, height, image_hash in tiles],
            )
            await self.db.commit()


    async def finish_doc(
        self,
        doc_id: str,
        *,
        tile_count: int,
        page_height: int,
        page_heights: list[int] | None = None,
    ) -> None:
        """Close a streaming ingest: fix the final counts and flip the doc to complete."""
        async with self._lock:
            await self.db.execute(
                "UPDATE docs SET tile_count = ?, page_height = ?, page_heights = ?, "
                "complete = 1, updated_at = ? WHERE doc_id = ?",
                (
                    tile_count,
                    page_height,
                    json.dumps(page_heights) if page_heights else None,
                    int(time.time()),
                    doc_id,
                ),
            )
            await self.db.commit()


    async def list_incomplete(self, older_than: int) -> list[str]:
        """Return doc_ids of incomplete ingests last touched before `older_than` (for the sweeper)."""
        cur = await self.db.execute(
            "SELECT doc_id FROM docs WHERE complete = 0 AND updated_at < ?",
            (older_than,),
        )
        return [row["doc_id"] for row in await cur.fetchall()]


    async def set_metadata(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Replace a doc's metadata JSON; the expression indexes follow automatically."""
        async with self._lock:
            await self.db.execute(
                "UPDATE docs SET metadata = ?, updated_at = ? WHERE doc_id = ?",
                (json.dumps(metadata, sort_keys=True) if metadata else None,
                 int(time.time()), doc_id),
            )
            await self.db.commit()

    async def delete_doc(self, doc_id: str) -> bool:
        """Delete a doc (cascading to its tiles); return whether it existed."""
        async with self._lock:
            cur = await self.db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
            await self.db.commit()
            return cur.rowcount > 0

    async def optimize(self) -> None:
        """Refresh planner statistics (cheap; a no-op when nothing changed)."""
        async with self._lock:
            await self.db.execute("PRAGMA optimize")
            await self.db.commit()

    async def count_docs(self) -> int:
        """Return the total number of stored documents."""
        cur = await self.db.execute("SELECT COUNT(*) AS n FROM docs")
        row = await cur.fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------ #
    async def search(
        self,
        filters: Sequence[SearchFilter],
        schema: MetadataSchema,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[DocRow], str | None]:
        """Run a filtered, paginated search and return the page plus next cursor."""
        sql, params = build_search_query(filters, schema, limit, cursor)
        cur = await self.db.execute(sql, params)
        rows = [DocRow.from_row(r) for r in await cur.fetchall()]
        next_cursor = None
        if len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1]
            next_cursor = encode_cursor(last.created_at, last.doc_id)
        return rows, next_cursor

    async def explain(self, sql: str, params: Sequence[Any]) -> list[str]:
        """Return the query plan detail lines for a SQL statement."""
        cur = await self.db.execute(f"EXPLAIN QUERY PLAN {sql}", list(params))
        return [row["detail"] for row in await cur.fetchall()]
