"""Tiler service — fixed-geometry image tiling for a pixel-based RAG embedder."""

# Route SQLite through pysqlite3 (a modern build compiled with SQLITE_ENABLE_STAT4)
# rather than the stdlib module, whose bundled SQLite lacks STAT4. Without STAT4 the
# planner has no histograms to gauge range selectivity, so a selective numeric filter
# such as `year > 2050` ignores its metadata index and walks idx_docs_created instead.
# The swap must run before aiosqlite first imports sqlite3; doing it at package import
# time guarantees that. Falls back to the stdlib module if pysqlite3 is not installed.
import sys as _sys

try:
    import pysqlite3 as _pysqlite3
except ImportError:
    pass
else:
    _sys.modules["sqlite3"] = _pysqlite3

__version__ = "3.2.0"
