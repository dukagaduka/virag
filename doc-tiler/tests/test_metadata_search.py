"""Declared metadata, expression-index usage, search and listing."""

from __future__ import annotations

import json

import pytest
from conftest import base_env, make_pdf, pdf_form, running_app

from tiler.meta.schema import meta_expr
from tiler.store.meta import SearchFilter, build_search_query


async def ingest(client, pages: int, metadata: dict | None = None):
    fields = {"metadata": json.dumps(metadata)} if metadata is not None else {}
    r = await client.post("/tile", **pdf_form(make_pdf(pages=pages), **fields))
    assert r.status_code == 200, r.text
    return r.json()["doc_id"]


async def test_schema_endpoint(client):
    body = (await client.get("/schema/metadata")).json()
    assert body["version"] == 1
    assert {f["key"] for f in body["fields"]} == {"project", "source.host", "year"}


async def test_metadata_is_stored_and_searchable(client):
    doc_id = await ingest(
        client, 1, {"project": "atlas", "year": 2021, "source": {"host": "example.com"}, "free": [1]}
    )
    stored = (await client.get(f"/doc/{doc_id}/metadata")).json()
    assert stored == {
        "project": "atlas",
        "year": 2021,
        "source": {"host": "example.com"},
        "free": [1],  # undeclared keys are stored verbatim
    }

    found = (await client.post("/docs/search", json={"filters": [
        {"key": "project", "op": "eq", "value": "atlas"}
    ]})).json()
    assert [i["doc_id"] for i in found["items"]] == [doc_id]
    assert found["items"][0]["metadata"]["free"] == [1]

    dotted = (await client.post("/docs/search", json={"filters": [
        {"key": "source.host", "value": "example.com"}
    ]})).json()
    assert [i["doc_id"] for i in dotted["items"]] == [doc_id]


async def test_numeric_ops_and_and_combination(client):
    a = await ingest(client, 1, {"project": "atlas", "year": 1999})
    b = await ingest(client, 2, {"project": "atlas", "year": 2021})
    await ingest(client, 3, {"project": "borealis", "year": 2030})

    newer = (await client.post("/docs/search", json={"filters": [
        {"key": "year", "op": "gt", "value": 2000}
    ]})).json()
    assert {i["doc_id"] for i in newer["items"]} == {b, await _doc_of(client, "borealis")}

    both = (await client.post("/docs/search", json={"filters": [
        {"key": "project", "value": "atlas"},
        {"key": "year", "op": "lt", "value": 2000},
    ]})).json()
    assert [i["doc_id"] for i in both["items"]] == [a]


async def _doc_of(client, project: str) -> str:
    res = (await client.post("/docs/search", json={"filters": [
        {"key": "project", "value": project}
    ]})).json()
    return res["items"][0]["doc_id"]


async def test_undeclared_filter_key_is_400(client):
    await ingest(client, 1, {"nope": "x"})
    r = await client.post("/docs/search", json={"filters": [{"key": "nope", "value": "x"}]})
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "UNDECLARED_FILTER_KEY"
    assert "nope" in body["message"]


async def test_numeric_op_on_text_field_is_400(client):
    r = await client.post("/docs/search", json={"filters": [
        {"key": "project", "op": "gt", "value": 5}
    ]})
    assert r.status_code == 400


async def test_empty_filters_list_all_newest_first(client):
    ids = [await ingest(client, i + 1, {"project": f"p{i}"}) for i in range(3)]
    listed = (await client.post("/docs/search", json={})).json()
    assert len(listed["items"]) == 3
    assert set(i["doc_id"] for i in listed["items"]) == set(ids)
    created = [i["created_at"] for i in listed["items"]]
    assert created == sorted(created, reverse=True)


async def test_cursor_pagination(client):
    for i in range(5):
        await ingest(client, i + 1, {"project": "page"})
    page1 = (await client.post("/docs/search", json={"limit": 2})).json()
    assert len(page1["items"]) == 2 and page1["next_cursor"]

    page2 = (await client.post("/docs/search", json={"limit": 2, "cursor": page1["next_cursor"]})).json()
    assert len(page2["items"]) == 2

    page3 = (await client.post("/docs/search", json={"limit": 2, "cursor": page2["next_cursor"]})).json()
    assert len(page3["items"]) == 1
    assert page3["next_cursor"] is None

    seen = [i["doc_id"] for p in (page1, page2, page3) for i in p["items"]]
    assert len(set(seen)) == 5


async def _populate(store, rows: int = 400) -> None:
    """Insert rows straight into SQLite — the planner needs a realistic table."""
    async with store._lock:
        await store.db.executemany(
            "INSERT INTO docs (doc_id, doc_type, blob_rel_path, source_url, page_height, "
            "tile_count, mime_type, metadata, render_version, created_at, updated_at) "
            "VALUES (?,'PDF','x',NULL,0,1,'application/pdf',?,'v',?,?)",
            [
                (
                    f"{i:064d}",
                    json.dumps(
                        {
                            "project": f"p{i % 40}",
                            "year": 1900 + (i % 160),
                            "source": {"host": f"h{i % 25}.example.com"},
                        }
                    ),
                    1_700_000_000 + i,
                    1_700_000_000 + i,
                )
                for i in range(rows)
            ],
        )
        await store.db.execute("ANALYZE")
        await store.db.commit()


async def test_expression_index_is_used(client):
    """A declared-key filter is answered through idx_meta_*, not a table scan."""
    store = client.service.store
    await _populate(store)

    for key, op, value, index in (
        ("project", "eq", "p7", "idx_meta_project"),
        ("source.host", "eq", "h3.example.com", "idx_meta_source_host"),
        ("year", "gt", 2050, "idx_meta_year"),  # selective numeric range
        ("year", "lte", 1905, "idx_meta_year"),
    ):
        sql, params = build_search_query(
            [SearchFilter(key=key, op=op, value=value)], client.service.schema, 50, None
        )
        plan = " ".join(await store.explain(sql, params))
        assert index in plan, f"{key} {op}: {plan}"
        assert "SEARCH docs" in plan, plan
        assert "SCAN docs" not in plan, plan


async def test_ne_is_answered_by_an_ordered_scan_not_a_table_scan(client):
    """`<>` is not a seekable B-tree constraint in any SQL engine, so `ne` walks the
    created_at index instead. It stays cheap because a `ne` predicate matches almost
    every row, so LIMIT is satisfied immediately (see README)."""
    store = client.service.store
    await _populate(store, rows=200)
    sql, params = build_search_query(
        [SearchFilter(key="project", op="ne", value="p7")], client.service.schema, 50, None
    )
    plan = " ".join(await store.explain(sql, params))
    assert "idx_docs_created" in plan, plan
    assert "SCAN docs USING INDEX" in plan, plan  # index-ordered, never a raw table scan

    result = await client.post(
        "/docs/search", json={"filters": [{"key": "project", "op": "ne", "value": "p7"}], "limit": 5}
    )
    assert result.status_code == 200
    assert all(i["metadata"]["project"] != "p7" for i in result.json()["items"])


async def test_search_never_scans_the_table_itself(client):
    """Even when the planner prefers the created_at index, `docs` is never read
    row-by-row without an index — there is no fallback full scan."""
    store = client.service.store
    await _populate(store, rows=200)
    for filters in ([], [SearchFilter(key="year", op="gt", value=1899)]):
        sql, params = build_search_query(filters, client.service.schema, 50, None)
        plan = " ".join(await store.explain(sql, params))
        assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan


def test_query_builder_uses_the_shared_expression():
    sql, _ = build_search_query(
        [SearchFilter(key="source.host", op="eq", value="x")],
        _schema(), 10, None,
    )
    assert meta_expr("source.host") in sql


def _schema():
    from tiler.meta.schema import parse_schema

    return parse_schema({"version": 1, "fields": [{"key": "source.host", "type": "text"}]})


async def test_validation_rejects_bad_types_and_missing_required(client):
    r = await client.post("/tile", **pdf_form(make_pdf(1), metadata=json.dumps({"year": "abc"})))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "METADATA_SCHEMA_VIOLATION"

    r = await client.post("/tile", **pdf_form(make_pdf(1), metadata="{not json"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_METADATA_JSON"

    r = await client.post("/tile", **pdf_form(make_pdf(1), metadata=json.dumps(["a"])))
    assert r.status_code == 422


async def test_required_field(tmp_path):
    schema = """
version: 2
fields:
  - key: project
    type: text
    required: true
"""
    env = base_env(tmp_path)  # writes the default schema first
    path = tmp_path / "required_schema.yaml"
    path.write_text(schema)
    env["TILER_METADATA_SCHEMA_PATH"] = str(path)
    async with running_app(env) as (client, _):
        r = await client.post("/tile", **pdf_form(make_pdf(1), metadata=json.dumps({"x": 1})))
        assert r.status_code == 422
        r = await client.post("/tile", **pdf_form(make_pdf(1), metadata=json.dumps({"project": "p"})))
        assert r.status_code == 200


async def test_put_and_patch_update_the_index_immediately(client):
    doc_id = await ingest(client, 1, {"project": "old", "year": 2000})

    put = await client.put(f"/doc/{doc_id}/metadata", json={"project": "new", "year": 2024})
    assert put.status_code == 200 and put.json()["project"] == "new"

    assert await _search_count(client, "project", "old") == 0
    assert await _search_count(client, "project", "new") == 1  # no manual reindex

    patch = await client.patch(f"/doc/{doc_id}/metadata", json={"project": "patched", "extra": 5})
    assert patch.json() == {"project": "patched", "year": 2024, "extra": 5}
    assert await _search_count(client, "project", "patched") == 1

    deleted = await client.patch(f"/doc/{doc_id}/metadata", json={"project": None})
    assert "project" not in deleted.json()
    assert await _search_count(client, "project", "patched") == 0


async def test_patch_validates(client):
    doc_id = await ingest(client, 1, {"project": "p"})
    r = await client.patch(f"/doc/{doc_id}/metadata", json={"year": "not a number"})
    assert r.status_code == 422


async def test_metadata_endpoints_404(client):
    missing = "0" * 64
    assert (await client.get(f"/doc/{missing}/metadata")).status_code == 404
    assert (await client.put(f"/doc/{missing}/metadata", json={})).status_code == 404
    assert (await client.patch(f"/doc/{missing}/metadata", json={})).status_code == 404


async def test_metadata_ignored_when_store_false(client):
    r = await client.post(
        "/tile",
        params={"store": "false"},
        **pdf_form(make_pdf(1), metadata=json.dumps({"year": "invalid-but-ignored"})),
    )
    assert r.status_code == 200
    assert r.json()["stored"] is False


async def _search_count(client, key: str, value) -> int:
    res = (await client.post("/docs/search", json={"filters": [{"key": key, "value": value}]})).json()
    return len(res["items"])


@pytest.mark.parametrize("bad_key", ["a b", "1abc", "a.b.", "a-b", "a')--"])
def test_schema_rejects_injectable_keys(bad_key):
    from tiler.meta.schema import SchemaError, parse_schema

    with pytest.raises(SchemaError):
        parse_schema({"version": 1, "fields": [{"key": bad_key, "type": "text"}]})
