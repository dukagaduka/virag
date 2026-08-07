# Metrics

The service exposes Prometheus metrics at **`GET /metrics`** (text exposition format,
`Content-Type: text/plain; version=0.0.4`). Scrape it directly; there is no push
gateway.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: tiler
    metrics_path: /metrics
    static_configs:
      - targets: ["tiler:8080"]
```

All metric names are prefixed `tiler_`.

---

## Request & render throughput

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `tiler_ingest_total` | counter | `doc_type`, `stored`, `result` | Ingest requests. `result` distinguishes success from error; `stored` reflects the `store` flag. |
| `tiler_renders_total` | counter | `doc_type`, `path` | Document renders **actually performed**. A cache hit or an idempotency hit does **not** increment this — it is the anchor for the idempotency tests. `path` ∈ {`ingest` (both `POST /tile` and the `POST /tile/stream` ingest path), `get_tile`, `revise` (the `POST /revision` re-render path)}. |
| `tiler_tiles_total` | counter | `doc_type`, `path` | Tiles produced. |
| `tiler_render_seconds` | histogram | `doc_type`, `path` | Wall time of a full document render. Buckets: `0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120` s. |

`RENDERS` not moving while `INGESTS` does is the signature of dedup/idempotency working
(identical bytes → stored manifest, no re-render).

## Cache

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `tiler_cache_total` | counter | `cache`, `outcome` | Tile/page cache outcomes. `cache` ∈ {`tile`, `page`}; `outcome` ∈ {`hit`, `miss`} (plus coalesced waits). |

Hit ratio per cache: `rate(tiler_cache_total{outcome="hit"}[5m]) /
rate(tiler_cache_total[5m])`. Live per-cache stats are also on `GET /info`.

## Errors & integrity

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `tiler_errors_total` | counter | `code` | Errors returned to clients, by HTTP status `code`. |
| `tiler_hash_mismatch_total` | counter | `doc_type` | Regenerated tile hash mismatches (pixel-tier drift). Non-zero for PDF/IMG indicates a bug — they are byte-deterministic. |
| `tiler_ssrf_blocked_total` | counter | — | Requests blocked by the SSRF policy. |

## Runtime state

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `tiler_browser_recycles_total` | counter | — | Chromium process recycles (`TILER_RECYCLE_AFTER`). |
| `tiler_docs` | gauge | — | Documents currently stored. Set at startup and maintained on write. |
| `tiler_inflight_renders` | gauge | `layer` | Renders currently in flight, by execution `layer` (browser vs cpu). |

---

## Suggested alerts

| Condition | Signal |
|---|---|
| Rising client errors | `rate(tiler_errors_total{code=~"5.."}[5m]) > 0` |
| Render latency regression | `histogram_quantile(0.95, rate(tiler_render_seconds_bucket[5m]))` above baseline |
| Pixel-tier drift on deterministic types | `increase(tiler_hash_mismatch_total{doc_type=~"PDF\|IMG"}[1h]) > 0` |
| Browser saturation | `tiler_inflight_renders{layer="browser"}` pinned at `TILER_N_BROWSERS` |
| SSRF probing | `increase(tiler_ssrf_blocked_total[15m])` unexpectedly high |

Liveness/readiness are separate from metrics: `GET /healthz` (liveness, touches
nothing) and `GET /readyz` (checks DB, blob dir, Chromium pool; `503` when not ready).
