# Security

Three independent controls: the **API key** (only the holder of `TILER_API_KEY`
reads or writes documents), **SSRF protection** (URL ingest can't be turned into a
request forgery), and the **non-root container** hardening. Deployment mechanics for the
last one live in [DEPLOYMENT.md](DEPLOYMENT.md); this page is the threat model and the
guarantees.

---

## 1. API key

Every route requires `Authorization: Bearer <TILER_API_KEY>` except the probes and the
metrics: `GET /healthz`, `GET /readyz` and `GET /metrics` stay open for orchestrators
and Prometheus (metrics carry low-cardinality counters only — no doc ids, names or
content). A missing or wrong key answers `401 UNAUTHORIZED` with `WWW-Authenticate:
Bearer`; keys are compared in constant time. The key is `repr`-suppressed in config so it
never lands in a log line or a stack trace. Swagger UI (`/docs`) has an **Authorize**
button for it.

An empty `TILER_API_KEY` leaves every route open, with a startup `WARNING` — acceptable
only on a network nobody untrusted can reach.

### Volume id (not a secret)

On first init the volume gets a random `volume_id`, stored in `service_meta` for life and
reported on `GET /info`. `render_version` tells you two volumes were *configured* the
same way; `volume_id` tells you they are the *same volume*, so a consumer (e.g. the RAG
gateway and its vector DB) can refuse a fresh or another volume behind the same URL. It
identifies, it does not authenticate — that is the API key's job.

## 2. SSRF protection (URL input)

Active only when `TILER_URL_FETCH_ENABLED=true` and `TILER_URL_ALLOW_PRIVATE=false`.

Validating the URL *before* navigation is not enough: it cannot stop DNS rebinding, a
redirect to an internal host, or a subresource pointing at `169.254.169.254` — Chromium
does its own DNS resolution. So the browser context is routed through an **in-process
loopback proxy** (`netguard.py`) that resolves every connection itself, applies the
policy, and dials **by validated IP** (defeating rebinding). Top-level, redirect and
subresource requests all pass through it.

Policy:

- Allow only `http` / `https`.
- Reject if any resolved IP is not globally routable (`not ip.is_global`) or is
  multicast: loopback, RFC 1918, link-local `169.254.0.0/16` (the cloud metadata
  endpoint `169.254.169.254`), CGNAT `100.64.0.0/10` (Alibaba Cloud metadata
  `100.100.100.200`), benchmarking, reserved, `::1`, `fc00::/7`, `fe80::/10`, … IPv4
  embedded in IPv6 (v4-mapped `::ffff:0:0/96`, 6to4 `2002::/16`, NAT64 `64:ff9b::/96`)
  is unwrapped and checked as IPv4.
- Re-validate across redirects; enforce size and time limits.
- Blocked target → `403`, counted in `tiler_ssrf_blocked_total`.

`TILER_URL_ALLOW_PRIVATE=true` disables all of the above — only for a trusted, isolated
network where fetching internal hosts is the point.

## 3. Non-root container

- Nothing runs as root; blobs are written by uid **10001**, baked in as a hard constant
  the entrypoint enforces (see [DEPLOYMENT.md](DEPLOYMENT.md)).
- Compose sets `no-new-privileges:true`.
- Chromium runs with the Playwright sandbox disabled (standard in containers) — the
  non-root uid + `no-new-privileges` are what contain it, so keep both.
- All 5xx from rendering are caught, so a hostile or malformed document can't crash a
  worker or the event loop; oversized uploads are rejected with `413` before rendering.
- An unexpected error answers a generic `500 INTERNAL_ERROR`; the exception text (which
  may carry paths, URLs or document content) and its traceback go to the log only.

---

## Operator checklist

- [ ] `TILER_API_KEY` set to a strong random value (e.g. `openssl rand -hex 32`), the
      same value configured on the consumer.
- [ ] `TILER_URL_ALLOW_PRIVATE=false` in any deployment reachable from untrusted input.
- [ ] `TILER_ALLOW_DELETE=false` unless deletion is genuinely required.
- [ ] `TILER_MAX_DOC_BYTES` sized to your inputs (default 64 MiB).
- [ ] Volume ownership is uid 10001; `no-new-privileges` left enabled.
