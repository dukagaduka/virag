# Security

Three independent controls: the **binding secret** (pairs a volume with one consumer),
**SSRF protection** (URL ingest can't be turned into a request forgery), and the
**non-root container** hardening. Deployment mechanics for the last one live in
[DEPLOYMENT.md](DEPLOYMENT.md); this page is the threat model and the guarantees.

---

## 1. Binding secret

`render_version` tells you two volumes were *configured* the same way; it cannot tell
you they are the *same instance*. The binding secret closes that gap so a consumer (e.g.
the RAG gateway and its vector DB) will only talk to the exact Tiler volume it was
provisioned against — a mismatched but identically-configured Tiler is refused.

**Mechanism.** `TILER_DB_SECRET` is hashed once — `secret_hash =
sha256(TILER_DB_SECRET)`, or `""` when unset — and stored in `service_meta.secret_hash`
on first init. Thereafter it behaves exactly like the geometry guard:

- **Set before first init.** The value is immutable per volume. Booting an existing
  volume with a different `TILER_DB_SECRET` is a **fatal startup error** — the process
  refuses to serve.
- **Empty is an explicit, unbindable slot.** An empty secret is stored as `""`; a volume
  initialised without one can never be bound later without a fresh volume. This is
  deliberate: "no secret" is a decision, not a default to be silently upgraded.
- **Exposure.** `secret_hash` (never the secret itself) is reported on `GET /info`, so a
  consumer can assert the pairing at connect time.

The same `TILER_DB_SECRET` is shared by the Tiler, its vector DB, and the gateway — one
source of truth. It is `repr`-suppressed in config so it never lands in a log line or a
stack trace.

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
- Reject if any resolved IP is in a private/loopback/link-local range:
  `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`
  (including `169.254.169.254`, the cloud metadata endpoint), `::1`, `fc00::/7`,
  `fe80::/10`.
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

---

## Operator checklist

- [ ] `TILER_DB_SECRET` set to a strong value **before** the volume's first boot (and
      the same value configured on the consumer/vector DB).
- [ ] `TILER_URL_ALLOW_PRIVATE=false` in any deployment reachable from untrusted input.
- [ ] `TILER_ALLOW_DELETE=false` unless deletion is genuinely required.
- [ ] `TILER_MAX_DOC_BYTES` sized to your inputs (default 64 MiB).
- [ ] Volume ownership is uid 10001; `no-new-privileges` left enabled.
