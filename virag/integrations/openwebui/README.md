# virag × Open WebUI

A ~150-line [Open WebUI](https://github.com/open-webui/open-webui) **Pipe function** that
puts virag's pixel-RAG behind Open WebUI's UI **and shows the retrieved tiles as native
citations** — the one thing a stock OpenAI-compatible frontend can't do on its own.

## Why a plugin at all

Any OpenAI-compatible frontend can *run* virag's RAG: it will happily forward a `rag`
field in the request body (Open WebUI lets you add arbitrary params). What it **won't** do
is surface `include_sources` — because virag returns sources as a **non-standard
extension** (a top-level `"sources"` key when non-streaming, or a synthetic
`chat.completion.chunk` with empty `choices` and a `sources` key right before
`data: [DONE]` when streaming). A generic frontend parses these as OpenAI schema and drops
them silently.

This Pipe function makes the HTTP call to virag itself, so it sees that raw stream, and
re-emits each source through Open WebUI's **citation event** — which the UI renders as
clickable reference chips under the answer. See `../../docs/API.md` → *Sources*.

## What you get

The function is a **manifold**: it registers **two models** in the normal model selector.

| Model in the dropdown | Behavior |
|---|---|
| `virag (pixel-RAG)` | Retrieval on, `include_sources: true` → answer + citation chips. |
| `virag (no RAG)` | Plain passthrough to the upstream VLM, no retrieval. |

Switching is just picking one or the other from the model dropdown — per chat, or mid-chat.

## Prerequisites

- A running virag gateway reachable from the Open WebUI container.
- At least one document enrolled (`POST /enroll`), otherwise retrieval returns nothing and
  no citations appear — that's expected, not a bug.
- The upstream VLM model name virag should call (the `MODEL` valve, e.g. `my-vlm`).

## Install (on the running frontend — no restart)

Open WebUI functions live in the database, not the image. You add this one entirely from
the browser; it goes live on Save, and the `requirements: httpx` line in the file header is
auto-installed.

1. **Admin Panel → Functions → `+` (New Function).**
2. Paste the entire contents of [`virag_pipe.py`](./virag_pipe.py).
3. Give it a name (e.g. *virag pixel-RAG*) and **Save**.
4. Make sure the function is **enabled** (toggle in the Functions list).
5. Click the **gear** on the function to set its **Valves** (see below).

That's it — open a new chat and the two `virag (…)` models are in the selector.

### Valves

| Valve | Default | Meaning |
|---|---|---|
| `VIRAG_BASE_URL` | `http://virag:8080` | Gateway base URL, no trailing slash. |
| `VIRAG_API_KEY` | *(empty)* | Optional bearer token if the gateway is protected. |
| `MODEL` | `my-vlm` | Upstream VLM name virag forwards to. |
| `TOP_N` | `4` | Tiles to retrieve. Each tile is a page image ≈ a few hundred vision tokens; keep it modest — more tiles = a bigger, slower request and a longer prefill. |
| `SCORE_THRESHOLD` | `0.25` | Drop Qdrant hits below this score. |
| `FRAMING_TEXT` | *(empty)* | Prepended to injected tiles; blank = virag's server default. |
| `SHOW_STATUS` | `true` | Show a "Retrieving…" / "Waiting for the model…" status line during RAG. |
| `HEARTBEAT_SEC` | `10` | While waiting for the first token, re-emit a status this often (seconds) so a long prefill doesn't leave the stream silent and trip an idle disconnect. `0` disables. |

## Set a Task Model (do this early)

After the first message, Open WebUI fires **background tasks** — chat-title generation,
tag generation, autocomplete — and by default it routes them through the **currently
selected model**. If that's `virag (pixel-RAG)`, every new chat silently runs a *second*
full retrieval + VLM round-trip just to title the chat, and any upstream hiccup surfaces
as a scary traceback in the Open WebUI logs (`generate_title … Exception occurred`).

Point those tasks at a cheap, plain model instead:

**Admin Panel → Settings → Interface → Set Task Model** → choose a lightweight non-RAG
model (e.g. virag added as a plain OpenAI connection, or any small local model). Or turn
**Title/Tags Generation** off entirely there.

The pipe is defensive regardless — if virag's upstream stream ends early it keeps the
partial answer and closes cleanly instead of throwing — but keeping RAG out of the task
path is faster and cheaper.

## Networking

`VIRAG_BASE_URL` is resolved **from inside the Open WebUI container**, not your laptop:

- **Same Docker network** as virag → use the service name: `http://virag:8080`. Put Open
  WebUI on virag's compose network (or vice versa) so the name resolves.
- **Open WebUI elsewhere, virag port published on the host** → `http://host.docker.internal:8080`
  (add `--add-host=host.docker.internal:host-gateway` on Linux if needed).

## How sources show up

After the answer streams in, each retrieved tile appears as a citation chip. The chip's
preview carries `name`, `doc_id`, `tile_num`, `score` (from virag's `sources` entries).
Citations render **text**, so the chip shows tile metadata rather than the image itself —
enough to see *which* document and page the model was grounded on.

> **Want the tile as an actual image?** Fetch it through virag's proxy —
> `GET /tiler/get_tile?doc_id=…&tile_num=…` — and inline it via an `image` event or a
> markdown image in the stream. That's a deliberate extension beyond this minimal plugin.

## Troubleshooting

- **No citations appear.** (a) virag logged `rag retrieved: 0 hits` / `rag injected: 0
  tiles` — retrieval found nothing: the queried collection is empty, isn't the one you
  enrolled into, or everything scored below `SCORE_THRESHOLD`. Enroll docs / lower the
  threshold. (b) Your Open WebUI build uses the newer event type: in `virag_pipe.py`
  change `"type": "citation"` to `"type": "source"`.
- **Answer cut off + `RemoteProtocolError` / `ReadTimeout` in logs.** The upstream VLM
  dropped the stream mid-generation (virag surfaces it as a read timeout, capped by
  `VLM_TIMEOUT_SEC`). Not a plugin bug — check the VLM's health/stability; the pipe keeps
  whatever streamed before the break.
- **Stuck on "Retrieving…"/"Waiting for the model…" then silently dies, no error anywhere,
  VLM logs `aborted`.** The model's *prefill* (time-to-first-token) is long — common with
  several page-image tiles — and the delivery channel idled out during the silence. The
  `HEARTBEAT_SEC` keep-alive is meant to prevent exactly this; if it still happens, lower
  `HEARTBEAT_SEC` (e.g. 5) and/or `TOP_N` so the first token arrives sooner.
- **`virag error 422`.** RAG on with no user text to build a query from, or a bad `rag`
  field — check the message content and valve types.
- **`virag error 502` / connection errors.** `VIRAG_BASE_URL` isn't reachable from the
  container — see *Networking*. Verify with `GET /ready` from inside the OWUI container.
- **Model not in the dropdown.** Function disabled, or a paste/syntax error — re-check the
  Functions list toggle and the editor's error banner.

## Alternative: Filter (inlet-only)

If you'd rather keep using virag as a normal **Connection** (Settings → Connections → an
OpenAI-API endpoint pointing at virag) and just toggle RAG on the base model, a **Filter**
function's `inlet` can inject the `rag` field into the request body. Note the limitation:
a Filter's `outlet` runs on the already-parsed response, where the non-standard `sources`
has **already been dropped** — so a Filter can turn RAG on but **cannot** show sources.
For citations you need this Pipe.
