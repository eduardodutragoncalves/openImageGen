# openImageGen Frontend — Plan

Status: proposed · Owner: Eduardo · Backend: FastAPI (shipped, working)

This plan covers the web UI for the FLUX.2 generation API that already runs in
this repo. It follows three methodologies: `architecture-designer` for the
decision records, Impeccable's `shape` for the design brief, and Impeccable's
craft floor for the visual quality bar.

---

## 1. What the backend actually gives us

Every number below was measured on this codebase, not assumed.

| Fact | Consequence for the UI |
| --- | --- |
| Generation is **asynchronous**: `202` + `job_id` | No request/response screen. The UI is a job tracker. |
| **~5.9 s/step**; 50 steps ≈ **5 min**; edits ~12.6 s/step | The wait *is* the product experience. This is the hardest design problem here. |
| Queue runs **one job at a time** (`OIG_WORKERS=1`) | Queue position is real information and must be shown. |
| Jobs live **in memory**, TTL 1h | History must move to disk or it is lost on restart. |
| `progress` is reported **per step** (0..1) | Real progress bars, not indeterminate spinners. |
| Startup takes minutes; `/healthz` reports `loading` | The app has a first-class "warming up" state. |
| Content filters can end a job as **`rejected`** | A distinct, non-alarming state that explains itself. |
| **No negative prompt** (guidance is distilled) | Do not build a field the model cannot honor. |
| `max_pixels` varies by GPU, exposed in `/v1/models` | Size controls are driven by the server, not hardcoded. |
| Optional `X-API-Key` | Auth is a config concern, not a login screen. |

Endpoints in use: `/healthz`, `/v1/models`, `/v1/gpus`, `/v1/images/generations`,
`/v1/images/edits/upload`, `/v1/jobs`, `/v1/jobs/{id}`, `/v1/jobs/{id}/image`,
`/v1/files/{name}`.

---

## 2. Design brief

**Job and audience.** One operator at a workstation driving a local GPU rig,
plus possibly a teammate on the LAN. They arrive with an intent ("make this
image") and leave with files. They are not browsing; they are working.

**Mode: Operate.** In Impeccable's taxonomy this is a tool, not a landing page.
Scanability, consistency and the real usage scene outrank expression. Brand
lives in precise details, not in decoration.

**The real usage scene.** The user types a prompt, hits generate, and then
**waits five minutes**. They will alt-tab away. They will come back. They will
run several prompts and lose track of which is which. Every important design
decision follows from this, not from the compose form.

**Primary outcome.** The user always knows: what is running, how far along it
is, and where every image they have made can be found.

**What would make it feel wrong.** A spinner with no information. Losing an
image because a tab closed. A form that offers knobs the model ignores.

**Anti-goals.** Not a gallery site. Not a prompt-sharing community. No login,
no accounts, no onboarding tour.

---

## 3. Information architecture

Three regions in one shell, not three disconnected pages:

```
┌──────────────────────────────────────────────────────────────┐
│  status bar: model · placement · GPU memory · queue depth    │
├───────────────────────┬──────────────────────────────────────┤
│  COMPOSE              │  CANVAS                              │
│                       │                                      │
│  prompt               │  active job: live progress + preview │
│  references (0-4)     │  ────────────────────────────────    │
│  size · steps         │  history: every past job             │
│  guidance · seed      │  (prompt, seed, params, images)      │
│  count · upsample     │                                      │
│                       │                                      │
│  [ generate ]         │                                      │
└───────────────────────┴──────────────────────────────────────┘
```

Routes:

- `/` — the studio (above)
- `/j/:jobId` — permalink to one job; makes any result shareable and
  bookmarkable, and is the structural answer to "I lost the id"

Compose and edit are **one form**, not two screens: dropping reference images
into it switches the submit target from `/v1/images/generations` to
`/v1/images/edits/upload`. The user never picks a "mode".

---

## 4. Architecture decisions

### ADR-001 — Static SPA served by FastAPI

**Decision.** Build React with Vite to `app/static/`, mount it in FastAPI, serve
UI and API from one origin.

**Why.** No CORS, one process to run, deploy is `npm run build` plus the
existing `serve.sh`. In dev, Vite proxies `/v1` and `/healthz` to `:8000`.

**Trade-off.** No SSR. Irrelevant here: every screen is authenticated-adjacent,
client-driven, and behind a GPU — there is nothing to pre-render.

**Consequence.** FastAPI needs a `StaticFiles` mount plus an SPA fallback that
does not shadow `/v1/*`.

### ADR-002 — Long-polling, not WebSockets

**Decision.** Track jobs with `GET /v1/jobs/{id}?wait=30`.

**Why.** It already exists and is proven. A job emits one meaningful update
every ~6 seconds; that does not justify a socket. Long-polling survives
proxies and needs no new backend surface.

**Trade-off.** One held connection per tracked job. With a single-worker queue
that is at most a handful.

**Migration path.** If we later parallelize workers, add SSE at
`/v1/jobs/{id}/events` and swap the transport behind the same hook. The UI
contract does not change.

### ADR-003 — Images on disk via `response_format=url`

**Decision.** The client always requests `url`. The API writes to `output/` and
serves from `/v1/files/{name}`.

**Why.** History survives restarts, the browser caches images, and job JSON
stays small instead of carrying ~1.9 MB of base64 per megapixel image.

**Trade-off.** Disk grows without bound and files outlive the in-memory job.

**Consequence.** The backend needs a retention policy (see §6). This is a real
new requirement, not a detail.

### ADR-004 — Typed client generated from OpenAPI

**Decision.** Generate TypeScript types from the FastAPI schema at
`/openapi.json`; hand-write only the thin fetch wrapper.

**Why.** The schema is already accurate and complete — it is the same source
`/docs` renders. Generating removes an entire class of drift between the two
sides of this repo.

### ADR-005 — TanStack Query for all server state

**Decision.** Server state lives in TanStack Query. No Redux, no Zustand.

**Why.** Everything on screen except the compose form is server state with
caching, polling and invalidation needs. Query does exactly that. Form state is
local; the selected job is in the URL. A global store would add a third source
of truth for no gain.

---

## 5. The wait — the part that must be designed, not decorated

Five minutes of silence is the failure mode this app must avoid. Concretely:

- **Real progress.** `progress` is per-step and truthful. Show percentage plus
  step count ("step 23 of 50"), never an indeterminate spinner.
- **ETA from measurement, not guesswork.** After ~3 steps the per-step cost is
  known; multiply by remaining steps. Show a range, and stop showing it once it
  would be a lie (e.g. after the queue reorders).
- **Queue position.** When `status=queued`, say what it is waiting behind.
- **The page stays useful.** Browsing history and queueing more work must never
  be blocked by a running job.
- **Tell them when it lands.** They will alt-tab away: use the Notification API
  (asked for once, on the first successful generation, never on load) and a
  title-bar count as fallback.
- **Warm-up is a state, not an error.** While `/healthz` is `loading`, the
  compose form accepts input and explains that the model is still loading;
  submissions queue normally.

---

## 6. Backend work this requires

The frontend is not purely additive. Four backend changes:

1. **Serve the SPA** — `StaticFiles` mount + SPA fallback ordered after `/v1/*`.
2. **Retention for `output/`** — a size/age cap (config: `OIG_OUTPUT_MAX_GB`,
   `OIG_OUTPUT_MAX_AGE_DAYS`) enforced on write. Without this, ADR-003 fills
   the disk.
3. **Job metadata in the list** — `/v1/jobs` returns prompt and status today;
   the gallery also needs `width`, `height`, `seed` and the image URLs so it can
   render without N follow-up requests.
4. **Dev CORS** — allow `http://localhost:5173` only when `OIG_DEV=true`.

---

## 7. Visual direction

The direction is settled at build time with `/impeccable init`, which writes
`PRODUCT.md` and `DESIGN.md`. What is already decided by the brief:

- **The images are the color.** The UI carries a restrained, near-neutral
  surface so generated images are the only saturated thing on screen. This is a
  functional decision, not a taste one: an accent-heavy chrome would fight every
  thumbnail.
- **Dark-first**, chosen from the use scene (a workstation next to a GPU rig,
  often in a dim room), with a real light theme — not an afterthought.
- **Tabular numerals** for seeds, dimensions, step counts and VRAM. These are
  measurements and must not shift width as they update.

Explicitly refused, per the craft floor:

- Inter, Roboto, Geist and the other saturated UI faces — the type must be
  sourced and self-hosted
- Purple→blue gradients and cyan-on-dark: the strongest tells of AI-generated UI
- Cards as the page scaffold, and nested cards in any form
- Gradient text, kicker/eyebrow labels above headings, emoji standing in for icons
- Gray text on colored surfaces (tint from the surface hue instead)
- Bounce/elastic easing; motion is one authored moment with exponential ease-out

The craft floor also demands what most builds skip: themed selection, caret,
scrollbars and focus rings. This UI ships them.

---

## 8. States to build (all of them)

| Surface | States |
| --- | --- |
| App | warming up · ready · API unreachable |
| Compose | idle · invalid · submitting · queue full (503) |
| Job | queued (with position) · running (with progress) · succeeded · failed · **rejected by filter** |
| Gallery | empty (first run) · populated · image missing from disk |
| References | empty · 1–4 files · rejected file type · too large |

`rejected` deserves its own treatment: it is not an error, it is the content
filter doing its job. It should state which check fired and what to change.

---

## 9. Delivery phases

Each phase ends with something runnable.

- **Phase 0 — Backend prep.** Static mount, SPA fallback, retention policy,
  richer `/v1/jobs`, dev CORS. Verified with the existing `api_examples.sh`.
- **Phase 1 — Shell.** Vite + TS + Tailwind, generated API client, TanStack
  Query, status bar wired to `/healthz` and `/v1/models`, warm-up state.
- **Phase 2 — Generate.** Compose form driven by `/v1/models` limits, submit,
  live progress, notification on completion.
- **Phase 3 — Gallery.** History from `/v1/jobs`, permalinks at `/j/:id`,
  download, "reuse these settings".
- **Phase 4 — Edit.** Drag-and-drop references (1–4), multipart submit,
  before/after comparison.
- **Phase 5 — Polish.** `/impeccable audit` (a11y, responsive, performance),
  `/impeccable critique` (hierarchy, clarity), `/impeccable polish`. Playwright
  covers the critical path: submit → track → retrieve.

---

## 10. Quality gates

- The Impeccable detector hook runs on every UI file edit and again on stop —
  59 deterministic rules, no LLM, no API key.
- Contrast ≥4.5:1 body, ≥3:1 large text, verified on the built result.
- Keyboard reachable end to end; visible focus everywhere.
- Playwright green on the critical path before Phase 5 closes.

---

## 11. Open decisions

These are deliberately not decided here:

1. **Type and palette** — owned by `/impeccable init`, which interviews for
   audience, voice and anti-references before choosing.
2. **Retention defaults** — how much disk the gallery may consume is your call.
3. **LAN exposure** — if the UI is reachable beyond localhost, `OIG_API_KEYS`
   should become mandatory rather than optional.
