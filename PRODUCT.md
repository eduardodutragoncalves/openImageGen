# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite + TypeScript + Tailwind, with TanStack Query for all server state.
Built to `app/static/` and served by the existing FastAPI process on one origin;
Vite proxies `/v1` and `/healthz` to `:8000` in development. Confirmed by the
user, matching ADR-001/ADR-005 in [docs/frontend-plan.md](docs/frontend-plan.md).
TypeScript types are generated from the live FastAPI schema at `/openapi.json`
(ADR-004) so the two sides of the repo cannot drift.

## Users

One operator driving a local GPU rig, plus a small number of trusted people who
reach the same box **over the internet**, not only the LAN. They arrive with an
intent ("make this image", "change this image") and leave with files. They are
working, not browsing.

**Desktop only.** The surface is designed for a large screen at a workstation.
Phone and tablet are not supported clients and no design effort is spent on
them, despite the service being remotely reachable — a deliberate decision by
the user, recorded here so future work does not quietly reintroduce a mobile
scope nobody asked for.

## Product Purpose

A web front end for this repo's FLUX.2 generation and editing API, turning a
single-worker GPU queue into something a person can drive: submit work, know
exactly what is happening during a multi-minute wait, and find every image they
have ever made.

Success: the user always knows what is running, how far along it is, and where
every image they have made can be found. Failure: a spinner with no information,
or an image lost because a tab closed.

## Positioning

Not a hosted image service and not a prompt-sharing community. It is the
operator's console for **their own hardware** — it exposes the machine's real
truth (which model, which GPU, which placement, how much VRAM, how deep the
queue) rather than hiding it behind a rented abstraction. Nothing that does not
run on your own GPU can honestly show that.

## Operating Context

- Generation is **asynchronous**: submit returns `202` with a `job_id`; the UI is
  a job tracker, not a request/response screen.
- **~5.9 s/step**; 50 steps ≈ **5 minutes**. Edits cost ~12.6 s/step. The wait is
  the central experience: users will alt-tab away, come back, run several prompts
  and lose track of which is which.
- Generation speed is a property of the loaded model, not of the product:
  FLUX.2 [dev] at 50 steps is ~5 minutes, while FLUX.1 [schnell] at 4 steps
  is seconds. The UI must read as honest at both extremes.
- The queue runs **one job at a time** (`OIG_WORKERS=1`, `OIG_QUEUE_MAX_SIZE=32`),
  so queue position is real, useful information.
- Startup takes minutes; `/healthz` reports `loading` until the weights are
  resident. "Warming up" is a first-class product state, not an error.
- Compose and edit are the same act with different inputs: attaching 1–4
  reference images switches the submit target from `/v1/images/generations` to
  `/v1/images/edits/upload`. The user should never have to pick a "mode".
- Endpoints in use: `/healthz`, `/v1/models`, `/v1/gpus`,
  `/v1/images/generations`, `/v1/images/edits/upload`, `/v1/jobs`,
  `/v1/jobs/{id}` (supports `?wait=` long-polling), `/v1/jobs/{id}/image`,
  `/v1/files/{name}`.

## Capabilities and Constraints

**Confirmed capabilities**

- Text-to-image, image edit, and multi-reference edit (1–4 references), as
  reported per-model in `/v1/models.capabilities`.
- Per-step progress `0..1` on every job — real progress bars are possible and
  indeterminate spinners are never justified.
- Job states: `queued`, `running`, `succeeded`, `failed`, and **`rejected`**.
- Optional prompt upsampling (`none`, `local`, `openrouter`).
- Parameter ranges enforced by the API: prompt ≤ 8000 chars, width/height
  256–2048, steps 1–100, guidance 0–20, seed 0–2³¹−1, 1–4 images per request.

**Constraints future work must respect**

- **No negative prompt.** FLUX.2 [dev] is guidance-distilled; `guidance` is an
  embedded scalar, not CFG. Never build a field the model cannot honor.
- `max_pixels` is derived from the VRAM left after the weights load and varies by
  machine. Size controls must be driven by `/v1/models`, never hardcoded.
- Two content filters (NSFW on output, integrity on prompt and image) can end a
  job as `rejected`. This is the system working, not an error, and must read as
  such.
- Model choice is a licensing decision the operator makes: the default
  `FLUX.2 [dev]` is non-commercial; `FLUX.2 [klein] 4B` is Apache-2.0. The UI
  reports the loaded model, it does not choose one.
- The result is placement-independent: the same seed produces the same image
  under `split`, `single` or `offload`.

**Decided here, requiring backend work that does not exist yet**

- **Model switching is a product capability, not an env var.** The UI has a
  configuration surface where the operator sees which models this server can
  run and switches between them. Scope confirmed: the **FLUX.2 family**
  (`FLUX.2-dev-bnb-4bit`, `FLUX.2-dev` bf16, `FLUX.2-klein-4B`) and the
  **FLUX.1 family** (`dev`, `schnell`, `Kontext`), with other families to
  follow. FLUX.1 does not share FLUX.2's loading path — it uses `FluxPipeline`
  with T5-XXL + CLIP-L instead of `Flux2Pipeline` with a Mistral3 encoder — so
  the engine becomes a pluggable backend behind a registry rather than one
  FLUX.2-specific class.
- **Switching happens in-process, live.** The server drains the queue, unloads
  ~35GB of weights, replans placement for the new model, and loads it. This
  takes minutes and cannot overlap a running job, so **`switching` is a
  first-class application state** with its own progress — the same honesty the
  product already owes the generation wait. A failed swap keeps the previous
  model loaded and says so.
- Model choice changes what the rest of the UI may offer: steps, guidance,
  `max_pixels`, and edit capability differ per model, and FLUX.1 [schnell]
  finishes in ~4 steps rather than 50. Every control stays driven by
  `/v1/models`, never by constants.

- **Auth is mandatory, not optional.** Because the service is reachable beyond
  the LAN, `OIG_API_KEYS` must be required rather than empty-by-default. Access
  is a shared secret entered once and kept locally in the browser — no accounts,
  no signup, no password reset, no login screen in the product sense.
- **History is scoped per API key.** Each person gets their own key from the
  comma-separated `OIG_API_KEYS` list, and a user sees only the jobs made with
  their key. Today `Job` records carry no owner at all, so attribution is new
  work in `app/jobs.py`.
- **History must survive restarts.** Jobs currently live in memory with a 1h TTL
  (`OIG_JOB_TTL_SECONDS`), so an archive requires on-disk job metadata plus a
  retention policy for `output/` (the disk otherwise grows without bound once
  images are written as URLs rather than base64).
- The SPA needs a `StaticFiles` mount and a fallback ordered after `/v1/*`.

**Explicitly undecided**

- Retention defaults — how much disk history may consume, and for how long.
- Rate limiting and abuse handling for internet exposure: agreed as necessary,
  with no thresholds chosen.

## Brand Commitments

Name: **openImageGen**. No logo, wordmark, or existing visual identity. No voice
guidelines beyond what the README already demonstrates: precise, measured,
unexcited, and willing to state a trade-off plainly. Numbers in this product are
measurements, not decoration.

## Evidence on Hand

- A working, shipped FastAPI backend (`app/`) — every performance number above
  was measured on it, not estimated.
- Real generated and edited output in `output/` (e.g. `fox.png`,
  `fox_edited.png`, `upsample_local.png`) usable as genuine sample imagery.
- `scripts/api_examples.sh` and `scripts/smoke_test.py` document real request and
  response shapes; `/openapi.json` is authoritative and current.
- `docs/frontend-plan.md` — the accepted plan whose ADRs and phases this record
  confirms.
- **No** users beyond the operator and their trusted circle, no testimonials, no
  benchmarks against other tools, no pricing, no deployment history. Future work
  must not fabricate any of these.

## Product Principles

1. **The wait is the product.** Five minutes of silence is the failure mode.
   Every state shows real, measured truth — step count, progress, queue position,
   an ETA derived from observed per-step cost and withdrawn the moment it would
   become a lie.
2. **Never lose an image.** A closed tab, a lost job id, or a restarted server
   must not cost the user work. Every job is addressable and recoverable.
3. **Expose the machine honestly.** Model, placement, GPU memory and queue depth
   are shown because the operator owns the hardware and is entitled to its state.
4. **Only knobs the model honors.** No control exists in the UI that the API
   ignores, and every limit comes from the server rather than from a guess.
5. **Refusal is not failure.** A `rejected` job explains which check fired and
   what to change, in the same measured voice as everything else.

## Accessibility & Inclusion

No user-specific requirement was established. The general floor applies: full
keyboard reachability with visible focus, contrast ≥4.5:1 for body text and ≥3:1
for large text, and tabular numerals so seeds, dimensions, step counts and VRAM
readings do not shift width as they update.
