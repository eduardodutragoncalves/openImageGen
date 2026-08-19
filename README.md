# openImageGen

An HTTP API for image generation and editing with **FLUX.2** (Black Forest
Labs), served with FastAPI. The service detects the GPUs available at startup
and places the models itself — from a single 12GB card to a multi-GPU host.

---

## ⚠️ License

`FLUX.2 [dev]`, the default model, is under the **FLUX Non-Commercial
License**. Commercial use requires a license from Black Forest Labs
(<https://bfl.ai/pricing/licensing>). The autoencoder is Apache-2.0. For a
commercial product, use `FLUX.2 [klein] 4B`, which is Apache-2.0 — see
[Switching models](#switching-models).

---

## Requirements

- An NVIDIA GPU with a recent driver and a working CUDA build of PyTorch
- Anaconda or Miniconda
- ~40GB of free disk space for the weights
- No device configuration needed: the service decides on its own

The default model (`diffusers/FLUX.2-dev-bnb-4bit`) quantizes both the 32B
transformer (~19GB) and the 24B Mistral text encoder (~16GB) to 4-bit.

| Available VRAM | What happens |
| --- | --- |
| 2+ GPUs, each able to hold one component | `split`: transformer + VAE on one card, text encoder on the other. No offloading, no PCIe traffic during sampling. |
| 1 large GPU (~40GB+) | `single`: everything resident on the same card. |
| 1 smaller GPU (12-24GB) | `offload`: diffusers moves each component onto the GPU only while it runs. Slower, but functional. |
| No GPU | Explicit error at `/healthz`. Use `OIG_DRY_RUN=true` to exercise the HTTP layer only. |

The decision and its reason show up in `GET /v1/models` and in the startup log:

```
placement=offload: NVIDIA GeForce RTX 3090 (24GB) cannot hold transformer
(~19GB) and text encoder (~16GB) together: enabling sequential CPU offload
```

To force a different layout, set `OIG_TRANSFORMER_DEVICE`,
`OIG_TEXT_ENCODER_DEVICE` and/or `OIG_CPU_OFFLOAD` in `.env`.

### Why split components across GPUs when there is more than one

The usual recipe for fitting in 24GB is `enable_model_cpu_offload()` on a
single card. With a second GPU available, keeping the text encoder resident on
it avoids moving weights between CPU and GPU during inference, and keeps the
encoder available for *prompt upsampling* and the integrity filter at no extra
VRAM cost. Only `prompt_embeds` cross the boundary between cards, once per
request.

This requires pinning the pipeline's `_execution_device` to the transformer's
device ([app/engine.py](app/engine.py)); without that, diffusers picks the
device of the first module it finds and the latents end up on the wrong card.

The result does **not** depend on placement: the same seed produces the same
image under `split`, `single` or `offload`.

---

## Setup

```bash
# 1. environment
conda env create -f environment.yml
conda activate openimagegen

# 2. configuration (optional: the defaults work without editing anything)
cp .env.example .env

# 3. weights (~34GB, do this before the first request)
python scripts/download_weights.py

# 4. serve
./scripts/serve.sh
```

The API comes up immediately and loads the models in the background:
`GET /healthz` reports `loading` until everything is ready. Requests sent
during that window sit in the queue and are processed once the model has
loaded.

To hide cards from the service, use `CUDA_VISIBLE_DEVICES` as usual —
`serve.sh` does not touch that variable.

---

## Endpoints

| Method | Route | Description |
| --- | --- | --- |
| `POST` | `/v1/images/generations` | Text-to-image |
| `POST` | `/v1/images/edits` | Edit with 1..N reference images (base64/data URI) |
| `POST` | `/v1/images/edits/upload` | Same, via multipart upload |
| `GET` | `/v1/jobs/{id}` | Job status (`?wait=30` blocks until it finishes) |
| `GET` | `/v1/jobs/{id}/image` | Raw image bytes (`?index=N`) |
| `GET` | `/v1/files/{name}` | File saved when `response_format="url"` |
| `GET` | `/v1/models` | Loaded model, placement and defaults |
| `GET` | `/v1/gpus` | VRAM usage per card and each one's role |
| `GET` | `/healthz` | Service and queue status |

Interactive docs at `http://localhost:8000/docs`.

A ready-to-run curl reference for every endpoint lives in
[scripts/api_examples.sh](scripts/api_examples.sh) — see
[Trying it from the command line](#trying-it-from-the-command-line).

Generation is **asynchronous**: the POST responds with `202` and a `job_id`.
A FLUX.2 [dev] image takes anywhere from tens of seconds to several minutes
depending on the GPU and the step count, so holding the HTTP connection open
would not be practical.

### Text-to-image

```bash
curl -s -X POST localhost:8000/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{
        "prompt": "macro photo of a hermit crab using a can as its shell, late afternoon light",
        "width": 1024, "height": 1024,
        "num_steps": 50, "guidance": 4.0, "seed": 42
      }'
# {"id":"...","status":"queued","queue_position":0,"poll_url":"/v1/jobs/..."}

curl -s "localhost:8000/v1/jobs/<id>?wait=60" | jq '.status, .progress'
curl -s "localhost:8000/v1/jobs/<id>/image" -o output.png
```

### Editing / multi-reference

```bash
curl -s -X POST localhost:8000/v1/images/edits/upload \
  -F "prompt=put the product from the first image on the table in the second" \
  -F "images=@product.png" \
  -F "images=@scene.jpg" \
  -F "match_image_size=1"
```

FLUX.2 VAE-encodes each reference and appends it to the image token sequence,
so several references can be combined in one call. diffusers caps each
reference at ~1MP automatically.

### Parameters

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | required |
| `width` / `height` | 1024 | rounded to multiples of 16 and capped by the pixel budget |
| `num_steps` | 50 | 28 is a good quality/time trade-off |
| `guidance` | 4.0 | a *distilled* scalar, not CFG — **there is no negative prompt** |
| `seed` | random | extra images use `seed+1`, `seed+2`, … |
| `num_images` | 1 | generated sequentially (memory) |
| `upsample_prompt` | `none` | `local` (resident text encoder) or `openrouter` |
| `response_format` | `b64_json` | or `url` (saved under `output/`) |
| `output_format` | `png` | `png`, `jpeg`, `webp` |

The pixel budget is derived from the VRAM left over after the weights load
(from 768² on tight cards to 1536² on roomy ones) and shows up in
`GET /v1/models`. Larger requests are scaled down keeping the aspect ratio,
instead of running out of memory. Pin it with `OIG_MAX_PIXELS` if you prefer.

---

## Content filters

Two independent layers, both configurable:

- **NSFW** (`OIG_ENABLE_NSFW_FILTER`): the Falconsai classifier applied to
  reference images and to every generated image.
- **Integrity** (`OIG_ENABLE_INTEGRITY_FILTER`): uses the already-loaded text
  encoder to answer yes/no about protected characters, brands and public
  figures — the same restricted-logits technique used by BFL's reference
  CLIs.

Blocked content ends the job with status `rejected` and the reason in `error`
(`GET /v1/jobs/{id}` returns 200; `GET /v1/jobs/{id}/image` returns 409).

---

## Switching models

The service depends only on `OIG_REPO_ID`; the planner recognizes the FLUX.2
families and recomputes placement on its own.

```bash
# FLUX.2 [klein] 4B — Apache-2.0, ~9GB, 4 steps, sub-second
OIG_REPO_ID=black-forest-labs/FLUX.2-klein-4B
OIG_DEFAULT_STEPS=4
OIG_DEFAULT_GUIDANCE=1.0
```

`klein` is distilled in both steps **and** guidance: `num_steps` and
`guidance` are fixed, changing them degrades the output.

For a checkpoint the planner does not recognize, supply the sizes with
`OIG_TRANSFORMER_VRAM_GB` and `OIG_TEXT_ENCODER_VRAM_GB` — it warns in the log
when it is guessing.

---

## Performance

Cost is dominated by the denoising loop, which scales with steps × pixels.
`bitsandbytes` 4-bit dequantizes on every matmul, and GPUs older than Ada have
no native fp4/fp8 support — the model fits on them, it just isn't fast.

Measured reference on a 24GB Ampere-class card, 1024×1024, to give a sense of
scale (your GPU will vary):

| Stage | `split` (2 GPUs) | `offload` (1 GPU) |
| --- | --- | --- |
| Denoise, per step | ~5.9s | ~7.1s |
| Prompt encoding | 0.5s | 0.5s |
| Filters (input + output) | ~1.4s | ~5.2s (classifier on CPU) |

At 50 steps that is a few minutes per image. For low latency, `klein 4B` is
orders of magnitude faster. Editing references roughly double the token
sequence, and therefore the per-step time.

---

## Trying it from the command line

[scripts/api_examples.sh](scripts/api_examples.sh) has a ready-to-run curl
call for every endpoint — health, models, GPUs, text-to-image, editing (both
JSON and multipart), job polling, image download and file retrieval. It is
meant to be read as documentation as much as run:

```bash
# health, models, GPUs
scripts/api_examples.sh health
scripts/api_examples.sh models
scripts/api_examples.sh gpus

# submit a generation and print the job id
scripts/api_examples.sh generate "a fox in a misty forest"

# submit, then poll until done and save the first image
scripts/api_examples.sh generate-wait "a fox in a misty forest" output/fox.png

# edit an existing image
scripts/api_examples.sh edit output/fox.png "make the fog orange" output/fox_edited.png

# poll a job you already have the id for
scripts/api_examples.sh job <job_id>

# print every command without running them (useful as a curl cheat sheet)
scripts/api_examples.sh --print-only generate "a fox"
```

Set `BASE_URL` and `API_KEY` to point it elsewhere or add authentication:

```bash
BASE_URL=http://gpu-host:8000 API_KEY=key-one scripts/api_examples.sh health
```

---

## Layout

```
app/
  config.py      Settings (pydantic-settings, OIG_ prefix)
  devices.py     GPU detection and placement planning
  schemas.py     request/response contracts
  engine.py      model loading, placement, generation
  jobs.py        FIFO queue + workers
  safety.py      NSFW + integrity filter
  upsampler.py   local and OpenRouter prompt upsampling
  images.py      base64/PIL helpers, pixel budget
  main.py        FastAPI routes
scripts/
  download_weights.py   pre-download the HF cache
  serve.sh              start the API inside the conda env
  smoke_test.py         end-to-end check against a running API
  api_examples.sh       curl reference for every endpoint
```

---

## Troubleshooting

**`CUDA out of memory` during denoising** — lower `OIG_MAX_PIXELS` or
`num_images`, or force `OIG_CPU_OFFLOAD=true`. `serve.sh` already exports
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which avoids most 4-bit
allocator fragmentation.

**Wrong placement** — the planner uses approximate footprints. Override with
`OIG_TRANSFORMER_DEVICE`, `OIG_TEXT_ENCODER_DEVICE`, `OIG_CPU_OFFLOAD` or the
`*_VRAM_GB` settings. Note: setting any of these turns off auto-detection.

**`conda.sh: line NN: PS1: unbound variable`** — fixed in `serve.sh`. conda's
shell hook dereferences `$PS1`, which is unset in the non-interactive shell a
script runs in, so `set -u` aborted the activation. It only surfaced when the
environment was *already* active in the calling shell, because that makes
`conda activate` take its reactivation path. The script now relaxes `nounset`
around the activation.

**`No prebuilt binary for CUDA 12.9`** — a bitsandbytes warning; it falls back
to the 12.8 binary and works normally.

**`/healthz` reports `error`** — the `detail` field carries the loading
exception. Common causes: incomplete weights (re-run the download) or an HF
token without access to a gated repo.

**Testing without a GPU** — `OIG_DRY_RUN=true ./scripts/serve.sh` responds
with solid-color images, exercising the whole HTTP layer.

**Local upsampling used to return gibberish** — fixed in
[app/upsampler.py](app/upsampler.py). diffusers' `Flux2Pipeline.upsample_prompt`
tokenizes with `padding="max_length", max_length=2048`, i.e. it right-pads
before calling `generate()`. On a decoder-only model that makes generation
continue past hundreds of pad tokens and the result comes out degenerate. Our
implementation reuses BFL's official system messages but tokenizes a single
conversation without padding.
