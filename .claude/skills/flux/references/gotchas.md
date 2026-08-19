# Gotchas, upstream bugs, silent failures

Verified against HEAD `802fb47`. Check these **before** debugging "it runs but nothing comes out".

## Silent no-output

- **NSFW filter eats the image.** `save_image` runs `Falconsai/nsfw_image_detection`; score ≥ 0.85
  → it prints "Your generated image may contain NSFW content", writes **nothing**, and does not
  increment `idx`. `cli_kontext` passes `None` as the classifier and instead relies on
  `PixtralContentFilter`.
- **Kontext content filter** (`PixtralContentFilter`, `mistral-community/pixtral-12b`, ~25 GB) checks
  the prompt, the input image *and* the output image for NSFW **and** for copyrighted characters /
  logos / public figures. Loaded on **CPU** by default → minutes per check. It is the single biggest
  latency surprise in that CLI. Instantiate it on GPU, or skip it entirely, when you write your own
  loop (`src/flux/cli_kontext.py:247`).
- **Wrong `--name` for a checkpoint does not raise.** `load_state_dict(strict=False)` only prints
  missing/unexpected keys, and `optionally_expand_state_dict` zero-pads shape mismatches. Result:
  a model that loads fine and produces noise. Read the load warnings.

## Upstream bugs to work around

| Where | Problem |
| --- | --- |
| `pyproject.toml` `[project.scripts]` | Declares `flux = "flux.cli:app"`, but `cli.py` defines no `app`. The installed `flux` command fails — use `python -m flux`. |
| `src/flux/cli_kontext.py:303-305` | Leftover debug code: dumps every prepared input tensor to `output/noise.sft` on **each** generation. Delete it in any derived code. |
| `src/flux/cli.py:285`, `cli_kontext.py:356` | Passes the original `prompt` argument to `save_image`, not `opts.prompt`, so EXIF metadata records the *first* prompt for the whole `--loop` / `a\|b\|c` session. |
| `src/flux/cli_kontext.py:319,334` | Unguarded `torch.cuda.synchronize()` — `--device cpu` crashes. |
| `src/flux/cli_control.py` | `**kwargs` swallows unknown flags (`--static_shape`, and any typo) with no error. |
| `docs/structural-conditioning.md` | TRT example says `python flux control ...`; the correct form is `python -m flux control ...`. |
| `pyproject.toml` dependencies | `ruff == 0.6.8` is a hard *runtime* dependency, and `accelerate` is listed twice. `torch == 2.6.0` is pinned in the `torch` extra — that pin fights most existing environments. |

## Shape and dtype traps

- **Always floor h/w to a multiple of 16** (`16 * (v // 16)`). `get_noise` uses `ceil`, `unpack` uses
  `ceil` — off-size requests silently change the output dimensions or crash in `rearrange`.
- `unpack` must receive `x.float()`; AE decode must run inside
  `torch.autocast(device_type=..., dtype=torch.bfloat16)`.
- `CannyImageEncoder` asserts batch size 1. Batch by prompt, never by conditioning image.
- Fill masks must match the conditioning image size exactly and be pure black/white; white = the
  region to regenerate.
- Everything is `torch.inference_mode()`; tensors coming out cannot be used in an autograd graph.

## Behavioural surprises

- **No negative prompt, no CFG.** `guidance` is a distilled scalar embedded into the modulation
  vector. `flux-schnell` ignores it completely (`guidance_embed=False`).
- **Guidance ranges differ wildly per model**: ~2.5–4 for t2i/Kontext, **10** for depth, **30** for
  canny and fill. Copying a t2i guidance into Fill produces mush.
- **Kontext always snaps the reference image** to one of 17 preferred ~1 MP resolutions. Aspect
  ratios far from those get letterboxed by resampling.
- **T5 runs with `attention_mask=None`** and pads to `max_length`, so padding tokens participate in
  attention. Intentional; changing it changes outputs.
- **`ae.encode` is stochastic** (samples from the posterior), so even a fixed seed gives slightly
  different conditioning latents between runs for image-conditioned models.
- **Seeds are CPU-generated** (`torch.Generator(device="cpu")`), so they reproduce across GPUs — but
  attention kernels still make bit-exactness across hardware unlikely.
- **`--loop` resets `opts.seed = None`** after each generation; pass `--seed` again per generation if
  you need determinism in a service.
- In the interactive prompt, `/h` means *help* with no argument and *height* with one. Kontext uses
  `/ar` instead.

## Environment and I/O

- `checkpoints/` is created at **import time** of `flux.util`, relative to the process CWD. A service
  started from a different directory re-downloads everything. Set `FLUX_MODEL`/`FLUX_AE`/etc. or fix
  the CWD.
- Output files are always `output/img_{idx}.jpg` — **JPEG only**, quality 95. Index continues from
  the highest existing `img_N.jpg`, so concurrent processes writing to the same `output_dir` will
  collide.
- `save_image` **always** embeds the invisible watermark (fixed 48-bit dwtDct message) and BFL EXIF
  tags. Bypassing `save_image` to get PNG output also removes the watermark — that is a provenance
  and licensing decision, not just a format one.
- `--track_usage` requires `BFL_API_KEY` (`assert` otherwise) and raises on any non-200 response,
  aborting the run after the image is already on disk.

## Offload

- The offload path moves **`ae.decoder`** to the GPU, not the whole AE. If your loop encodes after
  decoding (image-conditioned pipelines), move `ae` (or `ae.encoder`) back yourself.
- Load with `device="cpu"` when `offload=True`; loading to CUDA first defeats the purpose and OOMs.

## TensorRT

- `--trt` is rejected for the LoRA control variants (`assert not trt`).
- ONNX exports only cover height/width in **768–1344**.
- Engines build into `TRT_ENGINE_DIR` (default `checkpoints/trt_engines`) and take a long time on
  first run; set `TRT_TIMING_CACHE_FILE` to reuse tactics.
- Kontext's fp4 precision string is `fp4_sdvd32`, not `fp4`.
