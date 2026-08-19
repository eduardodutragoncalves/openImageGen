---
name: flux
description: Expert knowledge of black-forest-labs/flux — the official inference repo for the FLUX.1 open-weight image models (schnell, dev, Krea, Kontext, Fill, Canny/Depth, Redux). Use when working with FLUX.1 inference, the `python -m flux` CLI, the `flux` Python package (sampling, denoise, prepare_*, load_flow_model, autoencoder, LoRA, TensorRT), model weights/VRAM/licensing, or when building an app on top of these models. Triggers include: flux, FLUX.1, schnell, kontext, redux, fill, inpainting/outpainting, canny/depth control, Black Forest Labs, BFL, geração de imagem, edição de imagem.
---

# FLUX.1 (black-forest-labs/flux)

Minimal inference code for Black Forest Labs' FLUX.1 open-weight models. Repo state this skill
documents: HEAD `802fb47` (2025-07-31, "FLUX.1 Krea Release"); upstream `main` has not moved since.
FLUX.2 lives in a **separate** repo (`black-forest-labs/flux2`) — nothing here applies to it.

Upstream is a *reference* implementation: small, readable, no abstraction layer. When integrating it
into an app, prefer importing `flux.sampling` / `flux.util` directly and writing your own loop over
shelling out to the CLI. The CLI files (`cli*.py`) are the canonical example of that loop.

## Repo map

| Path | What it holds |
| --- | --- |
| `src/flux/model.py` | `Flux` transformer, `FluxParams`, `FluxLoraWrapper` |
| `src/flux/sampling.py` | `get_noise`, `prepare*`, `get_schedule`, `denoise`, `unpack` — the whole pipeline |
| `src/flux/util.py` | `configs` (per-model `ModelSpec`), loaders, checkpoint download, watermark, `save_image` |
| `src/flux/modules/` | `autoencoder.py`, `conditioner.py` (T5/CLIP), `image_embedders.py` (Canny/Depth/Redux), `layers.py`, `lora.py` |
| `src/flux/content_filters.py` | `PixtralContentFilter` (used by Kontext CLI) |
| `src/flux/cli*.py` | Fire-based CLIs: `cli` (t2i), `cli_control`, `cli_fill`, `cli_kontext`, `cli_redux` |
| `src/flux/trt/` | TensorRT engine building/loading (optional extra) |
| `demo_st.py`, `demo_st_fill.py`, `demo_gr.py` | Streamlit / Gradio demos |

Install: `pip install -e ".[all]"` (Python ≥3.10, pins `torch==2.6.0`). TensorRT needs the NVIDIA
PyTorch container plus `pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com`.

## The pipeline (identical for every model)

1. `get_noise(1, h, w, device, torch.bfloat16, seed)` → latent noise, shape `(1, 16, 2⌈h/16⌉, 2⌈w/16⌉)`.
2. `prepare*(...)` → dict with `img`, `img_ids`, `txt`, `txt_ids`, `vec` (+ conditioning keys).
   T5-XXL gives `txt`, CLIP-L gives the pooled `vec`. Latents are packed 2×2 → sequence length
   `(h/16)·(w/16)`.
3. `get_schedule(num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))` → timesteps,
   resolution-aware shift (mu interpolated between 0.5 @256 tokens and 1.15 @4096).
4. `denoise(model, **inp, timesteps=..., guidance=...)` → Euler steps on the flow-matching field.
5. `unpack(x.float(), h, w)` → back to latent grid.
6. `ae.decode(x)` under `torch.autocast(bfloat16)` → pixels in `[-1, 1]`; then `save_image(...)`.

Conditioning enters `denoise` in exactly two ways:
- **channel-wise** `img_cond` — concatenated on the last dim (Fill, Canny, Depth; those checkpoints
  have `in_channels` 384 / 128 instead of 64).
- **sequence-wise** `img_cond_seq` + `img_cond_seq_ids` — appended as extra tokens (Kontext).
  Redux is different again: it appends SigLIP-derived tokens to `txt`, not to `img`.

## Non-negotiable rules

- **Height and width must be multiples of 16.** Every CLI does `16 * (v // 16)` — do the same.
- **bfloat16 everywhere**; the AE decode runs inside `torch.autocast`, and `unpack` takes `x.float()`.
- **`checkpoints/` is created relative to the current working directory** at `import flux.util` time.
  Weights auto-download from HuggingFace (gated repos need `HF_TOKEN` / `huggingface-cli login`).
  Override paths with `FLUX_MODEL`, `FLUX_AE`, `FLUX_LORA`, `FLUX_REDUX`.
- **`save_image` always embeds an invisible watermark** (fixed 48-bit dwtDct message) and writes EXIF
  `Software=AI generated;...`, `Make=Black Forest Labs`. It saves JPEG only.
- **Licensing:** only `FLUX.1 [schnell]` (and the autoencoder) are Apache-2.0. Everything `[dev]` is
  under the FLUX.1-dev **non-commercial** license; commercial use needs a BFL license plus usage
  reporting (`--track_usage` with `BFL_API_KEY`). Never advise shipping `[dev]` weights commercially
  without that.
- **Seeds:** `get_noise` uses a CPU generator, so results are reproducible across devices for a seed.
  The CLIs set `opts.seed = None` after each generation so `--loop` re-randomizes.

## Defaults that matter

| Command | Model name(s) | Steps | Guidance |
| --- | --- | --- | --- |
| `t2i` | `flux-dev-krea` (CLI default), `flux-dev`, `flux-schnell` | 50 (4 for schnell) | 2.5 (schnell ignores it) |
| `kontext` | `flux-dev-kontext` | 30 | 2.5 |
| `fill` | `flux-dev-fill` | 50 | **30.0** |
| `control` | `flux-dev-canny[-lora]` | 50 | **30.0** |
| `control` | `flux-dev-depth[-lora]` | 50 | **10.0** |
| `redux` | base `flux-dev` / `flux-schnell` | 50 / 4 | 2.5 |

`flux-schnell` has `guidance_embed=False` and uses T5 `max_length=256`; all others use 512 and
consume the guidance embedding. LoRA control variants default to `lora_scale=0.85`.

## References

Load these only when the task needs them:

- `references/cli.md` — every subcommand, all flags, interactive `--loop` commands, TensorRT usage.
- `references/python-api.md` — copy-pasteable integration code for t2i, kontext, fill, control,
  redux; batching, offload, and how to strip the CLI-only bits.
- `references/architecture.md` — transformer internals, latent packing, RoPE ids, timestep shift,
  how each conditioning mode is wired.
- `references/models.md` — model table, checkpoint files, sha256s, VRAM budgets, licenses, ONNX/TRT repos.
- `references/gotchas.md` — known upstream bugs, silent failures, and traps (read this before
  debugging anything that "runs but produces nothing").
