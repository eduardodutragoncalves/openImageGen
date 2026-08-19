# CLI reference

Entry point is `src/flux/__main__.py`, dispatched with [Fire](https://github.com/google/python-fire):

```
python -m flux {t2i|control|fill|kontext|redux} [--flag value ...]
```

Fire maps every function argument to `--arg`. Booleans work as bare flags (`--loop`, `--offload`,
`--trt`) or `--loop=False`. There is also a `flux` console script declared in `pyproject.toml`
(`flux.cli:app`) but **that attribute does not exist** — use `python -m flux`.

## Shared flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--seed` | `None` | `None` → random per generation |
| `--device` | `cuda` if available | |
| `--num_steps` | see per-command | |
| `--guidance` | see per-command | ignored by `flux-schnell` |
| `--loop` | `False` | interactive session, see below |
| `--offload` | `False` | shuttle T5/CLIP/model/AE between CPU and GPU per stage |
| `--output_dir` | `output` | files are `img_{idx}.jpg`, idx continues from existing files |
| `--add_sampling_metadata` | `True` | writes the prompt into EXIF `ImageDescription` |
| `--track_usage` | `False` | POSTs to `api.bfl.ai` per saved image; needs `BFL_API_KEY` |
| `--trt` | `False` | TensorRT backend (not on all commands; never with LoRA) |
| `--trt_transformer_precision` | `bf16` | `bf16` \| `fp8` \| `fp4` (Kontext: `fp4_sdvd32`) |

## `t2i` — text to image (`cli.py`)

```bash
python -m flux t2i --name flux-dev --width 1360 --height 768 --prompt "..." --guidance 3.5
python -m flux t2i --name flux-schnell --num_steps 4 --prompt "..."
python -m flux t2i --loop                      # interactive, defaults to flux-dev-krea
```

- `--name`: `flux-dev-krea` (default), `flux-dev`, `flux-schnell`. Any key of `flux.util.configs`
  is technically accepted, but only these make sense here.
- Default size `1360x768`, guidance `2.5`, steps `50` (`4` when name is `flux-schnell`).
- `--prompt "a | b | c"` runs a **queue** of prompts (split on `|`); cannot be combined with `--loop`.
- Runs a `Falconsai/nsfw_image_detection` classifier; images scoring ≥0.85 are **not written**.

## `kontext` — instruction-based image editing (`cli_kontext.py`)

```bash
python -m flux kontext \
  --img_cond_path input.png \
  --prompt "replace the logo with the text 'Black Forest Labs'" \
  --num_steps 30 --aspect_ratio "16:9" --guidance 2.5 --seed 1
```

- `--name` must be `flux-dev-kontext` (asserted).
- `--img_cond_path` (default `assets/cup.png`) — jpg/jpeg/png/webp.
- `--aspect_ratio "W:H"` → `aspect_ratio_to_height_width()` picks a ~1 MP size, multiple of 16.
  Omit it to inherit the conditioning image's resolution.
- The conditioning image is always snapped to the nearest of `PREFERED_KONTEXT_RESOLUTIONS`
  (17 sizes from 672×1568 to 1568×672, all ≈1 MP) by aspect ratio.
- Runs `PixtralContentFilter` on prompt, input image, **and** output image. See `gotchas.md`.

## `fill` — inpainting / outpainting (`cli_fill.py`)

```bash
python -m flux fill --img_cond_path image.png --img_mask_path mask.png --prompt "a white paper cup"
```

- Mask must be the **same size** as the conditioning image, black/white only (white = region to fill).
- Guidance defaults to `30.0`, steps `50`. No `--name` flag: always `flux-dev-fill`.
- No `--width`/`--height`: output size comes from the input image.
- Streamlit alternative with a drawable mask canvas: `streamlit run demo_st_fill.py`.

## `control` — Canny / Depth structural conditioning (`cli_control.py`)

```bash
python -m flux control --name flux-dev-canny --img_cond_path assets/robot.webp --loop
python -m flux control --name flux-dev-depth-lora --lora_scale 0.85 --prompt "a robot made out of gold"
```

- `--name` (required): `flux-dev-canny`, `flux-dev-depth`, `flux-dev-canny-lora`, `flux-dev-depth-lora`.
- Guidance defaults by family: canny `30.0`, depth `10.0`.
- `--lora_scale` (default `0.85`) applies only to the `-lora` variants; `--trt` is rejected for them.
- Canny is computed with OpenCV thresholds 50/200; depth with `LiheYoung/depth-anything-large-hf`.
- The conditioning image is resized to the requested `--width`/`--height` (default 1024×1024).

## `redux` — image variation (`cli_redux.py`)

```bash
python -m flux redux --name flux-dev --img_cond_path assets/robot.webp --loop
```

- `--name` is the **base** model: `flux-dev` or `flux-schnell`. The Redux adapter
  (`flux1-redux-dev.safetensors`, env `FLUX_REDUX`) is downloaded separately.
- There is **no `--prompt` flag**: the prompt starts as `""` and the SigLIP embedding of the input
  image carries the conditioning. You can still type a prompt interactively under `--loop`.

## Interactive `--loop`

Prompt-line commands (t2i / control / fill / redux):

```
/w <width>   /h <height>   /s <seed>   /g <guidance>   /n <steps>   /q
```

Kontext uses `/ar <W:H>` (or `/ar auto`) instead of `/w`, then asks for the next input image path.
`control` additionally prompts for a new LoRA scale on the `-lora` variants. Empty input repeats the
previous value.

## TensorRT

```bash
python -m flux t2i --name flux-dev --loop --trt --trt_transformer_precision fp8
python -m flux kontext --loop --trt --trt_transformer_precision fp4_sdvd32
```

ONNX weights are pulled from `black-forest-labs/FLUX.1-*-onnx` into `checkpoints/`; engines are built
into `TRT_ENGINE_DIR` (defaults to `checkpoints/trt_engines`). For ONNX exports, height and width must
be within 768–1344. Other env vars: `TRT_T5_PRECISION` (default `bf16`), `TRT_TIMING_CACHE_FILE`,
`CUSTOM_ONNX_PATHS`.

## Demos

```bash
streamlit run demo_st.py          # t2i + img2img
streamlit run demo_st_fill.py     # inpainting with canvas mask
python demo_gr.py --name flux-schnell --device cuda [--offload] [--share]
```
