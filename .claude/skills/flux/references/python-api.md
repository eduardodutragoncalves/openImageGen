# Using `flux` as a library

The CLIs are thin wrappers. To embed FLUX in an app, import the same functions and drop the
interactive/Fire parts. Every snippet below mirrors the corresponding `cli_*.py` exactly.

## Text to image

```python
import torch
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

name = "flux-dev"          # or "flux-schnell", "flux-dev-krea"
device = torch.device("cuda")
h, w = 16 * (768 // 16), 16 * (1360 // 16)

t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
clip = load_clip(device)
model = load_flow_model(name, device=device)
ae = load_ae(name, device=device)

with torch.inference_mode():
    x = get_noise(1, h, w, device=device, dtype=torch.bfloat16, seed=42)
    inp = prepare(t5, clip, x, prompt="a photo of a forest with mist")
    timesteps = get_schedule(50, inp["img"].shape[1], shift=(name != "flux-schnell"))
    x = denoise(model, **inp, timesteps=timesteps, guidance=3.5)
    x = unpack(x.float(), h, w)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)          # (1, 3, h, w) in [-1, 1]
```

Convert to PIL yourself if you don't want the watermark/EXIF/NSFW behaviour of `save_image`:

```python
from einops import rearrange
from PIL import Image
img = rearrange(x[0].clamp(-1, 1).float(), "c h w -> h w c")
Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy()).save("out.png")
```

## Kontext (instruction editing)

`prepare_kontext` builds the noise itself and returns the resolved size.

```python
from flux.sampling import denoise, get_schedule, prepare_kontext, unpack

inp, height, width = prepare_kontext(
    t5=t5, clip=clip, prompt="replace the logo with the text 'BFL'",
    ae=ae, img_cond_path="input.png", seed=42, device=device,
    target_width=None, target_height=None, bs=1,   # None → follow the conditioning image
)
inp.pop("img_cond_orig")                            # not a model input
timesteps = get_schedule(30, inp["img"].shape[1], shift=True)
x = denoise(model, **inp, timesteps=timesteps, guidance=2.5)
x = unpack(x.float(), height, width)
```

The conditioning image is snapped to the nearest `PREFERED_KONTEXT_RESOLUTIONS` entry regardless of
`target_*`; `target_width/height` only control the *generated* canvas.

## Fill (inpaint / outpaint)

```python
from PIL import Image
from flux.sampling import get_noise, prepare_fill

with Image.open("image.png") as im:
    w, h = im.size                       # output size == input size
x = get_noise(1, h, w, device=device, dtype=torch.bfloat16, seed=42)
inp = prepare_fill(t5, clip, x, prompt="a white paper cup", ae=ae,
                   img_cond_path="image.png", mask_path="mask.png")
timesteps = get_schedule(50, inp["img"].shape[1], shift=True)
x = denoise(model, **inp, timesteps=timesteps, guidance=30.0)
```

Load with `name="flux-dev-fill"`. The mask is grayscale, same size as the image; masked pixels are
zeroed out of the conditioning before encoding, and the 8×8-unfolded mask is concatenated to the
latent channels.

## Control (Canny / Depth)

```python
from flux.modules.image_embedders import CannyImageEncoder, DepthImageEncoder
from flux.sampling import prepare_control

encoder = CannyImageEncoder(device)            # or DepthImageEncoder(device)
x = get_noise(1, h, w, device=device, dtype=torch.bfloat16, seed=42)
inp = prepare_control(t5, clip, x, prompt="a robot made out of gold",
                      ae=ae, encoder=encoder, img_cond_path="assets/robot.webp")
x = denoise(model, **inp, timesteps=get_schedule(50, inp["img"].shape[1], shift=True),
            guidance=30.0)                     # 30 for canny, 10 for depth
```

LoRA variants (`flux-dev-canny-lora`, `flux-dev-depth-lora`) load the base dev checkpoint plus the
adapter automatically; set the scale after loading:

```python
model = load_flow_model("flux-dev-canny-lora", device=device)
for _, module in model.named_modules():
    if hasattr(module, "set_scale"):
        module.set_scale(0.85)
```

## Redux (image variation)

```python
from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import prepare_redux
from flux.util import get_checkpoint_path

redux_path = str(get_checkpoint_path("black-forest-labs/FLUX.1-Redux-dev",
                                     "flux1-redux-dev.safetensors", "FLUX_REDUX"))
img_embedder = ReduxImageEncoder(device, redux_path=redux_path)
inp = prepare_redux(t5, clip, x, prompt="", encoder=img_embedder,
                    img_cond_path="assets/robot.webp")
```

Base model is `flux-dev` or `flux-schnell`. The SigLIP tokens are concatenated to the **text**
sequence, so a non-empty prompt still works but competes with the image tokens.

## Batching

`prepare*` accept a `list[str]` prompt and will broadcast a batch-1 image/conditioning to
`len(prompt)`. `CannyImageEncoder` asserts batch size 1, so batch by prompt, not by conditioning
image. Nothing else in the repo batches — `save_image` writes `x[0]` only.

## Offload pattern (fits 24 GB)

Move each stage on and off the GPU around its use, exactly as the CLIs do:

```python
t5, clip = t5.to(device), clip.to(device)
inp = prepare(t5, clip, x, prompt)
t5, clip = t5.cpu(), clip.cpu(); torch.cuda.empty_cache()

model = model.to(device)
x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)
model.cpu(); torch.cuda.empty_cache()

ae.decoder.to(x.device)
x = ae.decode(unpack(x.float(), h, w))
```

Load everything with `device="cpu"` up front when offloading.

## Custom sampling

- **Image-to-image / strength:** start from a partially noised latent and slice the schedule —
  `demo_gr.py` does `t_idx = int((1 - strength) * num_steps)`, then
  `x = t * noise + (1 - t) * ae.encode(init_image)` with `t = timesteps[t_idx]`, and denoises with
  `timesteps[t_idx:]`.
- **Step callbacks / previews:** `denoise` has no hook; copy its ~25-line loop from
  `src/flux/sampling.py` and yield inside it.
- **True CFG:** not supported. `[dev]`-family models are guidance-*distilled* — `guidance` is an
  embedding fed to the transformer, not a two-pass classifier-free guidance. There is no negative
  prompt.
- **Schedule:** `get_schedule(..., base_shift=0.5, max_shift=1.15, shift=True)`; pass `shift=False`
  to disable the resolution-dependent shift (schnell does).
