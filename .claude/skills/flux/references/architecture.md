# Architecture

## Shapes end to end (1024×1024, `flux-dev`)

```
pixels        (1, 3, 1024, 1024)
AE encode  →  (1, 16, 128, 128)          8× downsample, z_channels=16
pack 2×2   →  (1, 4096, 64)              seq_len = (h/16)·(w/16), in_channels = 16·2·2
transformer→  (1, 4096, 64)              out_channels = 64
unpack     →  (1, 16, 128, 128)
AE decode  →  (1, 3, 1024, 1024)
```

`get_noise` allocates `(1, 16, 2⌈h/16⌉, 2⌈w/16⌉)` directly in latent space — the AE is never run on
the noise. Latent scaling lives in the AE (`scale_factor=0.3611`, `shift_factor=0.1159`).

## The transformer (`model.py`, `modules/layers.py`)

`Flux` is a rectified-flow DiT, ~12B params for every FLUX.1 variant:

- `hidden_size=3072`, `num_heads=24` (head dim 128), `mlp_ratio=4.0`
- `depth=19` **DoubleStreamBlocks** — separate weights for image and text streams, joint attention
  over the concatenated `[txt, img]` sequence (MMDiT).
- `depth_single_blocks=38` **SingleStreamBlocks** — text and image concatenated into one stream,
  parallel attention+MLP (`linear1` produces qkv and MLP input; `linear2` merges them back).
- `LastLayer` with adaLN modulation → `patch_size²·out_channels`.
- `QKNorm` (RMSNorm on q and k) in every attention; LayerNorms are non-affine, all conditioning
  arrives through `Modulation` (shift/scale/gate from `SiLU(vec) → Linear`).

The modulation vector `vec` is built as:
`time_in(timestep_embedding(t, 256)) [+ guidance_in(timestep_embedding(guidance, 256))] + vector_in(clip_pooled)`.
The guidance term exists only when `params.guidance_embed` is true (all `[dev]` variants; **not**
schnell).

## Positional encoding

3-axis RoPE (`axes_dim=[16, 56, 56]`, `theta=10_000`, sums to the head dim 128) applied to the ids
tensor of shape `(B, L, 3)`:

- text tokens: ids all zero.
- image tokens: axis 0 = 0, axis 1 = row index, axis 2 = column index (in packed 2×2 units).
- **Kontext** context tokens: identical row/col grid but axis 0 = **1** — that single channel is how
  the model distinguishes the reference image from the canvas being generated.

`ids = cat(txt_ids, img_ids)`, so text always occupies the head of the sequence and is sliced back
off after the single-stream blocks.

## Text encoders (`modules/conditioner.py`)

- **T5-XXL** `google/t5-v1_1-xxl`, `last_hidden_state` → `txt` (context_in_dim 4096).
  `max_length=512` (256 for schnell). Padded to `max_length` with **`attention_mask=None`** — padding
  tokens are attended to; this is intentional upstream behaviour, don't "fix" it if you want to match
  reference outputs.
- **CLIP-L** `openai/clip-vit-large-patch14`, `pooler_output` → `vec` (vec_in_dim 768), max_length 77.

Both are wrapped by `HFEmbedder`, output cast to bfloat16.

## Autoencoder (`modules/autoencoder.py`)

Standard KL VAE: `ch=128`, `ch_mult=[1,2,4,4]`, `num_res_blocks=2`, `z_channels=16`, 8× spatial
downsample, attention at the lowest resolution. Apache-2.0, shared by every FLUX.1 checkpoint
(`ae.safetensors`). `encode` samples from the posterior (so it is stochastic) and applies
`scale_factor·(z − shift_factor)`.

## Sampling (`sampling.py`)

Flow matching with a plain Euler integrator:

```python
for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
    pred = model(img=..., timesteps=t_curr, guidance=guidance_vec, ...)
    img = img + (t_prev - t_curr) * pred
```

`get_schedule` builds `linspace(1, 0, num_steps+1)` and, when `shift=True`, warps it with
`time_shift(mu, 1.0, t) = e^mu / (e^mu + (1/t − 1))` where `mu` is linear in the image sequence
length (0.5 at 256 tokens → 1.15 at 4096). Higher resolution ⇒ more time spent at high noise.

`guidance` is a **distilled** scalar embedded into `vec`; there is no second unconditional forward
pass and no negative prompt. Schnell ignores it entirely.

## Conditioning mechanisms

| Model | Mechanism | Where it enters |
| --- | --- | --- |
| Fill | AE-encoded masked image ⊕ unfolded mask, concatenated on the channel axis (`in_channels=384` = 64 latent + 320 mask) | `denoise(img_cond=...)` → `cat(img, img_cond, dim=-1)` |
| Canny / Depth | edge or depth map → AE encode → channel concat (`in_channels=128`) | same as Fill |
| Kontext | reference image → AE encode → packed tokens **appended to the image sequence** with ids axis0=1 | `denoise(img_cond_seq=..., img_cond_seq_ids=...)`; prediction is sliced back to `img.shape[1]` |
| Redux | SigLIP `so400m-patch14-384` features → `redux_up`/SiLU/`redux_down` → 4096-d tokens **appended to `txt`** | inside `prepare_redux`, before `denoise` |

Fill's mask handling: `img_cond = img_cond * (1 - mask)` before encoding (white = area to
regenerate), then the mask is unfolded `8×8` and packed `2×2` to give 320 extra channels.

## LoRA (`modules/lora.py`)

`FluxLoraWrapper` recursively replaces **every** `nn.Linear` with `LinearLora`
(`rank=128`, `lora_bias=True`, `scale` runtime-adjustable via `set_scale`). Forward is
`base(x) + scale · B(A(x))`. Used for the Canny/Depth LoRA variants, which load the base
`flux1-dev.safetensors` and then the adapter over it. `optionally_expand_state_dict` zero-pads
`img_in.weight` when a 64-channel base checkpoint is loaded into a 128-channel control model.
