# Models, weights, licensing

## `flux.util.configs` keys

These strings are what `--name` and `load_flow_model()` accept.

| Key | HF repo | Main file | in_ch | guidance_embed | Use |
| --- | --- | --- | --- | --- | --- |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | `flux1-schnell.safetensors` | 64 | no | t2i, 1–4 steps |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | `flux1-dev.safetensors` | 64 | yes | t2i |
| `flux-dev-krea` | `black-forest-labs/FLUX.1-Krea-dev` | `flux1-krea-dev.safetensors` | 64 | yes | t2i, photographic/opinionated aesthetics |
| `flux-dev-kontext` | `black-forest-labs/FLUX.1-Kontext-dev` | `flux1-kontext-dev.safetensors` | 64 | yes | instruction editing |
| `flux-dev-fill` | `black-forest-labs/FLUX.1-Fill-dev` | `flux1-fill-dev.safetensors` | 384 | yes | inpaint / outpaint |
| `flux-dev-canny` | `black-forest-labs/FLUX.1-Canny-dev` | `flux1-canny-dev.safetensors` | 128 | yes | edge control |
| `flux-dev-depth` | `black-forest-labs/FLUX.1-Depth-dev` | `flux1-depth-dev.safetensors` | 128 | yes | depth control |
| `flux-dev-canny-lora` | base `FLUX.1-dev` + `FLUX.1-Canny-dev-lora` | `flux1-canny-dev-lora.safetensors` | 128 | yes | edge control, small download |
| `flux-dev-depth-lora` | base `FLUX.1-dev` + `FLUX.1-Depth-dev-lora` | `flux1-depth-dev-lora.safetensors` | 128 | yes | depth control, small download |

Not in `configs` (loaded separately): `black-forest-labs/FLUX.1-Redux-dev` →
`flux1-redux-dev.safetensors`, an adapter used on top of `flux-dev` / `flux-schnell`.

All variants share `ae.safetensors` from their respective repo, and the same
`FluxParams` skeleton (3072 hidden, 24 heads, 19 double + 38 single blocks, ≈12B params).

## Auxiliary models downloaded from HF at runtime

| Model | Pulled by | Size (bf16, approx.) |
| --- | --- | --- |
| `google/t5-v1_1-xxl` | every command | ~9.5 GB |
| `openai/clip-vit-large-patch14` | every command | ~0.25 GB |
| `Falconsai/nsfw_image_detection` | t2i, control, fill, redux, Pixtral filter | ~0.4 GB |
| `LiheYoung/depth-anything-large-hf` | `control --name flux-dev-depth*` | ~1.3 GB |
| `google/siglip-so400m-patch14-384` | `redux` | ~1.6 GB |
| `mistral-community/pixtral-12b` | `kontext` | **~25 GB** |

## VRAM (approximate, bf16)

- Transformer ≈ 24 GB, T5-XXL ≈ 9.5 GB, CLIP ≈ 0.25 GB, AE ≈ 0.3 GB → **~34 GB resident** without
  offload. Comfortable on A100 80GB / H100; tight on 40 GB.
- `--offload` keeps only one stage on the GPU at a time → works on **24 GB** cards at 1 MP, at the
  cost of PCIe transfers each step group.
- Activation cost scales with sequence length `(h/16)·(w/16)`; 1024² = 4096 tokens, 1536² = 9216.
- Kontext adds the reference tokens to the sequence (roughly doubling attention cost) *and* the
  Pixtral filter, which does not fit alongside the transformer — upstream runs it on CPU.
- TensorRT fp8/fp4 exports cut transformer memory and latency substantially; fp4 needs Blackwell.

## sha256 of the main checkpoints

```
flux1-schnell.safetensors      9403429e0052277ac2a87ad800adece5481eecefd9ed334e1f348723621d2a0a
flux1-dev.safetensors          4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7
flux1-krea-dev.safetensors     4610115bb0c89560703c892c59ac2742fa821e60ef5871b33493ba544683abd7
flux1-kontext-dev.safetensors  843a26dc765d3105dba081c30bce7b14c65b0988f9e8d14e9fbc8856a6deebd5
flux1-fill-dev.safetensors     03e289f530df51d014f48e675a9ffa2141bc003259bf5f25d75b957e920a41ca
flux1-canny-dev.safetensors    996876670169591cb412b937fbd46ea14cbed6933aef17c48a2dcd9685c98cdb
flux1-depth-dev.safetensors    41360d1662f44ca45bc1b665fe6387e91802f53911001630d970a4f8be8dac21
flux1-canny-dev-lora.safetensors 8eaa21b9c43d5e7242844deb64b8cf22ae9010f813f955ca8c05f240b8a98f7e
flux1-depth-dev-lora.safetensors 1938b38ea0fdd98080fa3e48beb2bedfbc7ad102d8b65e6614de704a46d8b907
```

(Upstream lists identical sha256 for `flux1-dev` and `flux1-krea-dev`; that is a doc copy/paste in
the repo, not something to rely on.)

## Weight resolution order

`get_checkpoint_path(repo_id, filename, env_var)`:

1. `$env_var` if set **and the file exists** (else it warns and falls through).
2. `checkpoints/<repo_id with / → _>/<filename>`.
3. `hf_hub_download` into that directory. Gated repos need `HF_TOKEN` or a cached
   `~/.cache/huggingface/token`; the CLI will interactively prompt for a token once.

Env vars: `FLUX_MODEL`, `FLUX_AE`, `FLUX_LORA`, `FLUX_REDUX`.

## Licensing

- **Apache-2.0:** `FLUX.1 [schnell]` and the autoencoder weights.
- **FLUX.1-dev Non-Commercial License:** everything else (`dev`, `Krea`, `Kontext`, `Fill`,
  `Canny`, `Depth`, the LoRAs, `Redux`). Outputs may be used commercially only under a paid BFL
  license: https://bfl.ai/pricing/licensing
- Commercial licensees must report usage. The repo implements this in
  `track_usage_via_api(name, n)` → `POST https://api.bfl.ai/v1/licenses/models/<slug>/usage` with
  header `x-key: $BFL_API_KEY`. Enable with `--track_usage`; it fires once per **saved** image
  (flagged/unsaved images are not counted). Model slugs: `flux-1-dev`, `flux-1-kontext-dev`,
  `flux-1-krea-dev`, `flux-tools` (Fill/Canny/Depth/Redux).
- Unknown model names are silently skipped by the tracker — verify the slug mapping if you add a
  variant.

## Related

- Hosted API (includes Pro/Ultra non-open models): https://docs.bfl.ai/
- Diffusers: `FluxPipeline`, `FluxControlPipeline`, `FluxFillPipeline`, `FluxKontextPipeline` —
  different code path, different defaults; don't mix guidance values between the two.
- **FLUX.2** is a separate repo (`black-forest-labs/flux2`); none of this applies to it.
