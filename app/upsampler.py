"""Prompt upsampling.

FLUX.2 [dev] responds well to long, concrete prompts. Two backends:

* ``local``  - delegates to ``Flux2Pipeline.upsample_prompt``, which runs the
  Mistral encoder already resident on the text-encoder GPU using Black Forest
  Labs' own upsampling system messages. Free in VRAM terms, a few seconds of
  generation per request.
* ``openrouter`` - any vision model on OpenRouter, for when a different style
  of rewriting is wanted or the local GPU should stay free. Requires a key.
"""

from __future__ import annotations

import base64
import io
import logging

import httpx
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Reuse the reference system messages so both backends aim at the same target.
try:  # pragma: no cover - depends on the installed diffusers build
    from diffusers.pipelines.flux2.pipeline_flux2 import (
        SYSTEM_MESSAGE_UPSAMPLING_I2I as _I2I,
        SYSTEM_MESSAGE_UPSAMPLING_T2I as _T2I,
    )

    SYSTEM_T2I, SYSTEM_I2I = _T2I, _I2I
except Exception:  # noqa: BLE001 - fall back to equivalent instructions
    SYSTEM_T2I = (
        "You rewrite short image prompts into detailed ones for a text-to-image model. "
        "Keep every element the user asked for, add concrete visual detail (composition, "
        "lighting, materials, framing, mood), and preserve any text the image must contain, "
        "quoted exactly. Answer with the rewritten prompt only, under 200 words."
    )
    SYSTEM_I2I = (
        "You rewrite short editing instructions for an image-editing model that receives the "
        "reference image(s) alongside the instruction. Describe only the requested change and "
        "state what must stay untouched. Answer with the rewritten instruction only, under 150 words."
    )


def _to_data_uri(image: Image.Image, max_side: int = 768) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class LocalUpsampler:
    """Prompt rewriting with the resident Mistral-Small encoder.

    This deliberately does *not* call ``Flux2Pipeline.upsample_prompt``. That
    method tokenizes with ``padding="max_length", max_length=2048``, i.e. it
    right-pads the conversation before calling ``generate()``. Right padding on
    a decoder-only model makes generation continue after a long run of pad
    tokens, and the rewrite comes back as degenerate text (observed here:
    fragments of the system message echoed back instead of a prompt).

    We reuse the pipeline's official system messages and image preparation, but
    tokenize a single conversation without padding, which is what generation
    wants.
    """

    def __init__(self, pipe, device: torch.device | str) -> None:
        self._pipe = pipe
        self._device = torch.device(device) if isinstance(device, str) else device

    @torch.no_grad()
    def upsample(
        self,
        prompt: str,
        images: list[Image.Image] | None = None,
        temperature: float = 0.15,
        max_new_tokens: int = 512,
    ) -> str:
        pipe = self._pipe
        try:
            from diffusers.pipelines.flux2.pipeline_flux2 import (
                _validate_and_process_images,
                format_input,
            )
        except Exception:  # noqa: BLE001 - older/newer builds: use the stock path
            logger.warning("diffusers upsampling helpers unavailable; using pipeline method")
            rewritten = pipe.upsample_prompt(
                [prompt],
                images=[list(images)] if images else None,
                temperature=temperature,
                device=self._device,
            )
            return (rewritten[0] if rewritten else "").strip() or prompt

        if images:
            system = pipe.system_message_upsampling_i2i
            prepared = _validate_and_process_images(
                [list(images)], pipe.image_processor, pipe.upsampling_max_image_size
            )
        else:
            system = pipe.system_message_upsampling_t2i
            prepared = None

        messages = format_input(prompts=[prompt], system_message=system, images=prepared)
        inputs = pipe.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        model = pipe.text_encoder
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            use_cache=True,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        tokenizer = getattr(pipe.tokenizer, "tokenizer", pipe.tokenizer)
        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        return text or prompt


class OpenRouterUpsampler:
    """Prompt rewriting through the OpenRouter API."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def upsample(self, prompt: str, images: list[Image.Image] | None = None) -> str:
        system = SYSTEM_I2I if images else SYSTEM_T2I
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(image)}})

        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return text.strip() or prompt
