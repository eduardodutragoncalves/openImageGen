"""Content filtering.

Two independent layers:

* an NSFW image classifier (small, ~0.4GB) applied to every produced image;
* an optional integrity screen for copyright / public-figure requests that
  reuses the Mistral encoder already resident on the text-encoder GPU, so it
  costs no extra VRAM.

The integrity screen constrains generation to the single tokens "yes"/"no",
which is the same trick Black Forest Labs uses in their reference CLIs.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)

INTEGRITY_SYSTEM = (
    "You are a strict content reviewer. Answer with exactly one word: yes or no."
)

INTEGRITY_TEXT_PROMPT = """Task: decide whether an image-generation prompt asks for protected content.

Answer "yes" when the prompt:
- names a character from a copyrighted work (film, series, comic, game);
- names or unmistakably describes a real public figure;
- requests a trademarked logo or brand mark.

Answer "no" in every other case, including when you cannot name the specific
work or person involved. Generic demographic description is not enough.

Prompt to review:
-----
{prompt}
-----

Does this prompt request protected content? Answer yes or no."""

INTEGRITY_IMAGE_PROMPT = """Task: decide whether this image contains protected content.

Answer "yes" when the image shows a recognizable copyrighted character, a
trademarked logo, or a recognizable real public figure. You must be able to
name the specific work or person. Otherwise answer "no"."""


class NsfwFilter:
    """Wraps the Falconsai NSFW classifier used by the FLUX reference code."""

    def __init__(self, model_id: str, device: str, threshold: float = 0.85) -> None:
        from transformers import pipeline

        # transformers wants an int index for CUDA devices.
        pipeline_device = int(device.split(":")[1]) if device.startswith("cuda:") else device
        self._pipe = pipeline("image-classification", model=model_id, device=pipeline_device)
        self.threshold = threshold

    def score(self, image: Image.Image) -> float:
        results = self._pipe(image)
        return next((r["score"] for r in results if r["label"] == "nsfw"), 0.0)

    def is_flagged(self, image: Image.Image) -> bool:
        return self.score(image) > self.threshold


class IntegrityFilter:
    """Yes/no screen for copyright and public-figure requests."""

    def __init__(self, model, processor) -> None:
        self._model = model
        self._processor = processor
        tokenizer = getattr(processor, "tokenizer", processor)
        yes_no = tokenizer.encode(["yes", "no"], add_special_tokens=False)
        # Some tokenizers return a flat list, others a list of lists.
        flat = [t[0] if isinstance(t, (list, tuple)) else t for t in yes_no]
        self._yes_token, self._no_token = flat[0], flat[1]

    def _logits_processor(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        yes = scores[:, self._yes_token].clone()
        no = scores[:, self._no_token].clone()
        scores[:, :] = scores.min() - 1
        scores[:, self._yes_token] = yes
        scores[:, self._no_token] = no
        return scores

    @torch.no_grad()
    def _ask(self, chat: list[dict]) -> bool:
        inputs = self._processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._model.dtype)

        generated = self._model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self._logits_processor],
            do_sample=False,
        )
        return generated[0, -1].item() == self._yes_token

    def check_text(self, prompt: str) -> bool:
        chat = [
            {"role": "system", "content": [{"type": "text", "text": INTEGRITY_SYSTEM}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": INTEGRITY_TEXT_PROMPT.format(prompt=prompt)}],
            },
        ]
        try:
            return self._ask(chat)
        except Exception:  # noqa: BLE001 - never fail a request because of the filter
            logger.exception("integrity text check failed; allowing prompt")
            return False

    def check_image(self, image: Image.Image) -> bool:
        # 512^2 pixels is plenty for this decision and keeps the check cheap.
        width, height = image.size
        factor = (512**2 / max(1, width * height)) ** 0.5
        if factor < 1:
            image = image.resize((max(1, int(width * factor)), max(1, int(height * factor))))

        chat = [
            {"role": "system", "content": [{"type": "text", "text": INTEGRITY_SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INTEGRITY_IMAGE_PROMPT},
                    {"type": "image", "image": image},
                ],
            },
        ]
        try:
            return self._ask(chat)
        except Exception:  # noqa: BLE001
            logger.exception("integrity image check failed; allowing image")
            return False
