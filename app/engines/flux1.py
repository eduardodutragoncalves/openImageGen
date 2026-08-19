"""FLUX.1 backend.

FLUX.1 predates FLUX.2's single vision-language encoder: it conditions on a
T5-XXL encoder for the prompt text plus a CLIP-L encoder for the pooled
vector. Three consequences shape this module:

* the "text encoder" the planner places is really two models, so both go on
  the encoder device and their combined footprint is what the registry quotes;
* there is no VLM in the pipeline, so local prompt upsampling is unavailable
  and ``supports_local_upsample`` reports False;
* editing is a different pipeline class (Kontext), not a parameter, so the
  capability recorded in the registry selects which one to load.
"""

from __future__ import annotations

import logging

import torch

from .base import BaseEngine

logger = logging.getLogger(__name__)

# T5's positional budget. Kontext and dev use the full window; schnell was
# trained at 256 and gains nothing from more.
_MAX_SEQUENCE_LENGTH = {"flux1-schnell": 256}
_DEFAULT_MAX_SEQUENCE_LENGTH = 512


class Flux1Engine(BaseEngine):
    """FLUX.1 [dev] / [schnell] / [Krea] / [Kontext]: FluxPipeline + T5 & CLIP."""

    @property
    def _max_sequence_length(self) -> int:
        return _MAX_SEQUENCE_LENGTH.get(self.spec.id, _DEFAULT_MAX_SEQUENCE_LENGTH)

    def _load_pipeline(self) -> None:
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers import FluxTransformer2DModel
        from transformers import (
            CLIPTextModel,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        if self.spec.supports_edit:
            from diffusers import FluxKontextPipeline as PipelineClass
        else:
            from diffusers import FluxPipeline as PipelineClass

        repo = self.spec.repo_id
        plan = self.plan
        dit_device = plan.transformer_device
        te_device = plan.text_encoder_device

        dit_map = "cpu" if plan.cpu_offload else dit_device
        te_map = "cpu" if plan.cpu_offload else te_device

        self._phase("loading transformer", 0.08)
        transformer = FluxTransformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map=dit_map,
        )

        self._phase("loading autoencoder", 0.55)
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=torch.bfloat16)
        if not plan.cpu_offload:
            vae = vae.to(dit_device)

        self._phase("loading T5 and CLIP encoders", 0.60)
        text_encoder = CLIPTextModel.from_pretrained(
            repo, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        if not plan.cpu_offload:
            text_encoder = text_encoder.to(te_device)
            text_encoder_2 = text_encoder_2.to(te_device)

        self._phase("loading tokenizers and scheduler", 0.90)
        tokenizer = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(repo, subfolder="tokenizer_2")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

        # Same reason as FLUX.2: with components on two cards the stock
        # pipeline would infer its device from whichever module it inspects
        # first, and the latents must live with the transformer.
        compute_device = torch.device(dit_device)

        class _PinnedDeviceFluxPipeline(PipelineClass):
            @property
            def _execution_device(self) -> torch.device:
                return compute_device

        self.pipe = _PinnedDeviceFluxPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
        self.pipe.set_progress_bar_config(disable=True)

        if plan.cpu_offload:
            self.pipe.enable_model_cpu_offload(device=dit_device)

        self._phase("loading safety filters", 0.95)
        self._load_nsfw_filter(te_device)
        # No integrity filter and no local upsampler: both need a generative
        # vision-language model, and T5 is an encoder only.

    def _encode(self, prompt: str) -> dict:
        te_device = torch.device(self.plan.text_encoder_device)
        dit_device = torch.device(self.plan.transformer_device)
        prompt_embeds, pooled_prompt_embeds, _text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=te_device,
            max_sequence_length=self._max_sequence_length,
        )
        return {
            "prompt_embeds": prompt_embeds.to(dit_device),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(dit_device),
        }

    def _reference_kwargs(self, references: list) -> dict:
        # Kontext conditions on exactly one image; the registry already caps
        # the request at one, so anything past the first would be silently
        # dropped and is refused earlier instead.
        return {"image": references[0]} if references else {}
