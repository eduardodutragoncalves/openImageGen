"""FLUX.2 backend.

One Mistral3 vision-language model does the text encoding, which is also what
makes local prompt upsampling and the integrity filter free in VRAM terms:
they reuse an encoder that is already resident.
"""

from __future__ import annotations

import logging

import torch

from ..safety import IntegrityFilter
from ..upsampler import LocalUpsampler
from .base import BaseEngine

logger = logging.getLogger(__name__)


class Flux2Engine(BaseEngine):
    """FLUX.2 [dev] / [klein]: Flux2Pipeline + Mistral3 text encoder."""

    def _load_pipeline(self) -> None:
        from diffusers import (
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2Pipeline,
            Flux2Transformer2DModel,
        )
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        repo = self.spec.repo_id
        plan = self.plan
        dit_device = plan.transformer_device
        te_device = plan.text_encoder_device

        # With CPU offload diffusers owns placement: the components must be
        # loaded onto the CPU first, otherwise accelerate refuses to move
        # already-dispatched modules.
        dit_map = "cpu" if plan.cpu_offload else dit_device
        te_map = "cpu" if plan.cpu_offload else te_device

        self._phase("loading transformer", 0.08)
        # A plain device string is what both diffusers and transformers
        # normalize into {"": device}; passing the dict form directly leaves a
        # str where accelerate expects a torch.device on some versions.
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map=dit_map,
        )

        self._phase("loading autoencoder", 0.55)
        vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch.bfloat16)
        if not plan.cpu_offload:
            vae = vae.to(dit_device)

        self._phase("loading text encoder", 0.60)
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map=te_map,
        )

        self._phase("loading tokenizer and scheduler", 0.90)
        processor = AutoProcessor.from_pretrained(repo, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

        # The stock pipeline derives its working device from the first module it
        # finds, which is ambiguous once components live on different cards (and
        # points at the CPU while offloading). Pin it to the compute device:
        # that is where latents, the VAE and the denoising loop must run.
        compute_device = torch.device(dit_device)

        class _PinnedDeviceFlux2Pipeline(Flux2Pipeline):
            @property
            def _execution_device(self) -> torch.device:
                return compute_device

        self.pipe = _PinnedDeviceFlux2Pipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=processor,
            transformer=transformer,
        )
        self.pipe.set_progress_bar_config(disable=True)

        if plan.cpu_offload:
            # Small-GPU path: diffusers moves each component in and out per stage.
            self.pipe.enable_model_cpu_offload(device=dit_device)

        self._phase("loading safety filters", 0.95)
        self._load_nsfw_filter(te_device)

        if self.settings.enable_integrity_filter:
            self._integrity = IntegrityFilter(text_encoder, processor)

        self._local_upsampler = LocalUpsampler(self.pipe, te_device)

    def _encode(self, prompt: str) -> dict:
        te_device = torch.device(self.plan.text_encoder_device)
        dit_device = torch.device(self.plan.transformer_device)
        prompt_embeds, _ = self.pipe.encode_prompt(prompt=prompt, device=te_device)
        # A no-op unless the encoder sits on a different card, in which case
        # this is the only tensor that crosses the PCIe boundary.
        return {"prompt_embeds": prompt_embeds.to(dit_device)}
