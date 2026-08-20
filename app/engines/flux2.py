"""FLUX.2 backend.

The family is not one architecture. [dev] pairs a Mistral3 vision-language
encoder with `Flux2Pipeline`; [klein] pairs a Qwen3 causal LM with
`Flux2KleinPipeline`. Rather than guess, this module reads the checkpoint's own
`model_index.json`, which names the pipeline and every component class — the
same file diffusers itself dispatches on. Hardcoding one variant made the other
instantiate a model its weights never described, which surfaced as an
out-of-memory error on load rather than as the type error it really was.
"""

from __future__ import annotations

import inspect
import json
import logging

import torch

from ..safety import IntegrityFilter
from ..upsampler import LocalUpsampler
from .base import BaseEngine

logger = logging.getLogger(__name__)

# Only a generative vision-language encoder can run the integrity filter or
# rewrite a prompt from a reference image.
_VISION_LANGUAGE_ARCHITECTURES = {"Mistral3ForConditionalGeneration"}


def _read_model_index(repo: str) -> dict:
    """The checkpoint's own description of itself, or {} when it has none."""
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(repo, "model_index.json")) as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - a mirror may omit the file
        logger.info("no model_index.json for %s (%s); falling back to configs", repo, exc)
        return {}


def _class_from(module, name: str, default):
    resolved = getattr(module, name, None) if name else None
    if resolved is None and name:
        logger.warning("%s is not exported by %s; using %s", name, module.__name__, default.__name__)
    return resolved or default


def _component_name(index: dict, key: str) -> str:
    entry = index.get(key)
    return entry[1] if isinstance(entry, list) and len(entry) > 1 else ""


def _encoder_architecture_from_config(repo: str) -> str:
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(repo, subfolder="text_encoder")
    except Exception:  # noqa: BLE001
        return ""
    architectures = getattr(config, "architectures", None) or []
    return architectures[0] if architectures else ""


class Flux2Engine(BaseEngine):
    """FLUX.2 [dev] and [klein], each loaded as its checkpoint describes."""

    def _load_pipeline(self) -> None:
        import diffusers
        import transformers
        from diffusers import (
            AutoencoderKLFlux2,
            FlowMatchEulerDiscreteScheduler,
            Flux2Pipeline,
            Flux2Transformer2DModel,
        )

        repo = self.spec.repo_id
        index = _read_model_index(repo)

        pipeline_cls = _class_from(diffusers, index.get("_class_name", ""), Flux2Pipeline)
        encoder_arch = _component_name(index, "text_encoder") or _encoder_architecture_from_config(repo)
        encoder_cls = _class_from(
            transformers, encoder_arch, transformers.AutoModelForCausalLM
        )
        tokenizer_cls = _class_from(
            transformers, _component_name(index, "tokenizer"), transformers.AutoProcessor
        )
        logger.info(
            "%s: %s with %s / %s",
            self.spec.label,
            pipeline_cls.__name__,
            encoder_arch or "?",
            tokenizer_cls.__name__,
        )

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
        text_encoder = encoder_cls.from_pretrained(
            repo,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map=te_map,
        )

        self._phase("loading tokenizer and scheduler", 0.90)
        tokenizer = tokenizer_cls.from_pretrained(repo, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")

        # The stock pipeline derives its working device from the first module it
        # finds, which is ambiguous once components live on different cards (and
        # points at the CPU while offloading). Pin it to the compute device:
        # that is where latents, the VAE and the denoising loop must run.
        compute_device = torch.device(dit_device)

        class _PinnedDevicePipeline(pipeline_cls):  # type: ignore[misc,valid-type]
            @property
            def _execution_device(self) -> torch.device:
                return compute_device

        kwargs = {
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
        }
        # klein is distilled in steps and guidance and says so in its index.
        if "is_distilled" in inspect.signature(pipeline_cls.__init__).parameters:
            kwargs["is_distilled"] = bool(index.get("is_distilled", False))

        self.pipe = _PinnedDevicePipeline(**kwargs)
        self.pipe.set_progress_bar_config(disable=True)

        if plan.cpu_offload:
            # Small-GPU path: diffusers moves each component in and out per stage.
            self.pipe.enable_model_cpu_offload(device=dit_device)

        self._phase("loading safety filters", 0.95)
        self._load_nsfw_filter(te_device)

        # Both of these drive the text encoder as a generative vision-language
        # model. A text-only causal LM cannot do it, and pretending otherwise
        # fails at request time instead of at load time.
        vision_language = encoder_arch in _VISION_LANGUAGE_ARCHITECTURES
        if self.settings.enable_integrity_filter and vision_language:
            self._integrity = IntegrityFilter(text_encoder, tokenizer)
        elif self.settings.enable_integrity_filter:
            logger.warning(
                "integrity filter unavailable on %s: %s is not a vision-language model",
                self.spec.label,
                encoder_arch,
            )
        if vision_language:
            self._local_upsampler = LocalUpsampler(self.pipe, te_device)

    def _encode(self, prompt: str) -> dict:
        te_device = torch.device(self.plan.text_encoder_device)
        dit_device = torch.device(self.plan.transformer_device)
        prompt_embeds, _ = self.pipe.encode_prompt(prompt=prompt, device=te_device)
        # A no-op unless the encoder sits on a different card, in which case
        # this is the only tensor that crosses the PCIe boundary.
        return {"prompt_embeds": prompt_embeds.to(dit_device)}
