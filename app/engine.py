"""FLUX.2 inference engine.

The placement of each component is decided at startup by :mod:`app.devices`
from whatever GPUs are present:

* two or more GPUs -> transformer (plus VAE) on one, text encoder on another.
  Nothing is offloaded, and only the prompt embeddings cross the device
  boundary, once per request.
* one large GPU -> everything resident on it.
* one small GPU -> diffusers' sequential CPU offload.

Nothing here assumes a specific number of cards; see ``self.plan``.
"""

from __future__ import annotations

import gc
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from PIL import Image

from .config import Settings
from .devices import plan_placement
from .jobs import RejectedContent
from .safety import IntegrityFilter, NsfwFilter
from .upsampler import LocalUpsampler, OpenRouterUpsampler

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


@dataclass
class EngineResult:
    images: list[Image.Image]
    seeds: list[int]
    width: int
    height: int
    revised_prompt: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class Flux2Engine:
    """Owns every model and serializes access to the GPUs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.plan = plan_placement(
            settings.repo_id,
            transformer_device=settings.transformer_device,
            text_encoder_device=settings.text_encoder_device,
            cpu_offload=settings.cpu_offload,
            max_pixels=settings.max_pixels,
            transformer_vram_gb=settings.transformer_vram_gb,
            text_encoder_vram_gb=settings.text_encoder_vram_gb,
        )
        logger.info("placement=%s: %s", self.plan.placement, self.plan.reason)
        self.pipe = None
        self._nsfw: NsfwFilter | None = None
        self._integrity: IntegrityFilter | None = None
        self._local_upsampler: LocalUpsampler | None = None
        self._openrouter: OpenRouterUpsampler | None = None
        self._loaded = False

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        settings = self.settings
        if settings.dry_run:
            logger.warning("OIG_DRY_RUN is set: skipping model load")
            self._loaded = True
            return

        if self.plan.placement == "none":
            raise RuntimeError(
                "no CUDA device was detected. FLUX.2 needs a GPU; running the "
                "transformer on CPU is not practical. Check your driver and "
                "torch installation (torch.cuda.is_available() must be True), "
                "or set OIG_DRY_RUN=true to exercise the API without a model."
            )

        from diffusers import (
            AutoencoderKLFlux2,
            Flux2Pipeline,
            Flux2Transformer2DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        repo = settings.repo_id
        plan = self.plan
        dit_device = plan.transformer_device
        te_device = plan.text_encoder_device

        # With CPU offload diffusers owns placement: the components must be
        # loaded onto the CPU first, otherwise accelerate refuses to move
        # already-dispatched modules.
        dit_map = "cpu" if plan.cpu_offload else dit_device
        te_map = "cpu" if plan.cpu_offload else te_device

        t0 = time.perf_counter()
        logger.info("loading transformer from %s onto %s", repo, dit_map)
        # A plain device string is what both diffusers and transformers
        # normalize into {"": device}; passing the dict form directly leaves a
        # str where accelerate expects a torch.device on some versions.
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map=dit_map,
        )

        logger.info("loading vae onto %s", dit_map)
        vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch.bfloat16)
        if not plan.cpu_offload:
            vae = vae.to(dit_device)

        logger.info("loading text encoder onto %s", te_map)
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map=te_map,
        )

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

        if settings.enable_nsfw_filter:
            # When memory is tight enough to need offloading, this small
            # classifier is not worth the VRAM it would hold permanently.
            filter_device = "cpu" if plan.cpu_offload else te_device
            logger.info("loading NSFW classifier onto %s", filter_device)
            self._nsfw = NsfwFilter(settings.nsfw_model, filter_device, settings.nsfw_threshold)

        if settings.enable_integrity_filter:
            self._integrity = IntegrityFilter(text_encoder, processor)

        self._local_upsampler = LocalUpsampler(self.pipe, te_device)
        if settings.openrouter_api_key:
            self._openrouter = OpenRouterUpsampler(
                settings.openrouter_api_key,
                settings.openrouter_model,
                settings.openrouter_base_url,
            )

        self._loaded = True
        logger.info("models ready in %.1fs", time.perf_counter() - t0)

    def unload(self) -> None:
        self.pipe = None
        self._nsfw = None
        self._integrity = None
        self._local_upsampler = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- describe
    def describe(self) -> dict:
        settings = self.settings
        plan = self.plan
        return {
            "id": settings.repo_id.rsplit("/", 1)[-1].lower(),
            "repo_id": settings.repo_id,
            "placement": plan.placement,
            "placement_reason": plan.reason,
            "transformer_device": plan.transformer_device,
            "text_encoder_device": plan.text_encoder_device,
            "cpu_offload": plan.cpu_offload,
            "max_pixels": plan.max_pixels,
            "capabilities": ["text-to-image", "image-edit", "multi-reference"],
            "defaults": {
                "num_steps": settings.default_steps,
                "guidance": settings.default_guidance,
                "width": settings.default_width,
                "height": settings.default_height,
            },
        }

    # -------------------------------------------------------------- prompting
    def upsample(self, prompt: str, references: list[Image.Image] | None, mode: str) -> str:
        if mode == "openrouter":
            if self._openrouter is None:
                raise RuntimeError(
                    "openrouter upsampling requested but OIG_OPENROUTER_API_KEY is not set"
                )
            return self._openrouter.upsample(prompt, references)
        if mode == "local":
            if self._local_upsampler is None:
                raise RuntimeError("local upsampling is unavailable")
            return self._local_upsampler.upsample(prompt, references)
        return prompt

    def _screen_prompt(self, prompt: str) -> None:
        if self._integrity is not None and self._integrity.check_text(prompt):
            raise RejectedContent(
                "prompt was flagged for copyrighted characters, brands or public figures"
            )

    def _screen_reference(self, image: Image.Image) -> None:
        if self._nsfw is not None and self._nsfw.is_flagged(image):
            raise RejectedContent("reference image was flagged as NSFW")
        if self._integrity is not None and self._integrity.check_image(image):
            raise RejectedContent("reference image was flagged for protected content")

    def _screen_output(self, image: Image.Image) -> bool:
        """Returns True when the image must be discarded."""
        if self._nsfw is not None and self._nsfw.is_flagged(image):
            return True
        if self._integrity is not None and self._integrity.check_image(image):
            return True
        return False

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        references: list[Image.Image] | None = None,
        width: int | None = None,
        height: int | None = None,
        num_steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        num_images: int = 1,
        upsample_mode: str = "none",
        progress: ProgressCallback | None = None,
    ) -> EngineResult:
        if not self._loaded:
            raise RuntimeError("engine is not loaded")

        settings = self.settings
        references = references or []
        timings: dict[str, float] = {}

        width = width or settings.default_width
        height = height or settings.default_height
        num_steps = num_steps or settings.default_steps
        guidance = settings.default_guidance if guidance is None else guidance

        # ------------------------------------------------------- input screen
        t0 = time.perf_counter()
        self._screen_prompt(prompt)
        for reference in references:
            self._screen_reference(reference)
        timings["screen_input_s"] = time.perf_counter() - t0

        if settings.dry_run:
            return self._dry_run_result(prompt, width, height, num_images, seed, timings)

        # ---------------------------------------------------------- upsample
        revised_prompt: str | None = None
        if upsample_mode != "none":
            t0 = time.perf_counter()
            revised_prompt = self.upsample(prompt, references or None, upsample_mode)
            timings["upsample_s"] = time.perf_counter() - t0
            # A rewritten prompt is new text: screen it too.
            self._screen_prompt(revised_prompt)

        effective_prompt = revised_prompt or prompt

        # -------------------------------------------- encode once, then sample
        t0 = time.perf_counter()
        te_device = torch.device(self.plan.text_encoder_device)
        dit_device = torch.device(self.plan.transformer_device)
        prompt_embeds, _ = self.pipe.encode_prompt(prompt=effective_prompt, device=te_device)
        # A no-op unless the encoder sits on a different card, in which case
        # this is the only tensor that crosses the PCIe boundary.
        prompt_embeds = prompt_embeds.to(dit_device)
        timings["encode_prompt_s"] = time.perf_counter() - t0

        # ------------------------------------------------------------ sample
        images: list[Image.Image] = []
        seeds: list[int] = []
        base_seed = seed if seed is not None else random.randrange(2**31)
        total_steps = num_steps * num_images

        t0 = time.perf_counter()
        for index in range(num_images):
            image_seed = base_seed + index
            generator = torch.Generator(device=dit_device).manual_seed(image_seed)

            def _step_callback(_pipe, step: int, _timestep, kwargs, _i=index):
                if progress is not None:
                    done = _i * num_steps + step + 1
                    progress(done / total_steps)
                return kwargs

            output = self.pipe(
                prompt_embeds=prompt_embeds,
                image=list(references) or None,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
                output_type="pil",
                callback_on_step_end=_step_callback,
            )
            images.append(output.images[0])
            seeds.append(image_seed)
            torch.cuda.empty_cache()

        timings["denoise_s"] = time.perf_counter() - t0

        # ------------------------------------------------------ output screen
        t0 = time.perf_counter()
        kept: list[Image.Image] = []
        kept_seeds: list[int] = []
        for image, image_seed in zip(images, seeds):
            if self._screen_output(image):
                logger.warning("discarding generated image (seed %s): flagged by filters", image_seed)
                continue
            kept.append(image)
            kept_seeds.append(image_seed)
        timings["screen_output_s"] = time.perf_counter() - t0

        if not kept:
            raise RejectedContent(
                "every generated image was flagged by the content filters; try another prompt"
            )

        actual_width, actual_height = kept[0].size
        return EngineResult(
            images=kept,
            seeds=kept_seeds,
            width=actual_width,
            height=actual_height,
            revised_prompt=revised_prompt,
            timings=timings,
        )

    # --------------------------------------------------------------- helpers
    def _dry_run_result(
        self,
        prompt: str,
        width: int,
        height: int,
        num_images: int,
        seed: int | None,
        timings: dict[str, float],
    ) -> EngineResult:
        """Deterministic placeholder so the HTTP layer can be tested GPU-free."""
        base_seed = seed if seed is not None else random.randrange(2**31)
        images = []
        for index in range(num_images):
            rng = random.Random(base_seed + index)
            color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            images.append(Image.new("RGB", (width, height), color))
        return EngineResult(
            images=images,
            seeds=[base_seed + i for i in range(num_images)],
            width=width,
            height=height,
            revised_prompt=None,
            timings={**timings, "denoise_s": 0.0},
        )
