"""Shared engine machinery.

Everything that does not depend on which model family is loaded lives here:
content screening, prompt upsampling, the sampling loop, the dry-run path and
the placement plan. A family module supplies only what genuinely differs —
which classes to load, and how a prompt becomes pipeline kwargs.
"""

from __future__ import annotations

import gc
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import torch
from PIL import Image

from ..config import Settings
from ..devices import PlacementChoice, available_gpus, plan_placement
from ..jobs import RejectedContent
from ..models_registry import ModelSpec
from ..safety import IntegrityFilter, NsfwFilter
from ..upsampler import LocalUpsampler, OpenRouterUpsampler

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float], None]


def free_vram_gb() -> list[float]:
    """Free memory per visible CUDA device, in GB."""
    if not torch.cuda.is_available():
        return []
    out = []
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        out.append(free / 1024**3)
    return out


def release_cuda_memory() -> None:
    """Return everything the allocator is holding, on every device.

    empty_cache() alone releases the caching allocator's free blocks for the
    current device only, so a two-card placement needs the loop; the double
    collection catches reference cycles between a pipeline and its components.
    """
    gc.collect()
    gc.collect()
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def choose_precision(spec: ModelSpec) -> str:
    """bf16 when the card can hold it, NF4 when it cannot.

    FLUX.1's 11.9B transformer is ~23.8GB at bf16, which does not fit a 24GB
    card with any room for activations. Quantizing it on the way in is how
    these models are actually run on consumer hardware, and it is the
    difference between offering FLUX.1 and only appearing to.
    """
    if not spec.can_quantize:
        return "bf16"
    gpus = available_gpus()
    largest = max((gpu[2] for gpu in gpus), default=0.0)
    return "bf16" if largest >= spec.transformer_vram_gb + 2.0 else "nf4"


@dataclass
class EngineResult:
    images: list[Image.Image]
    seeds: list[int]
    width: int
    height: int
    revised_prompt: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


class BaseEngine(ABC):
    """Owns every model for one checkpoint and serializes access to the GPUs."""

    def __init__(
        self,
        settings: Settings,
        spec: ModelSpec,
        precision: str | None = None,
        choice: PlacementChoice | None = None,
    ) -> None:
        self.settings = settings
        self.spec = spec
        self.precision = precision or choose_precision(spec)
        transformer_gb, encoder_gb = spec.footprints(self.precision)
        self.plan = plan_placement(
            spec.repo_id,
            transformer_device=settings.transformer_device,
            text_encoder_device=settings.text_encoder_device,
            cpu_offload=settings.cpu_offload,
            max_pixels=settings.max_pixels,
            # The registry knows this checkpoint's real footprints; the
            # planner's substring table is only the fallback for unknown ones.
            transformer_vram_gb=settings.transformer_vram_gb or transformer_gb,
            text_encoder_vram_gb=settings.text_encoder_vram_gb or encoder_gb,
            choice=choice,
        )
        logger.info(
            "placement=%s (%s): %s", self.plan.placement, self.precision, self.plan.reason
        )
        self.pipe = None
        self._nsfw: NsfwFilter | None = None
        self._integrity: IntegrityFilter | None = None
        self._local_upsampler: LocalUpsampler | None = None
        self._openrouter: OpenRouterUpsampler | None = None
        self._loaded = False
        # Set by the model manager so a load or a swap can report where it is.
        # Loading 35GB of weights takes minutes; a spinner would be a lie the
        # rest of this service does not tell.
        self.on_phase: Callable[[str, float], None] | None = None

    # ------------------------------------------------------------------ load
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def supports_local_upsample(self) -> bool:
        """Only a family whose text encoder is itself a VLM can rewrite prompts."""
        return self._local_upsampler is not None

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
                "no CUDA device was detected. Diffusion transformers need a GPU; "
                "running one on CPU is not practical. Check your driver and torch "
                "installation (torch.cuda.is_available() must be True), or set "
                "OIG_DRY_RUN=true to exercise the API without a model."
            )

        t0 = time.perf_counter()
        self._phase("resolving weights", 0.02)
        self._load_pipeline()
        self._phase("ready", 1.0)

        if settings.openrouter_api_key:
            self._openrouter = OpenRouterUpsampler(
                settings.openrouter_api_key,
                settings.openrouter_model,
                settings.openrouter_base_url,
            )

        self._loaded = True
        logger.info("%s ready in %.1fs", self.spec.label, time.perf_counter() - t0)

    def unload(self) -> None:
        """Give the VRAM back, and say how much came back.

        Dropping `self.pipe` is not enough on its own: the pipeline holds its
        components, the filters and the upsampler hold some of the same
        modules, and a reference surviving in any one of them keeps tens of
        gigabytes resident. A swap that leaks even a few GB per cycle runs the
        card out on the third switch, which is exactly how this was found.
        """
        before = free_vram_gb()

        pipe, self.pipe = self.pipe, None
        if pipe is not None:
            for name in list(getattr(pipe, "components", {}) or {}):
                try:
                    object.__setattr__(pipe, name, None)
                except Exception:  # noqa: BLE001 - best effort per component
                    pass
        del pipe

        self._nsfw = None
        self._integrity = None
        self._local_upsampler = None
        self._openrouter = None
        self._loaded = False

        release_cuda_memory()

        after = free_vram_gb()
        if before and after:
            freed = [f"{b - a:+.1f}" for a, b in zip(before, after)]
            logger.info(
                "unloaded %s: freed %s GB, now free %s GB",
                self.spec.label,
                " / ".join(freed),
                " / ".join(f"{value:.1f}" for value in after),
            )

    def _phase(self, label: str, progress: float) -> None:
        logger.info("%s: %s", self.spec.label, label)
        if self.on_phase is not None:
            self.on_phase(label, progress)

    def _load_nsfw_filter(self, preferred_device: str) -> None:
        if not self.settings.enable_nsfw_filter:
            return
        # When memory is tight enough to need offloading, this small classifier
        # is not worth the VRAM it would hold permanently.
        device = "cpu" if self.plan.cpu_offload else preferred_device
        logger.info("loading NSFW classifier onto %s", device)
        self._nsfw = NsfwFilter(self.settings.nsfw_model, device, self.settings.nsfw_threshold)

    # -------------------------------------------------------------- describe
    def describe(self) -> dict:
        spec = self.spec
        plan = self.plan
        settings = self.settings
        return {
            "id": spec.id,
            "repo_id": spec.repo_id,
            "family": spec.family,
            "label": spec.label,
            "licence": spec.licence,
            "licence_url": spec.licence_url,
            "commercial_use": spec.commercial_use,
            "placement": plan.placement,
            "placement_reason": plan.reason,
            "transformer_device": plan.transformer_device,
            "text_encoder_device": plan.text_encoder_device,
            "cpu_offload": plan.cpu_offload,
            "precision": self.precision,
            "max_pixels": plan.max_pixels,
            "capabilities": list(spec.capabilities),
            "supports_local_upsample": self.supports_local_upsample,
            "step_range": list(spec.step_range),
            "guidance_range": list(spec.guidance_range),
            "max_reference_images": (
                settings.max_reference_images if "multi-reference" in spec.capabilities
                else (1 if spec.supports_edit else 0)
            ),
            "defaults": {
                "num_steps": self.default_steps,
                "guidance": self.default_guidance,
                "width": settings.default_width,
                "height": settings.default_height,
            },
        }

    @property
    def default_steps(self) -> int:
        """Steps are a property of the checkpoint, not of the deployment.

        OIG_DEFAULT_STEPS stays meaningful only for a checkpoint the registry
        does not know: applying a global 50 to FLUX.1 [schnell], which finishes
        in four, would quietly make every request twelve times slower than the
        model intends.
        """
        return self.settings.default_steps if self.spec.custom else self.spec.default_steps

    @property
    def default_guidance(self) -> float:
        return self.settings.default_guidance if self.spec.custom else self.spec.default_guidance

    # ------------------------------------------------------------- prompting
    def upsample(self, prompt: str, references: list[Image.Image] | None, mode: str) -> str:
        if mode == "openrouter":
            if self._openrouter is None:
                raise RuntimeError(
                    "openrouter upsampling requested but OIG_OPENROUTER_API_KEY is not set"
                )
            return self._openrouter.upsample(prompt, references)
        if mode == "local":
            if self._local_upsampler is None:
                raise RuntimeError(
                    f"{self.spec.label} has no vision-language text encoder, so local "
                    "prompt upsampling is unavailable; use 'openrouter' or 'none'"
                )
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

        if references and not self.spec.supports_edit:
            raise RejectedContent(
                f"{self.spec.label} cannot edit images; it is text-to-image only. "
                "Switch to a model with the image-edit capability."
            )

        width = width or settings.default_width
        height = height or settings.default_height
        num_steps = num_steps or self.default_steps
        guidance = self.default_guidance if guidance is None else guidance
        # A model that ignores guidance must not be handed a value that makes
        # its output silently differ from what the UI showed.
        low, high = self.spec.guidance_range
        guidance = min(max(guidance, low), high)

        # ------------------------------------------------------- input screen
        t0 = time.perf_counter()
        self._screen_prompt(prompt)
        for reference in references:
            self._screen_reference(reference)
        timings["screen_input_s"] = time.perf_counter() - t0

        if settings.dry_run:
            return self._dry_run_result(
                width, height, num_steps, num_images, seed, timings, progress
            )

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
        conditioning = self._encode(effective_prompt)
        timings["encode_prompt_s"] = time.perf_counter() - t0

        # ------------------------------------------------------------ sample
        dit_device = torch.device(self.plan.transformer_device)
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
                **conditioning,
                **self._reference_kwargs(references),
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
                logger.warning(
                    "discarding generated image (seed %s): flagged by filters", image_seed
                )
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

    # ---------------------------------------------------------- family hooks
    @abstractmethod
    def _load_pipeline(self) -> None:
        """Load the components and assign self.pipe."""

    @abstractmethod
    def _encode(self, prompt: str) -> dict:
        """Encode the prompt into the kwargs this family's pipeline expects."""

    def _reference_kwargs(self, references: list[Image.Image]) -> dict:
        return {"image": list(references) or None} if references else {}

    # --------------------------------------------------------------- helpers
    def _dry_run_result(
        self,
        width: int,
        height: int,
        num_steps: int,
        num_images: int,
        seed: int | None,
        timings: dict[str, float],
        progress: ProgressCallback | None,
    ) -> EngineResult:
        """Deterministic placeholder so the HTTP layer can be tested GPU-free.

        It ticks through the same per-step progress a real run reports, at a
        configurable pace: the queue, the estimate and every live state in the
        UI are only reachable when a job takes measurable time.
        """
        base_seed = seed if seed is not None else random.randrange(2**31)
        total = max(1, num_steps * num_images)
        interval = max(0.0, self.settings.dry_run_step_seconds)

        t0 = time.perf_counter()
        images = []
        for index in range(num_images):
            rng = random.Random(base_seed + index)
            colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for step in range(num_steps):
                if interval:
                    time.sleep(interval)
                if progress is not None:
                    progress((index * num_steps + step + 1) / total)
            images.append(Image.new("RGB", (width, height), colour))

        return EngineResult(
            images=images,
            seeds=[base_seed + i for i in range(num_images)],
            width=width,
            height=height,
            revised_prompt=None,
            timings={**timings, "denoise_s": round(time.perf_counter() - t0, 3)},
        )
