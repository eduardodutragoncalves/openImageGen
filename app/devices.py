"""Hardware detection and model placement planning.

The service adapts to whatever GPUs it finds instead of assuming a particular
machine. Three placements are possible, in order of preference:

``split``    two or more GPUs, each big enough for one heavy component: the
             transformer (plus VAE) on one card, the text encoder on another.
             Nothing is offloaded, so no PCIe traffic during sampling.

``single``   one GPU large enough for everything at once.

``offload``  one GPU too small to hold both components: diffusers moves each
             component to the GPU only while it runs. Slower, but works on
             small cards.

Any part of the plan can be overridden through the environment when the
heuristics guess wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Placement = Literal["split", "single", "offload", "none"]

# Approximate resident footprints, in GB, of the components we load.
# Keyed by a substring of the repo id; the first match wins.
_FOOTPRINTS: list[tuple[str, float, float]] = [
    # (repo substring, transformer GB, text encoder GB)
    ("flux.2-dev-bnb-4bit", 19.0, 16.0),
    ("flux.2-dev", 65.0, 48.0),  # bf16 originals
    ("klein-4b", 9.0, 9.0),
    ("klein-9b", 19.0, 17.0),
    ("klein-base-4b", 9.0, 9.0),
    ("klein-base-9b", 19.0, 17.0),
]

_DEFAULT_FOOTPRINT = (19.0, 16.0)

# Room left for activations, the VAE and allocator fragmentation.
_HEADROOM_GB = 3.5


@dataclass(frozen=True)
class DevicePlan:
    placement: Placement
    transformer_device: str
    text_encoder_device: str
    cpu_offload: bool
    max_pixels: int
    reason: str

    @property
    def uses_two_gpus(self) -> bool:
        return self.transformer_device != self.text_encoder_device


def component_footprints(repo_id: str) -> tuple[float, float]:
    """Rough VRAM needs (transformer, text encoder) for a given checkpoint."""
    lowered = repo_id.lower()
    for needle, transformer_gb, encoder_gb in _FOOTPRINTS:
        if needle in lowered:
            return transformer_gb, encoder_gb
    logger.warning(
        "unknown repo %s: assuming %.0fGB transformer / %.0fGB text encoder; "
        "override with OIG_TRANSFORMER_VRAM_GB and OIG_TEXT_ENCODER_VRAM_GB",
        repo_id,
        *_DEFAULT_FOOTPRINT,
    )
    return _DEFAULT_FOOTPRINT


def available_gpus() -> list[tuple[int, str, float]]:
    """(index, name, total GB) for every visible CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return []
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append((index, props.name, props.total_memory / 1024**3))
    return gpus


def pixel_budget(free_gb: float) -> int:
    """Largest output area that comfortably fits in the leftover VRAM.

    Attention activations grow with the token count, which is proportional to
    the pixel count, so this scales the cap with whatever memory is left after
    the weights.
    """
    if free_gb >= 8.0:
        return 1536 * 1536
    if free_gb >= 5.0:
        return 1280 * 1280
    if free_gb >= 3.0:
        return 1024 * 1024
    return 768 * 768


def plan_placement(
    repo_id: str,
    *,
    transformer_device: str | None = None,
    text_encoder_device: str | None = None,
    cpu_offload: bool | None = None,
    max_pixels: int | None = None,
    transformer_vram_gb: float | None = None,
    text_encoder_vram_gb: float | None = None,
) -> DevicePlan:
    """Choose where each component lives, honouring explicit overrides."""
    default_transformer_gb, default_encoder_gb = component_footprints(repo_id)
    need_transformer = transformer_vram_gb or default_transformer_gb
    need_encoder = text_encoder_vram_gb or default_encoder_gb

    gpus = available_gpus()

    # ---------------------------------------------------------- no GPU at all
    if not gpus:
        return DevicePlan(
            placement="none",
            transformer_device=transformer_device or "cpu",
            text_encoder_device=text_encoder_device or "cpu",
            cpu_offload=False,
            max_pixels=max_pixels or 768 * 768,
            reason="no CUDA device detected",
        )

    # Biggest card first: that is where the transformer should go.
    ranked = sorted(gpus, key=lambda g: g[2], reverse=True)
    primary_index, primary_name, primary_gb = ranked[0]
    primary = f"cuda:{primary_index}"

    # ------------------------------------------------------- explicit override
    if transformer_device or text_encoder_device or cpu_offload is not None:
        resolved_transformer = transformer_device or primary
        resolved_encoder = text_encoder_device or resolved_transformer
        offload = bool(cpu_offload)
        free = primary_gb - need_transformer - (0 if resolved_encoder != resolved_transformer else need_encoder)
        return DevicePlan(
            placement="split" if resolved_encoder != resolved_transformer else ("offload" if offload else "single"),
            transformer_device=resolved_transformer,
            text_encoder_device=resolved_encoder,
            cpu_offload=offload,
            max_pixels=max_pixels or pixel_budget(max(free, 3.0) if not offload else 6.0),
            reason="placement set explicitly through the environment",
        )

    # ------------------------------------------------- two or more usable GPUs
    if len(ranked) >= 2:
        secondary_index, secondary_name, secondary_gb = ranked[1]
        fits_split = (
            primary_gb >= need_transformer + _HEADROOM_GB and secondary_gb >= need_encoder + 1.0
        )
        if fits_split:
            free = primary_gb - need_transformer
            return DevicePlan(
                placement="split",
                transformer_device=primary,
                text_encoder_device=f"cuda:{secondary_index}",
                cpu_offload=False,
                max_pixels=max_pixels or pixel_budget(free),
                reason=(
                    f"{len(ranked)} GPUs detected: transformer on {primary_name} "
                    f"({primary_gb:.0f}GB), text encoder on {secondary_name} ({secondary_gb:.0f}GB)"
                ),
            )

    # ------------------------------------------------------------- single GPU
    if primary_gb >= need_transformer + need_encoder + _HEADROOM_GB:
        free = primary_gb - need_transformer - need_encoder
        return DevicePlan(
            placement="single",
            transformer_device=primary,
            text_encoder_device=primary,
            cpu_offload=False,
            max_pixels=max_pixels or pixel_budget(free),
            reason=f"{primary_name} ({primary_gb:.0f}GB) holds every component at once",
        )

    return DevicePlan(
        placement="offload",
        transformer_device=primary,
        text_encoder_device=primary,
        cpu_offload=True,
        max_pixels=max_pixels or pixel_budget(max(primary_gb - need_transformer, 3.0)),
        reason=(
            f"{primary_name} ({primary_gb:.0f}GB) cannot hold transformer "
            f"(~{need_transformer:.0f}GB) and text encoder (~{need_encoder:.0f}GB) "
            "together: enabling sequential CPU offload"
        ),
    )
