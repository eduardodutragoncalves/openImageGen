"""Catalog of models this server knows how to run.

The catalog is deliberately explicit rather than discovered: every entry
carries the numbers the placement planner needs (component footprints), the
licence the operator is accountable for, and the parameter ranges the model
actually honours. That is what lets the configuration surface show a model it
*cannot* run, with the reason, instead of hiding it.

Two families are supported today and they do not share a loading path:

``flux2``   ``Flux2Pipeline`` with a Mistral3 text encoder (app/engines/flux2.py)
``flux1``   ``FluxPipeline`` with T5-XXL + CLIP-L    (app/engines/flux1.py)

Adding a third family means adding an engine module and entries here; nothing
else in the service needs to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Family = Literal["flux2", "flux1"]
Capability = Literal["text-to-image", "image-edit", "multi-reference"]

FLUX_NONCOMMERCIAL = "FLUX Non-Commercial"
APACHE_2 = "Apache-2.0"

_LICENCE_URL = {
    FLUX_NONCOMMERCIAL: "https://bfl.ai/pricing/licensing",
    APACHE_2: "https://www.apache.org/licenses/LICENSE-2.0",
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything the service needs to decide whether it can run a model."""

    id: str
    repo_id: str
    family: Family
    label: str
    summary: str
    licence: str
    # Resident bf16/quantized footprints in GB, fed straight to the planner.
    transformer_vram_gb: float
    text_encoder_vram_gb: float
    capabilities: tuple[Capability, ...]
    default_steps: int
    default_guidance: float
    step_range: tuple[int, int]
    guidance_range: tuple[float, float]
    # Gated repos need an accepted licence on huggingface.co plus HF_TOKEN.
    gated: bool = False
    # When bf16 will not fit the hardware, these are the footprints after
    # on-the-fly NF4 quantization. None means we do not re-quantize this
    # checkpoint — the FLUX.2 mirror already ships at 4-bit.
    nf4_transformer_vram_gb: float | None = None
    nf4_text_encoder_vram_gb: float | None = None
    notes: str = ""
    # Set on entries the operator typed in rather than ones we ship.
    custom: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def licence_url(self) -> str:
        return _LICENCE_URL.get(self.licence, "")

    @property
    def commercial_use(self) -> bool:
        return self.licence == APACHE_2

    @property
    def supports_edit(self) -> bool:
        return "image-edit" in self.capabilities

    @property
    def total_vram_gb(self) -> float:
        return self.transformer_vram_gb + self.text_encoder_vram_gb

    @property
    def can_quantize(self) -> bool:
        return self.nf4_transformer_vram_gb is not None

    def footprints(self, precision: str) -> tuple[float, float]:
        """(transformer, text encoder) GB at the precision actually loaded."""
        if precision == "nf4" and self.can_quantize:
            return (
                self.nf4_transformer_vram_gb,  # type: ignore[return-value]
                self.nf4_text_encoder_vram_gb,  # type: ignore[return-value]
            )
        return self.transformer_vram_gb, self.text_encoder_vram_gb


# --------------------------------------------------------------------- FLUX.2
# Footprints match the table in app/devices.py, which was measured against this
# hardware; the bf16 originals are computed from parameter counts.
_FLUX2: list[ModelSpec] = [
    ModelSpec(
        id="flux2-dev-4bit",
        repo_id="diffusers/FLUX.2-dev-bnb-4bit",
        family="flux2",
        label="FLUX.2 [dev] 4-bit",
        summary=(
            "The default. Both the 32B transformer and the 24B Mistral text "
            "encoder quantized to 4-bit, which is what makes FLUX.2 fit on "
            "consumer cards at all."
        ),
        licence=FLUX_NONCOMMERCIAL,
        transformer_vram_gb=19.0,
        text_encoder_vram_gb=16.0,
        capabilities=("text-to-image", "image-edit", "multi-reference"),
        default_steps=50,
        default_guidance=4.0,
        step_range=(1, 100),
        guidance_range=(0.0, 20.0),
        notes="Guidance-distilled: `guidance` is an embedded scalar, not CFG.",
    ),
    ModelSpec(
        id="flux2-dev",
        repo_id="black-forest-labs/FLUX.2-dev",
        family="flux2",
        label="FLUX.2 [dev] bf16",
        summary=(
            "The unquantized original. Highest fidelity of the family and far "
            "beyond a consumer rig: 65GB of transformer plus 48GB of encoder."
        ),
        licence=FLUX_NONCOMMERCIAL,
        transformer_vram_gb=65.0,
        text_encoder_vram_gb=48.0,
        capabilities=("text-to-image", "image-edit", "multi-reference"),
        default_steps=50,
        default_guidance=4.0,
        step_range=(1, 100),
        guidance_range=(0.0, 20.0),
        gated=True,
        notes="Needs an accepted licence on huggingface.co and HF_TOKEN.",
    ),
    ModelSpec(
        id="flux2-klein-4b",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        family="flux2",
        label="FLUX.2 [klein] 4B",
        summary=(
            "The small, fast, Apache-2.0 member of the family — the one to run "
            "when the output has to be usable commercially. Its text encoder is "
            "a Qwen3 causal LM, so local prompt upsampling is unavailable."
        ),
        licence=APACHE_2,
        # Measured from the checkpoint: 7.3GB transformer, 7.5GB Qwen3 encoder.
        transformer_vram_gb=8.0,
        text_encoder_vram_gb=8.5,
        capabilities=("text-to-image", "image-edit", "multi-reference"),
        default_steps=28,
        default_guidance=4.0,
        step_range=(1, 100),
        guidance_range=(0.0, 20.0),
    ),
    ModelSpec(
        id="flux2-klein-9b",
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        family="flux2",
        label="FLUX.2 [klein] 9B",
        summary="The larger klein: more capable than 4B, still Apache-2.0.",
        licence=APACHE_2,
        transformer_vram_gb=19.0,
        text_encoder_vram_gb=9.0,
        capabilities=("text-to-image", "image-edit", "multi-reference"),
        default_steps=28,
        default_guidance=4.0,
        step_range=(1, 100),
        guidance_range=(0.0, 20.0),
    ),
]

# --------------------------------------------------------------------- FLUX.1
# 11.9B transformer at bf16 is ~23.8GB, T5-XXL ~9.5GB and CLIP-L ~0.25GB.
# Those numbers are why FLUX.1 [dev] does not fit a 24GB card without
# offloading, and the catalog is where that becomes visible instead of
# surprising someone mid-load.
_FLUX1: list[ModelSpec] = [
    ModelSpec(
        id="flux1-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        family="flux1",
        label="FLUX.1 [schnell]",
        summary=(
            "Timestep-distilled: four steps to a finished image, in seconds "
            "rather than minutes. Apache-2.0, so the output is yours."
        ),
        licence=APACHE_2,
        transformer_vram_gb=23.8,
        text_encoder_vram_gb=9.8,
        capabilities=("text-to-image",),
        default_steps=4,
        default_guidance=0.0,
        step_range=(1, 12),
        guidance_range=(0.0, 0.0),
        nf4_transformer_vram_gb=7.0,
        nf4_text_encoder_vram_gb=3.5,
        notes="Ignores guidance entirely; the control is disabled for this model.",
    ),
    ModelSpec(
        id="flux1-dev",
        repo_id="black-forest-labs/FLUX.1-dev",
        family="flux1",
        label="FLUX.1 [dev]",
        summary=(
            "The 12B guidance-distilled model most FLUX.1 work is built on. "
            "Slower than schnell and markedly stronger."
        ),
        licence=FLUX_NONCOMMERCIAL,
        transformer_vram_gb=23.8,
        text_encoder_vram_gb=9.8,
        capabilities=("text-to-image",),
        default_steps=28,
        default_guidance=3.5,
        step_range=(1, 100),
        guidance_range=(0.0, 10.0),
        gated=True,
        nf4_transformer_vram_gb=7.0,
        nf4_text_encoder_vram_gb=3.5,
    ),
    ModelSpec(
        id="flux1-krea-dev",
        repo_id="black-forest-labs/FLUX.1-Krea-dev",
        family="flux1",
        label="FLUX.1 [Krea dev]",
        summary=(
            "FLUX.1 [dev] retrained with Krea for photographic output that "
            "avoids the plastic 'AI look'."
        ),
        licence=FLUX_NONCOMMERCIAL,
        transformer_vram_gb=23.8,
        text_encoder_vram_gb=9.8,
        capabilities=("text-to-image",),
        default_steps=28,
        default_guidance=4.5,
        step_range=(1, 100),
        guidance_range=(0.0, 10.0),
        gated=True,
        nf4_transformer_vram_gb=7.0,
        nf4_text_encoder_vram_gb=3.5,
    ),
    ModelSpec(
        id="flux1-kontext-dev",
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        family="flux1",
        label="FLUX.1 [Kontext dev]",
        summary=(
            "The FLUX.1 editor: takes a reference image and instructions, and "
            "changes what you asked while leaving the rest alone."
        ),
        licence=FLUX_NONCOMMERCIAL,
        transformer_vram_gb=23.8,
        text_encoder_vram_gb=9.8,
        capabilities=("text-to-image", "image-edit"),
        default_steps=28,
        default_guidance=2.5,
        step_range=(1, 100),
        guidance_range=(0.0, 10.0),
        gated=True,
        nf4_transformer_vram_gb=7.0,
        nf4_text_encoder_vram_gb=3.5,
        notes="One reference image at a time; multi-reference is FLUX.2 only.",
    ),
]

CATALOG: tuple[ModelSpec, ...] = tuple(_FLUX2 + _FLUX1)

_BY_ID = {spec.id: spec for spec in CATALOG}
_BY_REPO = {spec.repo_id.lower(): spec for spec in CATALOG}

FAMILY_LABELS: dict[Family, str] = {
    "flux2": "FLUX.2",
    "flux1": "FLUX.1",
}


def by_id(model_id: str) -> ModelSpec | None:
    return _BY_ID.get(model_id)


def by_repo_id(repo_id: str) -> ModelSpec | None:
    return _BY_REPO.get(repo_id.lower())


def slug_for_repo(repo_id: str) -> str:
    """Stable id for a repo the catalog does not ship."""
    return repo_id.rsplit("/", 1)[-1].lower().replace(".", "-").replace("_", "-")


def spec_for_repo(repo_id: str, *, family: Family | None = None) -> ModelSpec:
    """Resolve a repo id to a spec, inventing a conservative one if needed.

    An operator may point OIG_REPO_ID at a checkpoint we do not ship. Rather
    than refusing, describe it honestly: the family is guessed from the name,
    the footprints fall back to the FLUX.2 4-bit defaults, and the entry is
    marked ``custom`` so the UI can say the numbers are estimates.
    """
    known = by_repo_id(repo_id)
    if known is not None:
        return known

    lowered = repo_id.lower()
    guessed: Family = family or ("flux1" if "flux.1" in lowered or "flux1" in lowered else "flux2")
    return ModelSpec(
        id=slug_for_repo(repo_id),
        repo_id=repo_id,
        family=guessed,
        label=repo_id.rsplit("/", 1)[-1],
        summary=(
            "Not in the shipped catalog. Footprints are assumed, so placement "
            "may be wrong; override with OIG_TRANSFORMER_VRAM_GB and "
            "OIG_TEXT_ENCODER_VRAM_GB if it misplaces."
        ),
        licence="unknown",
        transformer_vram_gb=19.0,
        text_encoder_vram_gb=16.0,
        capabilities=("text-to-image", "image-edit", "multi-reference")
        if guessed == "flux2"
        else ("text-to-image",),
        default_steps=50 if guessed == "flux2" else 28,
        default_guidance=4.0 if guessed == "flux2" else 3.5,
        step_range=(1, 100),
        guidance_range=(0.0, 20.0),
        custom=True,
    )
