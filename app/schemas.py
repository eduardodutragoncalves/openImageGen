"""Request/response models for the public API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UpsampleMode(str, Enum):
    none = "none"
    local = "local"
    openrouter = "openrouter"


class ResponseFormat(str, Enum):
    b64_json = "b64_json"
    url = "url"


class OutputFormat(str, Enum):
    png = "png"
    jpeg = "jpeg"
    webp = "webp"


class GenerationParams(BaseModel):
    """Sampling knobs shared by text-to-image and editing."""

    prompt: str = Field(min_length=1, max_length=8000)
    width: int | None = Field(default=None, ge=256, le=2048)
    height: int | None = Field(default=None, ge=256, le=2048)
    num_steps: int | None = Field(default=None, ge=1, le=100)
    # FLUX.2 [dev] is guidance-distilled: this is an embedded scalar, not CFG.
    # There is no negative prompt.
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    num_images: int = Field(default=1, ge=1, le=4)
    upsample_prompt: UpsampleMode = UpsampleMode.none
    # Which OpenRouter model rewrites the prompt. Unset falls back to
    # OIG_OPENROUTER_MODEL.
    upsample_model: str | None = None
    # Where the image is made. None or "local" uses the loaded checkpoint on
    # this machine; a pinned key like "openrouter:google/gemini-3-pro-image"
    # makes it through that provider instead.
    model: str | None = None
    response_format: ResponseFormat = ResponseFormat.b64_json
    output_format: OutputFormat = OutputFormat.png

    @model_validator(mode="after")
    def _dimensions_multiple_of_16(self) -> "GenerationParams":
        # The FLUX.2 latent grid is height//16 x width//16; anything else
        # silently changes the output size.
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and value % 16 != 0:
                setattr(self, name, 16 * (value // 16))
        return self


class GenerationRequest(GenerationParams):
    """POST /v1/images/generations"""


class EditRequest(GenerationParams):
    """POST /v1/images/edits (JSON variant).

    `images` accepts base64 payloads or data URIs. FLUX.2 conditions on the
    VAE-encoded references appended to the image token sequence, so several
    references can be combined in one call.
    """

    images: list[str] = Field(min_length=1, max_length=4)
    match_image_size: int | None = Field(
        default=None,
        ge=0,
        description="Index of the reference image whose dimensions the output should copy.",
    )


class ImagePayload(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    seed: int
    width: int
    height: int


class GenerationResponse(BaseModel):
    id: str
    model: str
    created: int
    prompt: str
    revised_prompt: str | None = None
    images: list[ImagePayload]
    timings: dict[str, float]


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    rejected = "rejected"


class JobSubmitted(BaseModel):
    id: str
    status: JobState
    queue_position: int
    poll_url: str


class JobImage(BaseModel):
    """A produced file as the archive refers to it."""

    url: str | None = None
    seed: int
    width: int
    height: int
    # False once retention has removed the file the job produced. The row stays
    # so the prompt and settings remain recoverable.
    available: bool = True


class JobSummary(BaseModel):
    """One row of GET /v1/jobs.

    Carries everything a gallery cell renders — prompt, size, seed, model and
    the image URLs — because a grid of hundreds of rows cannot afford a
    follow-up request per cell.
    """

    id: str
    status: JobState
    kind: str
    created: int
    started: int | None = None
    finished: int | None = None
    progress: float | None = None
    prompt: str
    revised_prompt: str | None = None
    model_id: str | None = None
    model_label: str | None = None
    width: int | None = None
    height: int | None = None
    num_steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    num_images: int = 1
    upsample_mode: str | None = None
    reference_count: int = 0
    queue_position: int | None = None
    duration_s: float | None = None
    images: list[JobImage] = Field(default_factory=list)
    image_count: int = 0
    error: str | None = None


class JobPage(BaseModel):
    """A window onto the archive, with the total so the UI can page honestly."""

    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int


class JobRequest(BaseModel):
    """What was asked for, independent of what came back.

    A refused or failed job has no result, and that is exactly when the
    operator most needs the settings back to adjust and try again.
    """

    prompt: str
    kind: str = "generation"
    width: int | None = None
    height: int | None = None
    num_steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    num_images: int = 1
    upsample_mode: str | None = None
    upsample_model: str | None = None
    model_id: str | None = None
    model_label: str | None = None
    remote: bool = False
    reference_count: int = 0
    # The reference images this job was given, kept so an edit can be retried
    # without hunting for the originals again.
    references: list[JobImage] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    id: str
    status: JobState
    created: int
    started: int | None = None
    finished: int | None = None
    queue_position: int | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    request: JobRequest | None = None
    result: GenerationResponse | None = None
    error: str | None = None


Capability = Literal["text-to-image", "image-edit", "multi-reference"]
Placement = Literal["split", "single", "offload", "none"]


class ModelInfo(BaseModel):
    """The model that is loaded right now, and how it sits on the hardware."""

    id: str
    repo_id: str
    family: str
    label: str
    licence: str
    licence_url: str
    commercial_use: bool
    placement: Placement
    placement_reason: str
    transformer_device: str
    text_encoder_device: str
    cpu_offload: bool
    # "bf16", or "nf4" when the weights are quantized on the way in to fit.
    precision: str
    max_pixels: int
    capabilities: list[Capability]
    supports_local_upsample: bool
    step_range: list[int]
    guidance_range: list[float]
    max_reference_images: int
    defaults: dict[str, float | int]


class CatalogEntry(BaseModel):
    """One model this server knows about, runnable here or not."""

    id: str
    repo_id: str
    family: str
    label: str
    summary: str
    licence: str
    licence_url: str
    commercial_use: bool
    capabilities: list[Capability]
    default_steps: int
    default_guidance: float
    step_range: list[int]
    guidance_range: list[float]
    # Footprints at the precision this machine would actually load, not at the
    # checkpoint's nominal one.
    precision: str
    transformer_vram_gb: float
    text_encoder_vram_gb: float
    total_vram_gb: float
    gated: bool
    notes: str
    custom: bool
    loaded: bool
    placement: Placement
    placement_reason: str
    # False when this hardware cannot hold the model. The entry is still
    # returned: the reason is more useful than the absence.
    runnable: bool
    max_pixels: int


class ModelStatusResponse(BaseModel):
    """Where a load or a swap has got to."""

    state: Literal["loading", "ready", "switching", "error"]
    model_id: str | None = None
    target_id: str | None = None
    phase: str
    progress: float = Field(ge=0.0, le=1.0)
    detail: str | None = None
    started: int
    finished: int | None = None


class ModelSwitchRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model: str = Field(
        min_length=1,
        description="A catalog id (e.g. 'flux1-schnell') or a huggingface repo id.",
    )


class ProviderInfoResponse(BaseModel):
    id: str
    label: str
    summary: str
    docs_url: str
    key_url: str
    supports_generation: bool
    # True when a key is available, from the environment or set here. The key
    # itself is never returned.
    configured: bool
    key_source: Literal["none", "env", "stored"]
    catalog_is_public: bool


class RemoteModelInfo(BaseModel):
    id: str
    name: str
    description: str
    input_modalities: list[str]
    output_modalities: list[str]
    context_length: int | None = None
    price_image: str | None = None
    price_prompt: str | None = None
    is_router: bool
    makes_images: bool
    reads_images: bool
    pinned: bool = False


class RemoteModelPage(BaseModel):
    models: list[RemoteModelInfo]
    total: int
    # How many the provider offers before the modality filter, so the UI can
    # say "11 image generators out of 414" rather than implying that is all
    # there is.
    catalog_total: int


class PinnedModelInfo(BaseModel):
    key: str
    provider: str
    model_id: str
    label: str
    makes_images: bool
    reads_images: bool
    price_image: str | None = None


class ProviderKeyRequest(BaseModel):
    key: str = Field(min_length=1)


class PinRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = Field(min_length=1)


class GpuInfo(BaseModel):
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    role: str | None = None


class StorageInfo(BaseModel):
    used_bytes: int
    file_count: int
    max_bytes: int | None = None
    max_age_days: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error", "switching"]
    model_loaded: bool
    model: ModelStatusResponse
    queue_depth: int
    queue_active: int
    queue_paused: bool
    auth_required: bool
    gpus: list[GpuInfo]
    storage: StorageInfo
    detail: str | None = None
