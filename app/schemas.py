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


class JobStatusResponse(BaseModel):
    id: str
    status: JobState
    created: int
    started: int | None = None
    finished: int | None = None
    queue_position: int | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    result: GenerationResponse | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    id: str
    repo_id: str
    placement: Literal["split", "single", "offload", "none"]
    placement_reason: str
    transformer_device: str
    text_encoder_device: str
    cpu_offload: bool
    max_pixels: int
    capabilities: list[Literal["text-to-image", "image-edit", "multi-reference"]]
    defaults: dict[str, float | int]


class GpuInfo(BaseModel):
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    role: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    queue_depth: int
    gpus: list[GpuInfo]
    detail: str | None = None
