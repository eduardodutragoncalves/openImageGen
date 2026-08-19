"""Runtime configuration.

Everything is overridable through environment variables (or a .env file) so the
same deployment adapts to a single small GPU or to a multi-GPU host.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="OIG_",
        extra="ignore",
    )

    # ---------------------------------------------------------------- models
    repo_id: str = Field(
        default="diffusers/FLUX.2-dev-bnb-4bit",
        description=(
            "Diffusers repo holding the FLUX.2 [dev] weights. The bnb-4bit repo "
            "quantizes both the 32B transformer and the Mistral text encoder; "
            "black-forest-labs/FLUX.2-dev is the bf16 original (needs 80GB+)."
        ),
    )

    # ------------------------------------------------------------- placement
    # All of these are auto-detected at startup (see app/devices.py). Set them
    # only to override the heuristics.
    transformer_device: str | None = None
    text_encoder_device: str | None = None
    cpu_offload: bool | None = None
    # Rough footprints used by the planner; override for checkpoints it does
    # not know about.
    transformer_vram_gb: float | None = None
    text_encoder_vram_gb: float | None = None

    # ------------------------------------------------------------- inference
    default_steps: int = 50
    default_guidance: float = 4.0
    default_width: int = 1024
    default_height: int = 1024
    # None means "derive from the VRAM left after the weights are loaded".
    max_pixels: int | None = None
    max_reference_images: int = 4

    # ------------------------------------------------------------------ jobs
    queue_max_size: int = 32
    job_ttl_seconds: int = 3600
    # One replica: the DiT spans both cards, so generations run one at a time.
    workers: int = 1

    # ---------------------------------------------------------------- safety
    enable_nsfw_filter: bool = True
    nsfw_threshold: float = 0.85
    nsfw_model: str = "Falconsai/nsfw_image_detection"
    # Copyright / public-figure screening reuses the Mistral encoder already
    # resident on the text-encoder GPU, so it costs no extra VRAM.
    enable_integrity_filter: bool = False

    # ------------------------------------------------------------ upsampling
    openrouter_api_key: str | None = None
    openrouter_model: str = "mistralai/pixtral-large-2411"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ------------------------------------------------------------------- api
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: list[str] = Field(default_factory=list)
    output_dir: Path = PROJECT_ROOT / "output"
    # Skip model loading; useful to exercise the HTTP layer without a GPU.
    dry_run: bool = False

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, v: object) -> object:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @property
    def watermark_enabled(self) -> bool:
        # invisible-watermark is optional; see app/safety.py.
        return True


@lru_cache
def get_settings() -> Settings:
    return Settings()
