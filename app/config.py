"""Runtime configuration.

Everything is overridable through environment variables (or a .env file) so the
same deployment adapts to a single small GPU or to a multi-GPU host.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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
    # How long a *finished* job stays in memory. History itself is on disk and
    # is governed by the retention settings below, not by this.
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

    # ------------------------------------------------- providers / web models
    # A key set through the studio is stored under state_dir and wins over
    # these; neither has to be set for the server to start.
    openrouter_api_key: str | None = None
    openrouter_model: str = "mistralai/pixtral-large-2411"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    runware_api_key: str | None = None
    runware_base_url: str = "https://api.runware.ai/v1"

    # ------------------------------------------------------------- retention
    # Generated files outlive the job record, so without a cap the gallery
    # fills the disk. Enforced after every write, oldest first.
    output_max_gb: float | None = 50.0
    output_max_age_days: int | None = None

    # ------------------------------------------------------------------- api
    host: str = "0.0.0.0"
    port: int = 8000
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # Refusing to listen on a public interface without keys is the whole of the
    # auth story for a service that answers from the internet. Set this only
    # when something in front of it (a tunnel, a reverse proxy) does the
    # authenticating.
    allow_open_access: bool = False
    output_dir: Path = PROJECT_ROOT / "output"
    state_dir: Path = PROJECT_ROOT / "state"
    # Serve the built UI from app/static when it exists.
    serve_ui: bool = True
    # Allow the Vite dev server's origin. Off in production: the built UI is
    # same-origin and needs no CORS at all.
    dev: bool = False
    dev_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    # Skip model loading; useful to exercise the HTTP layer without a GPU.
    dry_run: bool = False
    # Seconds per simulated step under OIG_DRY_RUN. A dry run that finishes
    # instantly cannot exercise progress, queue position or the ETA, which are
    # most of what the UI does while a job is alive.
    dry_run_step_seconds: float = 0.08

    @field_validator("api_keys", "dev_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept a comma-separated string from the environment.

        The fields are annotated NoDecode so pydantic-settings hands the raw
        string through instead of trying to read it as JSON first, which is
        what made OIG_API_KEYS=a,b fail before it ever reached this.
        """
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @property
    def auth_required(self) -> bool:
        return bool(self.api_keys)

    @property
    def binds_publicly(self) -> bool:
        return self.host not in ("127.0.0.1", "localhost", "::1")

    @property
    def watermark_enabled(self) -> bool:
        # invisible-watermark is optional; see app/safety.py.
        return True


@lru_cache
def get_settings() -> Settings:
    return Settings()
