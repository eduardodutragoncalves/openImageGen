"""Owns the loaded model and the act of replacing it.

Swapping checkpoints on a live server is not a setting change: it unloads tens
of gigabytes, replans placement for a different architecture, and loads again,
which takes minutes and cannot overlap a running denoise loop. So it is a
state, with phases, that the whole API reports honestly:

    loading    the first model is coming up after a cold start
    ready      a model is resident and jobs run
    switching  the queue is drained and weights are being replaced
    error      nothing is loaded and requests will fail until this is fixed

A failed swap is expected to be survivable: the previous model is reloaded and
the failure is reported, rather than leaving the server with no model at all.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from .config import Settings
from .devices import plan_placement
from .engines import BaseEngine, create_engine
from .jobs import JobQueue
from .models_registry import CATALOG, ModelSpec, by_id, spec_for_repo

logger = logging.getLogger(__name__)

# A cold load of the 4-bit FLUX.2 pair takes minutes on a warm cache and much
# longer when the weights still have to be downloaded.
LOAD_TIMEOUT_S = 3600
# How long to let a running generation finish before giving up on a swap.
DRAIN_TIMEOUT_S = 1800


@dataclass
class ModelStatus:
    state: str = "loading"  # loading | ready | switching | error
    model_id: str | None = None
    target_id: str | None = None
    phase: str = "starting"
    progress: float = 0.0
    detail: str | None = None
    started: int = field(default_factory=lambda: int(time.time()))
    finished: int | None = None

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "model_id": self.model_id,
            "target_id": self.target_id,
            "phase": self.phase,
            "progress": round(self.progress, 4),
            "detail": self.detail,
            "started": self.started,
            "finished": self.finished,
        }


class ModelBusy(RuntimeError):
    """A load or swap is already running."""


class UnknownModel(ValueError):
    """No catalog entry and no usable repo id."""


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._queue: JobQueue | None = None
        self._engine: BaseEngine | None = None
        self._lock = threading.Lock()
        self._busy = threading.Event()
        self._ready = threading.Event()
        self.status = ModelStatus()
        self.initial_spec = spec_for_repo(settings.repo_id)

    # ----------------------------------------------------------------- wiring
    def attach_queue(self, queue: JobQueue) -> None:
        self._queue = queue

    @property
    def engine(self) -> BaseEngine | None:
        return self._engine

    @property
    def spec(self) -> ModelSpec:
        return self._engine.spec if self._engine is not None else self.initial_spec

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout=timeout)

    # ------------------------------------------------------------------- load
    def start_initial_load(self) -> None:
        threading.Thread(
            target=self._load_into_place,
            args=(self.initial_spec, None),
            name="model-loader",
            daemon=True,
        ).start()

    def switch(self, model_ref: str) -> ModelStatus:
        """Begin replacing the loaded model. Returns immediately."""
        spec = by_id(model_ref)
        if spec is None:
            if "/" not in model_ref:
                raise UnknownModel(
                    f"{model_ref!r} is not in the catalog; pass a catalog id or a "
                    "huggingface repo id like 'black-forest-labs/FLUX.1-schnell'"
                )
            spec = spec_for_repo(model_ref)

        if self._busy.is_set():
            raise ModelBusy(
                f"a model operation is already running ({self.status.phase}); "
                "wait for it to finish"
            )
        if self._engine is not None and self._engine.spec.id == spec.id and self._engine.loaded:
            raise ModelBusy(f"{spec.label} is already loaded")

        previous = self._engine.spec if self._engine is not None else None
        threading.Thread(
            target=self._load_into_place,
            args=(spec, previous),
            name="model-switch",
            daemon=True,
        ).start()
        # The caller polls; give it the state it is about to observe rather
        # than whatever the worker thread has managed to set by now.
        self.status = ModelStatus(
            state="switching" if previous is not None else "loading",
            model_id=previous.id if previous else None,
            target_id=spec.id,
            phase="draining the queue",
        )
        return self.status

    # ---------------------------------------------------------------- interns
    def _load_into_place(self, spec: ModelSpec, previous: ModelSpec | None) -> None:
        self._busy.set()
        switching = previous is not None
        self.status = ModelStatus(
            state="switching" if switching else "loading",
            model_id=previous.id if previous else None,
            target_id=spec.id,
            phase="draining the queue" if switching else "starting",
        )
        if switching:
            self._ready.clear()

        try:
            if switching:
                self._drain()
                self._set_phase("unloading " + (previous.label if previous else "model"), 0.05)
                self._teardown()

            self._activate(spec)
            self.status.state = "ready"
            self.status.model_id = spec.id
            self.status.target_id = None
            self.status.phase = "ready"
            self.status.progress = 1.0
            self.status.finished = int(time.time())
            self._ready.set()
            logger.info("model ready: %s", spec.label)

        except Exception as exc:  # noqa: BLE001 - reported through the API
            detail = f"{type(exc).__name__}: {exc}"
            logger.exception("loading %s failed", spec.label)
            if previous is not None and self._restore(previous, detail):
                return
            self._engine = None
            self.status.state = "error"
            self.status.model_id = None
            self.status.target_id = None
            self.status.phase = "failed"
            self.status.detail = detail
            self.status.finished = int(time.time())
            # Unblock anything waiting: it needs the error, not a deadlock.
            self._ready.set()
        finally:
            self._busy.clear()
            if self._queue is not None:
                self._queue.resume()

    def _restore(self, previous: ModelSpec, failure: str) -> bool:
        """Put the old model back so a bad swap does not cost the service."""
        logger.warning("restoring %s after a failed swap", previous.label)
        self._set_phase(f"swap failed, restoring {previous.label}", 0.1)
        self._teardown()
        try:
            self._activate(previous)
        except Exception:  # noqa: BLE001
            logger.exception("restoring %s failed too", previous.label)
            return False
        self.status.state = "ready"
        self.status.model_id = previous.id
        self.status.target_id = None
        self.status.phase = "ready"
        self.status.progress = 1.0
        self.status.detail = (
            f"{failure} — the previous model was restored, so generation still works. "
            "A restart may be needed before the swap can succeed."
        )
        self.status.finished = int(time.time())
        self._ready.set()
        return True

    def _activate(self, spec: ModelSpec) -> None:
        engine = create_engine(self.settings, spec)
        engine.on_phase = self._set_phase
        engine.load()
        with self._lock:
            self._engine = engine

    def _teardown(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.unload()

    def _drain(self) -> None:
        if self._queue is None:
            return
        self._queue.pause()
        if not self._queue.wait_idle(timeout=DRAIN_TIMEOUT_S):
            raise RuntimeError(
                "a generation is still running after "
                f"{DRAIN_TIMEOUT_S // 60} minutes; refusing to unload under it"
            )

    def _set_phase(self, label: str, progress: float) -> None:
        self.status.phase = label
        self.status.progress = progress

    # -------------------------------------------------------------- catalogue
    def catalogue(self) -> list[dict]:
        """Every known model, with whether this machine can actually run it.

        A model that does not fit is still listed, with the reason. Hiding it
        would answer "why can't I pick X?" with silence.
        """
        entries = []
        current_id = self._engine.spec.id if self._engine is not None else None
        known = {spec.id for spec in CATALOG}
        specs = list(CATALOG)
        if self.spec.id not in known:
            # Whatever OIG_REPO_ID pointed at belongs in the list too.
            specs.insert(0, self.spec)

        for spec in specs:
            plan = plan_placement(
                spec.repo_id,
                transformer_device=self.settings.transformer_device,
                text_encoder_device=self.settings.text_encoder_device,
                cpu_offload=self.settings.cpu_offload,
                max_pixels=self.settings.max_pixels,
                transformer_vram_gb=spec.transformer_vram_gb,
                text_encoder_vram_gb=spec.text_encoder_vram_gb,
            )
            entries.append(
                {
                    "id": spec.id,
                    "repo_id": spec.repo_id,
                    "family": spec.family,
                    "label": spec.label,
                    "summary": spec.summary,
                    "licence": spec.licence,
                    "licence_url": spec.licence_url,
                    "commercial_use": spec.commercial_use,
                    "capabilities": list(spec.capabilities),
                    "default_steps": spec.default_steps,
                    "default_guidance": spec.default_guidance,
                    "step_range": list(spec.step_range),
                    "guidance_range": list(spec.guidance_range),
                    "transformer_vram_gb": spec.transformer_vram_gb,
                    "text_encoder_vram_gb": spec.text_encoder_vram_gb,
                    "total_vram_gb": round(spec.total_vram_gb, 1),
                    "gated": spec.gated,
                    "notes": spec.notes,
                    "custom": spec.custom,
                    "loaded": spec.id == current_id,
                    "placement": plan.placement,
                    "placement_reason": plan.reason,
                    "runnable": plan.placement != "none",
                    "max_pixels": plan.max_pixels,
                }
            )
        return entries
