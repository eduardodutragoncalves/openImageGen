"""Regressions from a live session where three model swaps bricked the server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestUnloadReleasesReferences:
    def test_unload_drops_every_pipeline_component(self):
        """A component surviving in the pipeline keeps its weights resident.

        The leak that broke a live server was a few GB per swap; by the third
        switch the card was full and both the load and its recovery failed.
        """
        from app.engines.base import BaseEngine

        class _Pipe:
            def __init__(self):
                self.transformer = object()
                self.vae = object()
                self.text_encoder = object()
                self.components = {
                    "transformer": self.transformer,
                    "vae": self.vae,
                    "text_encoder": self.text_encoder,
                }

            def set_progress_bar_config(self, **_):
                pass

        class _Engine(BaseEngine):
            def _load_pipeline(self):  # pragma: no cover - not exercised here
                raise NotImplementedError

            def _encode(self, prompt):  # pragma: no cover
                raise NotImplementedError

        engine = _Engine.__new__(_Engine)
        pipe = _Pipe()
        engine.pipe = pipe
        engine.spec = type("S", (), {"label": "test"})()
        engine._nsfw = object()
        engine._integrity = object()
        engine._local_upsampler = object()
        engine._openrouter = object()
        engine._loaded = True

        BaseEngine.unload(engine)

        assert engine.pipe is None
        assert engine._loaded is False
        for name in ("transformer", "vae", "text_encoder"):
            assert getattr(pipe, name) is None, f"{name} still referenced after unload"


class TestHeadroom:
    def _manager(self, free):
        from app.config import Settings
        from app.model_manager import ModelManager

        manager = ModelManager.__new__(ModelManager)
        manager.settings = Settings(_env_file=None)
        return manager

    def test_a_load_with_no_room_is_refused_before_any_weights_move(self, monkeypatch):
        import app.model_manager as mm
        from app.models_registry import by_id

        manager = self._manager(free=[0.3, 0.2])
        monkeypatch.setattr(mm, "free_vram_gb", lambda: [0.3, 0.2])
        with pytest.raises(RuntimeError, match="restart"):
            manager._require_headroom(by_id("flux2-dev-4bit"))

    def test_a_load_that_fits_is_allowed(self, monkeypatch):
        import app.model_manager as mm
        from app.models_registry import by_id

        manager = self._manager(free=[23.0, 23.0])
        monkeypatch.setattr(mm, "free_vram_gb", lambda: [23.0, 23.0])
        manager._require_headroom(by_id("flux2-klein-4b"))

    def test_no_gpu_means_no_headroom_opinion(self, monkeypatch):
        import app.model_manager as mm
        from app.models_registry import by_id

        manager = self._manager(free=[])
        monkeypatch.setattr(mm, "free_vram_gb", lambda: [])
        manager._require_headroom(by_id("flux2-dev-4bit"))


class TestCheckpointDispatch:
    """Each FLUX.2 variant must be loaded as its own model_index describes."""

    def test_component_names_are_read_from_the_index(self):
        from app.engines.flux2 import _component_name

        index = {"text_encoder": ["transformers", "Qwen3ForCausalLM"]}
        assert _component_name(index, "text_encoder") == "Qwen3ForCausalLM"
        assert _component_name(index, "missing") == ""

    def test_an_unknown_class_falls_back_instead_of_raising(self):
        import transformers

        from app.engines.flux2 import _class_from

        default = transformers.AutoModelForCausalLM
        assert _class_from(transformers, "Qwen3ForCausalLM", default) is transformers.Qwen3ForCausalLM
        assert _class_from(transformers, "NotARealClass", default) is default
        assert _class_from(transformers, "", default) is default


class TestReleasingACard:
    """Clearing one GPU, where "one" is the whole question.

    A model is placed across cards. Dropping it from one and keeping it on the
    others would leave a pipeline that cannot encode a prompt, so the honest
    operation is all-or-nothing — and the caller has to be told which of the
    two things happened.
    """

    def _manager(self, monkeypatch, *, devices=2, plan=("cuda:0", "cuda:1"), free=(1000, 20000)):
        import threading

        from app.model_manager import ModelManager, ModelStatus

        class _Plan:
            transformer_device, text_encoder_device = plan

        class _Engine:
            def __init__(self):
                self.plan = _Plan()
                self.spec = type("S", (), {"id": "flux2-dev-4bit", "label": "FLUX.2"})()
                self.unloaded = False

            def unload(self):
                self.unloaded = True

        class _Queue:
            def __init__(self):
                self.paused = False
                self.resumed = False

            def pause(self):
                self.paused = True

            def resume(self):
                self.resumed = True

            def wait_idle(self, timeout=None):
                return True

        manager = ModelManager.__new__(ModelManager)
        manager._engine = _Engine()
        manager._queue = _Queue()
        manager._lock = threading.Lock()
        manager._busy = threading.Event()
        manager._ready = threading.Event()
        manager._closing = threading.Event()
        manager.status = ModelStatus(state="ready", model_id="flux2-dev-4bit")

        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)

        swept = {"all": 0, "one": []}
        monkeypatch.setattr(
            "app.model_manager.release_cuda_memory",
            lambda: swept.__setitem__("all", swept["all"] + 1),
        )
        monkeypatch.setattr(
            "app.model_manager.release_device_memory", lambda index: swept["one"].append(index)
        )
        # Free memory before, then after: the answer is measured across the
        # call, never inferred from what was dropped.
        readings = iter([(free[0], 24000), (free[1], 24000)])
        monkeypatch.setattr(
            "app.model_manager.device_memory_mb", lambda index: next(readings)
        )
        return manager, swept

    def test_clearing_a_card_that_holds_the_model_unloads_all_of_it(self, monkeypatch):
        manager, swept = self._manager(monkeypatch)
        engine = manager._engine

        result = manager.release_device(1)  # the text encoder's card

        assert engine.unloaded is True
        assert manager._engine is None
        assert result["unloaded_model"] == "flux2-dev-4bit"
        # Every card, not just the one asked about.
        assert swept["all"] == 1 and swept["one"] == []
        assert "every" not in result["detail"]  # the warning is the UI's job
        assert "Unloaded flux2-dev-4bit" in result["detail"]

    def test_the_queue_is_drained_first_and_resumed_after(self, monkeypatch):
        """Nothing is unloaded out from under a running generation."""
        manager, _ = self._manager(monkeypatch)
        manager.release_device(0)
        assert manager._queue.paused is True
        assert manager._queue.resumed is True

    def test_a_card_holding_no_part_of_the_model_leaves_it_alone(self, monkeypatch):
        manager, swept = self._manager(monkeypatch, plan=("cuda:0", "cuda:0"), devices=2)
        engine = manager._engine

        result = manager.release_device(1)

        assert engine.unloaded is False
        assert manager._engine is engine
        assert result["unloaded_model"] is None
        # Only that card is touched: synchronising the others would stall work
        # still running on them for nothing.
        assert swept["one"] == [1] and swept["all"] == 0
        assert manager._queue.paused is False

    def test_a_job_submitted_after_a_clear_fails_with_the_reason(self, monkeypatch):
        """Left ready on purpose: the job should fail at once, not block for
        the warm-up timeout waiting for a load nobody asked for."""
        manager, _ = self._manager(monkeypatch)
        manager.release_device(0)
        assert manager._ready.is_set() is True
        assert manager.status.state == "empty"
        assert manager.status.model_id is None
        assert "load a model" in (manager.status.detail or "").lower()

    def test_a_load_in_flight_is_refused_rather_than_raced(self, monkeypatch):
        from app.model_manager import ModelBusy

        manager, _ = self._manager(monkeypatch)
        manager._busy.set()
        with pytest.raises(ModelBusy, match="settles"):
            manager.release_device(0)

    def test_a_card_this_process_cannot_see_is_refused(self, monkeypatch):
        from app.model_manager import UnknownDevice

        manager, _ = self._manager(monkeypatch, devices=2)
        with pytest.raises(UnknownDevice, match="no cuda:5"):
            manager.release_device(5)
        with pytest.raises(UnknownDevice):
            manager.release_device(-1)

    def test_memory_another_process_owns_is_reported_as_such(self, monkeypatch):
        """The common case on a shared box, and the one where a cheerful
        "cleared" would be a lie."""
        manager, _ = self._manager(
            monkeypatch, plan=("cuda:0", "cuda:0"), free=(4000, 4000)
        )
        result = manager.release_device(1)
        assert result["freed_mb"] == 0
        assert "another process" in result["detail"]

    def test_the_queue_comes_back_even_when_the_unload_fails(self, monkeypatch):
        """A card left dirty is bad; a queue left paused is a dead server."""
        manager, _ = self._manager(monkeypatch)

        def boom():
            raise RuntimeError("the driver refused")

        manager._engine.unload = boom
        with pytest.raises(RuntimeError, match="driver refused"):
            manager.release_device(0)
        assert manager._queue.resumed is True

    def test_the_freed_figure_is_measured_not_assumed(self, monkeypatch):
        manager, _ = self._manager(monkeypatch, free=(1000, 20000))
        result = manager.release_device(0)
        assert result["freed_mb"] == 19000
        assert result["free_mb"] == 20000 and result["total_mb"] == 24000
