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
