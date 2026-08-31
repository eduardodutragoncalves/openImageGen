"""Giving the cards back when the process exits.

Three different things can be holding VRAM when a shutdown starts, and only one
of them is an engine you can call unload() on: a generation may be running, a
load may be part way through with no engine assigned yet, or a model may simply
be sitting there. The middle one is the one that used to be missed — the
lifespan asked `if manager.engine is not None`, which is exactly False while a
daemon thread is copying weights onto the card.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeEngine:
    """An engine that reports phases the way a real load does."""

    def __init__(self, spec, phases=3, phase_delay=0.05, fail=False):
        self.spec = spec
        self.plan = None
        self.loaded = False
        self.on_phase = None
        self.unloaded = 0
        self._phases = phases
        self._delay = phase_delay
        self._fail = fail

    def load(self):
        for index in range(self._phases):
            if self.on_phase is not None:
                self.on_phase(f"phase {index}", index / self._phases)
            time.sleep(self._delay)
        if self._fail:
            raise RuntimeError("the weights would not fit")
        self.loaded = True

    def unload(self):
        self.unloaded += 1
        self.loaded = False


@pytest.fixture()
def manager(monkeypatch, settings_factory, tmp_path):
    from app.model_manager import ModelManager

    return ModelManager(settings_factory(state_dir=tmp_path, dry_run=True))


class TestNothingLoaded:
    def test_shutting_down_with_no_engine_is_quiet(self, manager):
        manager.shutdown(timeout=1.0)
        assert manager.engine is None


class TestAModelSittingThere:
    def test_it_is_unloaded(self, manager, monkeypatch):
        from app.models_registry import by_id

        engine = _FakeEngine(by_id("flux1-schnell"), phases=1, phase_delay=0)
        monkeypatch.setattr("app.model_manager.create_engine", lambda *a, **k: engine)
        manager._activate(by_id("flux1-schnell"))
        assert manager.engine is engine

        manager.shutdown(timeout=2.0)
        assert engine.unloaded == 1
        assert manager.engine is None

    def test_the_allocator_is_not_swept_twice(self, manager, monkeypatch):
        """unload() already returns what the allocator is holding; a second
        sweep over both cards only costs time on the way out."""
        from app.models_registry import by_id

        sweeps = {"n": 0}
        monkeypatch.setattr(
            "app.model_manager.release_cuda_memory",
            lambda: sweeps.__setitem__("n", sweeps["n"] + 1),
        )
        engine = _FakeEngine(by_id("flux1-schnell"), phases=1, phase_delay=0)
        monkeypatch.setattr("app.model_manager.create_engine", lambda *a, **k: engine)
        manager._activate(by_id("flux1-schnell"))

        manager.shutdown(timeout=2.0)
        assert sweeps["n"] == 0, "the engine's own unload does this"


class TestALoadInFlight:
    """The case the old shutdown missed entirely."""

    def _slow_load(self, manager, monkeypatch, **kwargs):
        from app.models_registry import by_id

        spec = by_id("flux1-schnell")
        engine = _FakeEngine(spec, phases=20, phase_delay=0.05, **kwargs)
        monkeypatch.setattr("app.model_manager.create_engine", lambda *a, **k: engine)
        return spec, engine

    def test_it_abandons_itself_and_drops_what_it_had(self, manager, monkeypatch):
        spec, engine = self._slow_load(manager, monkeypatch)
        thread = threading.Thread(target=manager._load_into_place, args=(spec, None))
        thread.start()
        # Let it get part way in, the way a real load is part way in when the
        # signal arrives.
        time.sleep(0.15)
        assert manager.engine is None, "there is no engine to unload yet"

        manager.shutdown(timeout=5.0)
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        # What already reached the card is given back by _activate's own
        # cleanup, even though the load never produced an engine to hold it.
        assert engine.unloaded == 1
        assert manager.engine is None

    def test_a_phase_after_closing_raises(self, manager):
        from app.model_manager import ShuttingDown

        manager._set_phase("loading transformer", 0.1)  # fine
        manager._closing.set()
        with pytest.raises(ShuttingDown):
            manager._set_phase("loading text encoder", 0.6)

    def test_the_previous_model_is_not_put_back(self, manager, monkeypatch):
        """A load that fails normally restores what was loaded before. On the
        way out that would copy the weights straight back onto the card."""
        from app.models_registry import by_id

        previous = by_id("flux1-dev")
        spec, _ = self._slow_load(manager, monkeypatch)
        restored = {"n": 0}
        monkeypatch.setattr(
            "app.model_manager.ModelManager._restore",
            lambda self, prev, failure: restored.__setitem__("n", restored["n"] + 1) or True,
        )

        thread = threading.Thread(target=manager._load_into_place, args=(spec, previous))
        thread.start()
        time.sleep(0.15)
        manager.shutdown(timeout=5.0)
        thread.join(timeout=5.0)

        assert restored["n"] == 0
        assert manager.engine is None


class TestAGenerationRunning:
    def test_the_queue_is_paused_before_anything_is_unloaded(self, manager, monkeypatch):
        from app.jobs import JobQueue

        queue = JobQueue(lambda job: None, workers=1)
        manager.attach_queue(queue)
        order = []
        monkeypatch.setattr(
            JobQueue, "pause", lambda self: order.append("paused")
        )
        monkeypatch.setattr(
            "app.model_manager.ModelManager._teardown",
            lambda self: order.append("unloaded"),
        )
        manager.shutdown(timeout=1.0)
        assert order == ["paused", "unloaded"]

    def test_a_running_job_is_waited_for_briefly_then_given_up_on(
        self, manager, monkeypatch, caplog
    ):
        """A generation takes minutes and the process is leaving. Waiting for
        it only delays giving the card back, and the archive marks the job
        interrupted on the next start."""
        import logging

        from app.jobs import JobQueue

        queue = JobQueue(lambda job: None, workers=1)
        manager.attach_queue(queue)
        monkeypatch.setattr(JobQueue, "wait_idle", lambda self, timeout=None: False)

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="app.model_manager"):
            manager.shutdown(timeout=1.0)
        elapsed = time.monotonic() - started

        assert elapsed < 3.0, "shutdown must be bounded"
        assert "still running" in caplog.text


class TestTheLifespanDoesIt:
    """The end-to-end claim: closing the application leaves nothing of ours on
    the card. Driven through the real app rather than the shared `client`
    fixture, because what is being tested is what happens after it exits."""

    def _app(self, tmp_path, monkeypatch):
        import sys as _sys

        from fastapi.testclient import TestClient

        monkeypatch.setenv("OIG_DRY_RUN", "true")
        monkeypatch.setenv("OIG_DRY_RUN_STEP_SECONDS", "0")
        monkeypatch.setenv("OIG_API_KEYS", "alpha-key")
        monkeypatch.setenv("OIG_ENABLE_NSFW_FILTER", "false")
        monkeypatch.setenv("OIG_STATE_DIR", str(tmp_path / "state"))
        monkeypatch.setenv("OIG_OUTPUT_DIR", str(tmp_path / "output"))

        from app.config import get_settings

        get_settings.cache_clear()
        for module in [m for m in list(_sys.modules) if m.startswith("app.")]:
            del _sys.modules[module]

        from app.main import app, state

        return TestClient(app), state

    def test_the_engine_is_gone_once_the_app_closes(self, tmp_path, monkeypatch):
        client, state = self._app(tmp_path, monkeypatch)
        with client:
            client.get("/healthz")
            assert state.manager is not None
            state.manager.wait_ready(timeout=10)
            assert state.manager.engine is not None, "nothing was loaded to unload"

        # Outside the block the lifespan has shut down.
        assert state.manager.engine is None

    def test_the_queue_stops_too(self, tmp_path, monkeypatch):
        client, state = self._app(tmp_path, monkeypatch)
        with client:
            client.get("/healthz")
            assert state.queue is not None
            queue = state.queue

        assert not any(thread.is_alive() for thread in queue._threads)
