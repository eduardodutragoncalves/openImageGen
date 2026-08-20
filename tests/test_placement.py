"""Where the operator asks a model to go, and what refuses that.

The planner is good at fitting the largest checkpoint onto whatever hardware it
finds, and that is exactly why it cannot be the only voice: splitting a model
across both cards is what makes FLUX.2 [dev] runnable, and it is also what
stops you doing anything else with the second one. These cover the override and
the refusals that keep it honest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TWO_CARDS = [(0, "RTX 3090", 24.0), (1, "RTX 3090", 24.0)]
ONE_CARD = [(0, "RTX 3090", 24.0)]


@pytest.fixture()
def two_gpus(monkeypatch):
    monkeypatch.setattr("app.devices.available_gpus", lambda: list(TWO_CARDS))


@pytest.fixture()
def one_gpu(monkeypatch):
    monkeypatch.setattr("app.devices.available_gpus", lambda: list(ONE_CARD))


def _plan(choice=None, transformer=7.0, encoder=3.5):
    from app.devices import plan_placement

    return plan_placement(
        "flux1-schnell",
        transformer_vram_gb=transformer,
        text_encoder_vram_gb=encoder,
        choice=choice,
    )


class TestTheDefault:
    def test_two_cards_are_split_without_being_asked(self, two_gpus):
        plan = _plan()
        assert plan.placement == "split"
        assert plan.transformer_device == "cuda:0"
        assert plan.text_encoder_device == "cuda:1"

    def test_an_auto_choice_changes_nothing(self, two_gpus):
        from app.devices import PlacementChoice

        assert _plan(PlacementChoice()).reason == _plan().reason


class TestPinningToOneCard:
    def test_the_whole_model_lands_on_the_card_that_was_named(self, two_gpus):
        from app.devices import PlacementChoice

        plan = _plan(PlacementChoice("single", 1))
        assert plan.placement == "single"
        assert plan.transformer_device == "cuda:1"
        assert plan.text_encoder_device == "cuda:1"
        # The reason is what the UI shows, so it has to say which card and why.
        assert "GPU 1" in plan.reason and "request" in plan.reason

    def test_it_leaves_the_other_card_alone(self, two_gpus):
        from app.devices import PlacementChoice

        plan = _plan(PlacementChoice("single", 0))
        assert plan.uses_two_gpus is False

    def test_a_card_that_does_not_exist_is_refused(self, two_gpus):
        from app.devices import PlacementChoice

        with pytest.raises(ValueError, match="GPU 7 is not available"):
            _plan(PlacementChoice("single", 7))

    def test_without_a_number_it_takes_the_roomiest(self, two_gpus):
        from app.devices import PlacementChoice

        assert _plan(PlacementChoice("single")).transformer_device == "cuda:0"


class TestForcingASplit:
    def test_it_names_both_cards(self, two_gpus):
        from app.devices import PlacementChoice

        plan = _plan(PlacementChoice("split"))
        assert plan.placement == "split"
        assert (plan.transformer_device, plan.text_encoder_device) == ("cuda:0", "cuda:1")

    def test_one_card_cannot_be_split_across(self, one_gpu):
        from app.devices import PlacementChoice

        with pytest.raises(ValueError, match="two GPUs"):
            _plan(PlacementChoice("split"))


class TestHeadroom:
    """A pinned load asks a different question of the hardware."""

    def _manager(self, monkeypatch, free):
        from app.config import Settings
        from app.model_manager import ModelManager

        monkeypatch.setattr("app.model_manager.free_vram_gb", lambda: list(free))
        return ModelManager(Settings(_env_file=None))

    def test_a_pinned_load_measures_the_card_it_was_pinned_to(self, monkeypatch):
        from app.devices import PlacementChoice
        from app.models_registry import by_id

        # GPU 0 is busy, GPU 1 is empty. Split would fit; whole-on-0 does not.
        manager = self._manager(monkeypatch, [4.0, 24.0])
        spec = by_id("flux1-schnell")
        with pytest.raises(RuntimeError, match="GPU 0 has 4.0GB free"):
            manager._require_headroom(spec, PlacementChoice("single", 0))

        # The same load onto the free card is fine.
        manager._require_headroom(spec, PlacementChoice("single", 1))

    def test_without_a_choice_the_freest_card_is_the_measure(self, monkeypatch):
        from app.models_registry import by_id

        manager = self._manager(monkeypatch, [4.0, 24.0])
        manager._require_headroom(by_id("flux1-schnell"), None)


class TestTheLoadRoute:
    def test_a_placement_reaches_the_manager(self, client, monkeypatch):
        seen = {}

        def _switch(self, model_ref, choice=None):
            seen["model"] = model_ref
            seen["choice"] = choice
            return self.status

        monkeypatch.setattr("app.model_manager.ModelManager.switch", _switch)
        response = client.post(
            "/v1/models/load",
            json={"model": "flux1-schnell", "placement": "single", "device": 1},
        )
        assert response.status_code == 202
        assert seen["model"] == "flux1-schnell"
        assert seen["choice"].mode == "single"
        assert seen["choice"].device == 1

    def test_the_default_is_still_automatic(self, client, monkeypatch):
        seen = {}

        def _switch(self, model_ref, choice=None):
            seen["choice"] = choice
            return self.status

        monkeypatch.setattr("app.model_manager.ModelManager.switch", _switch)
        client.post("/v1/models/load", json={"model": "flux1-schnell"})
        assert seen["choice"].is_auto is True

    def test_an_impossible_placement_is_a_bad_request_not_a_failed_load(
        self, client, monkeypatch
    ):
        def _switch(self, model_ref, choice=None):
            raise ValueError("GPU 9 is not available; this machine offers 0, 1")

        monkeypatch.setattr("app.model_manager.ModelManager.switch", _switch)
        response = client.post(
            "/v1/models/load",
            json={"model": "flux1-schnell", "placement": "single", "device": 9},
        )
        # 422, not 500: the request was answerable and the answer is no.
        assert response.status_code == 422
        assert "GPU 9" in response.json()["detail"]

    def test_a_negative_card_never_reaches_the_planner(self, client):
        response = client.post(
            "/v1/models/load",
            json={"model": "flux1-schnell", "placement": "single", "device": -1},
        )
        assert response.status_code == 422
