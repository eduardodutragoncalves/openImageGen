"""Remote providers: the catalog filter, pinning, and credential handling.

These mock the provider's HTTP layer. The filter is the whole point of the Web
models tab — of the hundreds of models OpenRouter lists, only the handful that
output an image can generate one — so it is tested against a fixture shaped
like the real payload rather than against the live catalog.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CATALOG = {
    "data": [
        {
            "id": "google/gemini-3-pro-image",
            "name": "Gemini 3 Pro Image",
            "description": "Image generation and editing.",
            "architecture": {
                "input_modalities": ["image", "text"],
                "output_modalities": ["image", "text"],
            },
            "pricing": {"image": "0.000002", "prompt": "0.000001"},
            "context_length": 32768,
        },
        {
            "id": "openai/gpt-5-image",
            "name": "GPT-5 Image",
            "description": "Makes pictures.",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["image", "text"],
            },
            "pricing": {},
        },
        {
            "id": "anthropic/claude-sonnet-5",
            "name": "Claude Sonnet 5",
            "description": "A text model.",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {"prompt": "0.000003"},
        },
        {
            "id": "openrouter/auto",
            "name": "Auto Router",
            "description": "Picks a model per request.",
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["image", "text"],
            },
            "pricing": {},
        },
    ]
}


@pytest.fixture()
def provider(monkeypatch):
    from app.providers.openrouter import OpenRouterProvider

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return CATALOG

    monkeypatch.setattr("httpx.get", lambda *a, **k: _Response())
    return OpenRouterProvider(api_key=None)


class TestCatalogFilter:
    def test_image_models_are_the_ones_that_output_an_image(self, provider):
        models = provider.list_models()
        makers = {model.id for model in models if model.makes_images}
        assert makers == {
            "google/gemini-3-pro-image",
            "openai/gpt-5-image",
            "openrouter/auto",
        }
        assert "anthropic/claude-sonnet-5" not in makers

    def test_input_modality_says_which_can_edit(self, provider):
        models = {model.id: model for model in provider.list_models()}
        assert models["google/gemini-3-pro-image"].reads_images is True
        assert models["openai/gpt-5-image"].reads_images is False

    def test_routers_are_flagged(self, provider):
        models = {model.id: model for model in provider.list_models()}
        assert models["openrouter/auto"].is_router is True
        assert models["openai/gpt-5-image"].is_router is False

    def test_search_matches_id_name_and_description(self, provider):
        from app.providers import search

        models = provider.list_models()
        assert {m.id for m in search(models, "gemini")} == {"google/gemini-3-pro-image"}
        assert {m.id for m in search(models, "pictures")} == {"openai/gpt-5-image"}
        assert len(search(models, "")) == len(models)


class TestGenerationGuards:
    def test_generating_without_a_key_says_where_to_put_one(self, provider):
        from app.providers import ProviderError

        with pytest.raises(ProviderError, match="Web models tab"):
            provider.generate(model="google/gemini-3-pro-image", prompt="x")


class TestRegistry:
    def _registry(self, tmp_path):
        from app.config import Settings
        from app.providers import ProviderRegistry

        settings = Settings(_env_file=None, state_dir=tmp_path)
        return ProviderRegistry(settings)

    def test_a_stored_key_is_written_private_and_survives(self, tmp_path):
        registry = self._registry(tmp_path)
        registry.set_key("openrouter", "sk-or-secret")

        path = tmp_path / "providers.json"
        assert oct(path.stat().st_mode)[-3:] == "600"
        assert json.loads(path.read_text())["keys"]["openrouter"] == "sk-or-secret"

        # A fresh registry over the same directory sees it.
        assert self._registry(tmp_path).get("openrouter").configured is True

    def test_the_stored_key_wins_over_the_environment(self, tmp_path):
        from app.config import Settings
        from app.providers import ProviderRegistry

        settings = Settings(_env_file=None, state_dir=tmp_path, openrouter_api_key="from-env")
        registry = ProviderRegistry(settings)
        assert registry.get("openrouter").key_source == "env"

        registry.set_key("openrouter", "from-ui")
        assert registry.get("openrouter").key_source == "stored"
        registry.clear_key("openrouter")
        assert registry.get("openrouter").key_source == "env"

    def test_pinning_is_idempotent_and_reversible(self, tmp_path):
        from app.providers import RemoteModel

        registry = self._registry(tmp_path)
        model = RemoteModel(
            id="google/gemini-3-pro-image",
            name="Gemini 3 Pro Image",
            output_modalities=("image", "text"),
            input_modalities=("image", "text"),
        )
        registry.pin("openrouter", model)
        registry.pin("openrouter", model)
        assert len(registry.pinned()) == 1

        entry = registry.pinned()[0]
        assert entry.key == "openrouter:google/gemini-3-pro-image"
        assert entry.makes_images is True
        assert registry.unpin(entry.key) is True
        assert registry.pinned() == []
        assert registry.unpin(entry.key) is False

    def test_an_unknown_provider_is_refused(self, tmp_path):
        from app.providers import ProviderError

        registry = self._registry(tmp_path)
        with pytest.raises(ProviderError):
            registry.get("nope")
        with pytest.raises(ProviderError):
            registry.set_key("nope", "x")
