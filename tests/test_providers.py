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


# --------------------------------------------------------------------- runware
# Runware's catalog is searched rather than listed, so what is exercised here is
# the request it builds and the mapping of its answer — the two places where a
# wrong assumption would show up as an empty tab rather than as an exception.

RUNWARE_RESULTS = [
    {
        "air": "bfl:flux@2-dev",
        "name": "FLUX.2 [dev]",
        "category": "checkpoint",
        "architecture": "flux2",
        "capabilities": ["text-to-image", "image-to-image"],
        "tags": ["photorealistic", "base model"],
        "shortDescription": "Black Forest Labs' open weights model.",
        "private": False,
    },
    {
        "air": "civitai:305149@392545",
        "name": "Promissing_Realistic_XL",
        "category": "checkpoint",
        "architecture": "sdxl",
        "capabilities": ["textToImage"],
        "tags": ["photorealistic"],
        "private": False,
    },
    {
        "air": "civitai:1@2",
        "name": "A Style LoRA",
        "category": "lora",
        "architecture": "sdxl",
        "capabilities": [],
        "private": False,
    },
    {
        "air": "runware:legacy@1",
        "name": "An Older Checkpoint",
        "category": "checkpoint",
        "private": False,
    },
]


class _Recorder:
    """Stands in for httpx.post and keeps what was sent."""

    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.sent = []

    def __call__(self, url, *, headers=None, json=None, timeout=None):
        self.sent.append({"url": url, "headers": headers or {}, "body": json})
        recorder = self

        class _Response:
            status_code = recorder.status

            @staticmethod
            def json():
                return recorder.payload

        return _Response()


def _runware(monkeypatch, payload, *, key="rw-key", status=200):
    from app.providers.runware import RunwareProvider

    recorder = _Recorder(payload, status)
    monkeypatch.setattr("httpx.post", recorder)
    return RunwareProvider(api_key=key), recorder


def _catalog(results, total=None):
    return {
        "data": [
            {
                "taskType": "modelSearch",
                "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada",
                "results": results,
                "totalResults": len(results) if total is None else total,
            }
        ]
    }


class TestRunwareCatalog:
    def test_the_image_filter_asks_for_checkpoints_and_drops_the_rest(self, monkeypatch):
        provider, recorder = _runware(monkeypatch, _catalog(RUNWARE_RESULTS))
        page = provider.search_catalog(query="realistic", kind="image", limit=10)

        body = recorder.sent[0]["body"][0]
        assert body["taskType"] == "modelSearch"
        assert body["search"] == "realistic"
        assert body["category"] == "checkpoint"
        assert body["limit"] == 10
        assert recorder.sent[0]["headers"]["Authorization"] == "Bearer rw-key"

        # A LoRA is an ingredient, not something you can generate with.
        assert [model.id for model in page.models] == [
            "bfl:flux@2-dev",
            "civitai:305149@392545",
            "runware:legacy@1",
        ]

    def test_capabilities_decide_what_takes_a_reference(self, monkeypatch):
        provider, _ = _runware(monkeypatch, _catalog(RUNWARE_RESULTS))
        models = {model.id: model for model in provider.search_catalog(kind="all").models}

        assert models["bfl:flux@2-dev"].reads_images is True
        # Declared text-to-image only: an edit would be refused before it costs
        # anything.
        assert models["civitai:305149@392545"].reads_images is False
        # No capabilities at all: a checkpoint generates, but nothing is assumed
        # about what it accepts.
        assert models["runware:legacy@1"].makes_images is True
        assert models["runware:legacy@1"].reads_images is False
        assert models["civitai:1@2"].makes_images is False

    def test_capability_spelling_does_not_matter(self, monkeypatch):
        entry = dict(RUNWARE_RESULTS[0], capabilities=["Text to Image", "Inpainting"])
        provider, _ = _runware(monkeypatch, _catalog([entry]))
        model = provider.search_catalog(kind="all").models[0]
        assert model.makes_images is True
        assert model.reads_images is True

    def test_the_namespaced_taxonomy_is_understood(self, monkeypatch):
        """Runware's own vocabulary is namespaced — io: for what goes in and
        out, op: for what the model does, form: for what kind of artefact it
        is. Matching the bare word against `io:text-to-image` would classify
        every model as generating nothing, and the tab would come back empty.
        """
        catalog = [
            dict(
                RUNWARE_RESULTS[0],
                air="xai:grok-imagine@image-2.0",
                capabilities=["io:text-to-image", "io:image-to-image", "form:checkpoint"],
            ),
            dict(
                RUNWARE_RESULTS[0],
                air="zai:glm@5.3",
                capabilities=["io:text-to-text", "form:checkpoint"],
            ),
            dict(
                RUNWARE_RESULTS[0],
                air="lightricks:ltx@2.5-fast",
                capabilities=["io:text-to-video", "io:image-to-video", "form:checkpoint"],
            ),
            dict(
                RUNWARE_RESULTS[0],
                air="topazlabs:wonder@3.5",
                capabilities=["op:upscale", "form:checkpoint"],
            ),
        ]
        provider, _ = _runware(monkeypatch, _catalog(catalog))
        models = {model.id: model for model in provider.search_catalog(kind="all").models}

        assert models["xai:grok-imagine@image-2.0"].makes_images is True
        assert models["xai:grok-imagine@image-2.0"].reads_images is True
        # A text model, a video model and an upscaler all return something, but
        # none of them returns a picture of what you asked for.
        assert models["zai:glm@5.3"].makes_images is False
        assert models["lightricks:ltx@2.5-fast"].makes_images is False
        assert models["topazlabs:wonder@3.5"].makes_images is False
        assert models["topazlabs:wonder@3.5"].reads_images is True

    def test_the_architecture_is_carried_into_what_the_operator_reads(self, monkeypatch):
        provider, _ = _runware(monkeypatch, _catalog(RUNWARE_RESULTS))
        model = provider.search_catalog(kind="all").models[0]
        assert "flux2" in model.description
        assert "Black Forest Labs" in model.description

    def test_a_searched_catalog_reports_no_total(self, monkeypatch):
        provider, _ = _runware(monkeypatch, _catalog(RUNWARE_RESULTS, total=4210))
        page = provider.search_catalog(kind="all")
        # There is no "of N" to show: the number would be the size of civitai.
        assert page.catalog_total == 0
        assert page.total == 4210

    def test_a_model_is_resolved_by_its_air(self, monkeypatch):
        provider, recorder = _runware(monkeypatch, _catalog(RUNWARE_RESULTS))
        model = provider.get_model("civitai:305149@392545")
        assert model is not None
        assert model.name == "Promissing_Realistic_XL"
        assert recorder.sent[0]["body"][0]["search"] == "civitai:305149@392545"
        assert provider.get_model("nobody:nothing@0") is None

    def test_the_catalog_cannot_be_listed_in_full(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _runware(monkeypatch, _catalog([]))
        with pytest.raises(ProviderError, match="search it instead"):
            provider.list_models()


class TestRunwareErrors:
    def _error(self, code, parameter="apiKey"):
        return {
            "errors": [
                {
                    "code": code,
                    "message": f"{code} happened",
                    "parameter": parameter,
                    "taskType": "authentication",
                }
            ]
        }

    def test_a_rejected_key_is_reported_as_one(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _runware(monkeypatch, self._error("invalidApiKey"), status=401)
        with pytest.raises(ProviderError, match="rejected the API key"):
            provider.search_catalog()

    def test_an_empty_account_says_so(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _runware(monkeypatch, self._error("insufficientCredits"), status=402)
        with pytest.raises(ProviderError, match="no credit left"):
            provider.search_catalog()

    def test_a_refused_search_term_says_what_to_do(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _runware(
            monkeypatch, self._error("invalidParameter", parameter="search"), status=400
        )
        with pytest.raises(ProviderError, match="Type a model name or an AIR id"):
            provider.search_catalog()

    def test_without_a_key_nothing_is_attempted(self, monkeypatch):
        from app.providers import ProviderError
        from app.providers.runware import RunwareProvider

        def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the catalog was requested without a key")

        monkeypatch.setattr("httpx.post", _fail)
        with pytest.raises(ProviderError, match="not public"):
            RunwareProvider(api_key=None).search_catalog()


class TestRunwareGeneration:
    def _image_payload(self):
        import base64
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "data": [
                {
                    "taskType": "imageInference",
                    "imageUUID": "ca6b2d39-5f83-47b9-b22b-71f9afc935e8",
                    "imageBase64Data": encoded,
                    "seed": 42,
                    "cost": 0.0021,
                }
            ]
        }

    def test_a_generation_asks_for_bytes_not_a_link(self, monkeypatch):
        provider, recorder = _runware(monkeypatch, self._image_payload())
        images = provider.generate(model="bfl:flux@2-dev", prompt="a lighthouse")

        body = recorder.sent[0]["body"][0]
        assert body["taskType"] == "imageInference"
        assert body["positivePrompt"] == "a lighthouse"
        # Runware drops generated files after seven days; this archive outlives
        # that, so the bytes come back inline.
        assert body["outputType"] == "base64Data"
        assert len(images) == 1 and images[0].size == (8, 8)

    def test_sizes_are_squared_up_to_what_runware_accepts(self, monkeypatch):
        provider, recorder = _runware(monkeypatch, self._image_payload())
        provider.generate(model="bfl:flux@2-dev", prompt="x", width=1200, height=90)
        body = recorder.sent[0]["body"][0]
        assert body["width"] == 1216  # nearest multiple of 64
        assert body["height"] == 128  # clamped to the floor

    def test_references_go_up_as_data_uris(self, monkeypatch):
        from PIL import Image

        provider, recorder = _runware(monkeypatch, self._image_payload())
        provider.generate(
            model="bfl:flux@2-dev", prompt="x", references=[Image.new("RGB", (32, 32))]
        )
        sent = recorder.sent[0]["body"][0]["referenceImages"]
        assert len(sent) == 1 and sent[0].startswith("data:image/jpeg;base64,")

    def test_an_answer_without_an_image_is_an_error(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _runware(monkeypatch, {"data": [{"taskType": "imageInference"}]})
        with pytest.raises(ProviderError, match="returned no image"):
            provider.generate(model="bfl:flux@2-dev", prompt="x")


class TestBothProviders:
    def test_each_offers_only_the_filters_it_can_honour(self, tmp_path):
        from app.config import Settings
        from app.providers import ProviderRegistry

        registry = ProviderRegistry(Settings(_env_file=None, state_dir=tmp_path))
        kinds = {entry["id"]: entry["kinds"] for entry in registry.list_providers()}
        assert kinds["openrouter"] == ["image", "text", "all"]
        # Runware hosts image checkpoints; a text filter there would only ever
        # come back empty.
        assert kinds["runware"] == ["image", "all"]

    def test_the_environment_key_is_found_per_provider(self, tmp_path):
        from app.config import Settings
        from app.providers import ProviderRegistry

        settings = Settings(_env_file=None, state_dir=tmp_path, runware_api_key="from-env")
        registry = ProviderRegistry(settings)
        assert registry.get("runware").key_source == "env"
        assert registry.get("openrouter").key_source == "none"

        registry.set_key("runware", "from-ui")
        assert registry.get("runware").key_source == "stored"


class TestProviderRoutes:
    """The HTTP surface, where a missing key has to read as a step rather than
    as a failure."""

    def test_every_provider_is_listed_without_its_credential(self, client):
        response = client.get("/v1/providers")
        assert response.status_code == 200
        entries = {entry["id"]: entry for entry in response.json()}
        assert set(entries) == {"openrouter", "runware"}
        for entry in entries.values():
            assert "key" not in entry and "api_key" not in entry
        assert entries["runware"]["catalog_is_public"] is False
        assert entries["runware"]["kinds"] == ["image", "all"]

    def test_a_catalog_needing_a_key_answers_409_not_502(self, client):
        response = client.get("/v1/providers/runware/models")
        assert response.status_code == 409
        assert "API key" in response.json()["detail"]

    def test_a_filter_the_provider_cannot_honour_is_refused(self, client):
        response = client.get("/v1/providers/runware/models", params={"kind": "text"})
        assert response.status_code == 422

    def test_an_unknown_provider_is_a_404(self, client):
        assert client.get("/v1/providers/nope/models").status_code == 404
        assert client.post("/v1/providers/nope/pin", json={"model_id": "x"}).status_code == 404
