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
# Runware answers about its catalog twice: a curated document served publicly,
# and the community mirror behind the paid API. What is exercised here is which
# of the two a request reaches, and how an entry from either one is read — the
# places where a wrong assumption shows up as an empty tab rather than as an
# exception.

CURATED = [
    {
        "air": "bfl:flux@2-dev",
        "name": "FLUX.2 [dev]",
        "creator": "black-forest-labs",
        "weight": 9,
        "headline": "Black Forest Labs' open weights model.",
        "capabilities": ["io:text-to-image", "io:image-to-image", "op:edit", "form:checkpoint"],
        "coverImage": "https://assets.runware.ai/covers/flux-2-dev.jpg",
        "pricingOverview": "$0.03 per image output",
    },
    {
        "air": "ideogram:4@0",
        "name": "Ideogram 4.0",
        "creator": "ideogram",
        "weight": 7,
        "headline": "Typography that survives a render.",
        "capabilities": ["io:text-to-image", "form:checkpoint"],
    },
    {
        # Returns an image, faithfully, but not a picture of what you wrote.
        "air": "topazlabs:wonder@3.5",
        "name": "Topaz Labs Wonder 3.5",
        "creator": "topazlabs",
        "weight": 8,
        "capabilities": ["io:image-to-image", "op:upscale", "form:checkpoint"],
    },
    {
        # Declares op:edit, but everything it outputs is video.
        "air": "xai:grok-imagine@video",
        "name": "Grok Imagine Video",
        "creator": "xai",
        "weight": 8,
        "capabilities": ["io:text-to-video", "io:image-to-video", "op:edit", "form:checkpoint"],
    },
    {
        "air": "zai:glm@5.3",
        "name": "GLM-5.3",
        "creator": "zai",
        "weight": 6,
        "capabilities": ["io:text-to-text", "form:checkpoint"],
    },
]

COMMUNITY_RESULTS = [
    {
        "air": "civitai:305149@392545",
        "name": "Promissing_Realistic_XL",
        "category": "checkpoint",
        "architecture": "sdxl",
        "capabilities": ["textToImage"],
        "tags": ["photorealistic"],
        "heroImage": "https://mim.runware.ai/r/66a70a0bb7c38-450x450.jpg",
        "provider": "civitai",
    },
    {
        "air": "civitai:1@2",
        "name": "An Older Checkpoint",
        "category": "checkpoint",
        "architecture": "sd15",
    },
]


@pytest.fixture(autouse=True)
def _forget_the_curated_catalog():
    """The curated document is cached across provider instances, which is the
    point of it — but a cache that outlives a test would answer the next one."""
    from app.providers import runware as module

    module._curated_cache = (0.0, [])
    yield
    module._curated_cache = (0.0, [])


class _Recorder:
    """Stands in for an httpx verb and keeps what was sent."""

    def __init__(self, payload, status=200, raises=None):
        self.payload = payload
        self.status = status
        self.raises = raises
        self.sent = []

    def __call__(self, url, *, headers=None, json=None, timeout=None):
        self.sent.append({"url": url, "headers": headers or {}, "body": json})
        if self.raises is not None:
            raise self.raises
        recorder = self

        class _Response:
            status_code = recorder.status

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return recorder.payload

        return _Response()


def _curated(monkeypatch, entries=None, **kwargs):
    """A Runware provider whose curated catalog is the given document."""
    from app.providers.runware import RunwareProvider

    recorder = _Recorder(CURATED if entries is None else entries, **kwargs)
    monkeypatch.setattr("httpx.get", recorder)
    return RunwareProvider(api_key=kwargs.pop("key", None)), recorder


def _paid(monkeypatch, payload, *, key="rw-key", status=200):
    """A provider whose paid API answers with the given payload."""
    from app.providers.runware import RunwareProvider

    recorder = _Recorder(payload, status)
    monkeypatch.setattr("httpx.post", recorder)
    return RunwareProvider(api_key=key), recorder


def _search_payload(results, total=None):
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


class TestCuratedCatalog:
    def test_it_is_readable_without_a_key(self, monkeypatch):
        def _refuse(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the paid API was called to browse the catalog")

        monkeypatch.setattr("httpx.post", _refuse)
        provider, recorder = _curated(monkeypatch)

        page = provider.search_catalog(kind="image")
        assert recorder.sent[0]["url"] == "https://content.runware.ai/models"
        assert "Authorization" not in recorder.sent[0]["headers"]
        assert page.catalog_total == len(CURATED)

    def test_only_the_models_that_draw_are_offered(self, monkeypatch):
        provider, _ = _curated(monkeypatch)
        assert [m.id for m in provider.search_catalog(kind="image").models] == [
            "bfl:flux@2-dev",
            "ideogram:4@0",
        ]

    def test_what_returns_an_image_is_not_the_same_as_what_draws_one(self, monkeypatch):
        """The three near misses, each excluded for its own reason."""
        provider, _ = _curated(monkeypatch)
        models = {m.id: m for m in provider.search_catalog(kind="all").models}

        # An upscaler returns an image, faithfully, but not one of what you wrote.
        assert models["topazlabs:wonder@3.5"].makes_images is False
        assert models["topazlabs:wonder@3.5"].reads_images is True
        # A video model that declares op:edit edits video.
        assert models["xai:grok-imagine@video"].makes_images is False
        # And a text model is a text model.
        assert models["zai:glm@5.3"].makes_images is False

    def test_the_namespaced_taxonomy_is_understood(self, monkeypatch):
        """Runware's vocabulary is namespaced — io: for what goes in and out,
        op: for what the model does, form: for what it is. Matching the bare
        word against `io:text-to-image` classifies every model as generating
        nothing, and the tab comes back empty."""
        provider, _ = _curated(monkeypatch)
        flux = provider.get_model("bfl:flux@2-dev")
        assert flux is not None
        assert flux.makes_images is True
        assert flux.reads_images is True

    def test_the_spelling_of_a_capability_does_not_matter(self, monkeypatch):
        entry = dict(CURATED[0], capabilities=["Text to Image", "Inpainting"])
        provider, _ = _curated(monkeypatch, [entry])
        model = provider.search_catalog(kind="all").models[0]
        assert model.makes_images is True
        assert model.reads_images is True

    def test_the_catalog_is_ordered_the_way_runware_orders_it(self, monkeypatch):
        provider, _ = _curated(monkeypatch)
        # By weight, not by name: sorting alphabetically opens the list on
        # whatever happens to start with a B.
        assert [m.id for m in provider.search_catalog(kind="all").models] == [
            "bfl:flux@2-dev",  # weight 9
            "xai:grok-imagine@video",  # weight 8, and G sorts before T
            "topazlabs:wonder@3.5",  # weight 8
            "ideogram:4@0",  # weight 7
            "zai:glm@5.3",  # weight 6
        ]

    def test_a_cover_and_a_quoted_price_survive(self, monkeypatch):
        provider, _ = _curated(monkeypatch)
        flux = provider.get_model("bfl:flux@2-dev")
        assert flux.cover_image.endswith("flux-2-dev.jpg")
        assert flux.creator == "black-forest-labs"
        # Quoted as written: rewriting someone's pricing is how you misquote it.
        assert flux.price_note == "$0.03 per image output"

    def test_the_document_is_fetched_once(self, monkeypatch):
        provider, recorder = _curated(monkeypatch)
        provider.search_catalog(kind="image")
        provider.search_catalog(kind="all")
        provider.get_model("bfl:flux@2-dev")
        assert len(recorder.sent) == 1

    def test_a_held_copy_outlives_a_failed_refresh(self, monkeypatch):
        import httpx

        provider, _ = _curated(monkeypatch)
        assert provider.search_catalog(kind="image").models

        # The catalog changes when Runware ships a model, not by the minute:
        # stale beats empty.
        monkeypatch.setattr("httpx.get", _Recorder(None, raises=httpx.ConnectError("down")))
        assert provider.search_catalog(kind="image").models

    def test_an_unreachable_catalog_with_nothing_held_is_an_error(self, monkeypatch):
        import httpx

        from app.providers import ProviderError
        from app.providers.runware import RunwareProvider

        monkeypatch.setattr("httpx.get", _Recorder(None, raises=httpx.ConnectError("down")))
        with pytest.raises(ProviderError, match="could not reach"):
            RunwareProvider(api_key=None).search_catalog()


class TestCommunityMirror:
    def test_it_searches_the_paid_api_for_checkpoints(self, monkeypatch):
        provider, recorder = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS, total=4210))
        page = provider.search_catalog(query="realistic", kind="community", limit=10)

        body = recorder.sent[0]["body"][0]
        assert body["taskType"] == "modelSearch"
        assert body["search"] == "realistic"
        assert body["category"] == "checkpoint"
        assert recorder.sent[0]["headers"]["Authorization"] == "Bearer rw-key"
        assert [m.id for m in page.models] == ["civitai:305149@392545", "civitai:1@2"]

    def test_an_entry_declaring_nothing_is_taken_as_text_to_image(self, monkeypatch):
        provider, _ = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS))
        models = {m.id: m for m in provider.search_catalog(kind="community").models}
        # A checkpoint draws by definition; that it also accepts a reference is
        # not something to assume, so an edit is refused rather than billed.
        assert models["civitai:1@2"].makes_images is True
        assert models["civitai:1@2"].reads_images is False

    def test_the_mirror_reports_no_total_to_be_a_fraction_of(self, monkeypatch):
        provider, _ = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS, total=4210))
        page = provider.search_catalog(kind="community")
        assert page.total == 4210
        assert page.catalog_total == 0

    def test_it_needs_a_key(self, monkeypatch):
        from app.providers import ProviderError
        from app.providers.runware import RunwareProvider

        def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("the paid API was called without a key")

        monkeypatch.setattr("httpx.post", _fail)
        with pytest.raises(ProviderError, match="needs an API key"):
            RunwareProvider(api_key=None).search_catalog(kind="community", query="x")

    def test_a_community_pin_is_resolved_by_its_air(self, monkeypatch):
        provider, recorder = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS))
        monkeypatch.setattr("httpx.get", _Recorder(CURATED))

        model = provider.get_model("civitai:305149@392545")
        assert model is not None and model.name == "Promissing_Realistic_XL"
        # Curated first — free — and only then the paid lookup.
        assert recorder.sent[0]["body"][0]["search"] == "civitai:305149@392545"
        assert provider.get_model("nobody:nothing@0") is None


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

        provider, _ = _paid(monkeypatch, self._error("invalidApiKey"), status=401)
        with pytest.raises(ProviderError, match="rejected the API key"):
            provider.search_catalog(kind="community", query="x")

    def test_an_empty_account_says_so(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _paid(monkeypatch, self._error("insufficientCredits"), status=402)
        with pytest.raises(ProviderError, match="no credit left"):
            provider.search_catalog(kind="community", query="x")

    def test_a_refused_search_term_says_what_to_do(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _paid(
            monkeypatch, self._error("invalidParameter", parameter="search"), status=400
        )
        with pytest.raises(ProviderError, match="Type a model name or an AIR id"):
            provider.search_catalog(kind="community", query="x")


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
        provider, recorder = _paid(monkeypatch, self._image_payload())
        images = provider.generate(model="bfl:flux@2-dev", prompt="a lighthouse")

        body = recorder.sent[0]["body"][0]
        assert body["taskType"] == "imageInference"
        assert body["positivePrompt"] == "a lighthouse"
        # Runware drops generated files after seven days; this archive outlives
        # that, so the bytes come back inline.
        assert body["outputType"] == "base64Data"
        assert len(images) == 1 and images[0].size == (8, 8)

    def test_sizes_are_squared_up_to_what_runware_accepts(self, monkeypatch):
        provider, recorder = _paid(monkeypatch, self._image_payload())
        provider.generate(model="bfl:flux@2-dev", prompt="x", width=1200, height=90)
        body = recorder.sent[0]["body"][0]
        assert body["width"] == 1216  # nearest multiple of 64
        assert body["height"] == 128  # clamped to the floor

    def test_references_go_up_as_data_uris(self, monkeypatch):
        from PIL import Image

        provider, recorder = _paid(monkeypatch, self._image_payload())
        provider.generate(
            model="bfl:flux@2-dev", prompt="x", references=[Image.new("RGB", (32, 32))]
        )
        sent = recorder.sent[0]["body"][0]["referenceImages"]
        assert len(sent) == 1 and sent[0].startswith("data:image/jpeg;base64,")

    def test_an_answer_without_an_image_is_an_error(self, monkeypatch):
        from app.providers import ProviderError

        provider, _ = _paid(monkeypatch, {"data": [{"taskType": "imageInference"}]})
        with pytest.raises(ProviderError, match="returned no image"):
            provider.generate(model="bfl:flux@2-dev", prompt="x")

    def test_what_an_image_billed_travels_with_that_image(self, monkeypatch):
        """Runware prices each result, and the price has to reach the file."""
        from app.providers import image_cost

        provider, _ = _paid(monkeypatch, self._image_payload())
        images = provider.generate(model="bfl:flux@2-dev", prompt="a lighthouse")
        assert image_cost(images[0]) == 0.0021

    def test_a_batch_keeps_each_image_at_its_own_price(self, monkeypatch):
        """Two results at different prices must not be averaged into one."""
        from app.providers import image_cost

        payload = self._image_payload()
        cheap = dict(payload["data"][0], cost=0.0009)
        payload["data"].append(cheap)
        provider, _ = _paid(monkeypatch, payload)
        images = provider.generate(model="bfl:flux@2-dev", prompt="x", num_images=2)
        assert [image_cost(image) for image in images] == [0.0021, 0.0009]

    def test_an_answer_that_quotes_no_price_claims_none(self, monkeypatch):
        """No price and a free generation are different claims."""
        from app.providers import image_cost

        payload = self._image_payload()
        payload["data"][0].pop("cost")
        provider, _ = _paid(monkeypatch, payload)
        assert image_cost(provider.generate(model="bfl:flux@2-dev", prompt="x")[0]) is None


class TestSalvagingARepliedImage:
    """A live server billed twice for images it then threw away.

    Both replies were HTTP 200 carrying a finished image and its price, and
    both were rejected whole as not-JSON. The head of each was well-formed, so
    whatever broke the parse came after the first document — and an image the
    operator has already paid for is worth recovering rather than discarding.
    """

    def _reply(self, *, trailing: str = "", cost: float = 0.138) -> str:
        import base64
        import io
        import json as jsonlib

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "green").save(buffer, format="PNG")
        body = {
            "data": [
                {
                    "taskType": "imageInference",
                    "imageUUID": "f4e2b686-e79c-4d91-b267-677d695c5156",
                    "cost": cost,
                    "imageBase64Data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            ]
        }
        return jsonlib.dumps(body) + trailing

    def _provider(self, monkeypatch, text: str, status: int = 200):
        from app.config import Settings
        from app.providers.runware import RunwareProvider

        class _Response:
            status_code = status

            def __init__(self, body):
                self.text = body

            def json(self):
                import json as jsonlib

                return jsonlib.loads(self.text)

        monkeypatch.setattr("httpx.post", lambda *a, **k: _Response(text))
        Settings(_env_file=None)
        return RunwareProvider(api_key="rw-key")

    def test_a_reply_with_bytes_after_the_document_still_yields_its_image(
        self, monkeypatch
    ):
        from app.providers import image_cost

        provider = self._provider(monkeypatch, self._reply(trailing="\n{}"))
        images = provider.generate(model="google:4@1", prompt="an athlete")
        assert len(images) == 1
        # And the price the operator was charged comes with it.
        assert image_cost(images[0]) == 0.138

    def test_the_salvage_is_recorded_rather_than_done_silently(self, monkeypatch, caplog):
        import logging

        provider = self._provider(monkeypatch, self._reply(trailing="garbage"))
        with caplog.at_level(logging.WARNING, logger="app.providers.runware"):
            provider.generate(model="google:4@1", prompt="x")
        assert "recovered the first document" in caplog.text

    def test_a_truly_broken_body_says_how_long_it_was_and_what_the_parser_said(
        self, monkeypatch, caplog
    ):
        """Truncation and trailing bytes need different fixes, and only the
        length, the tail and the parser's complaint tell them apart."""
        import logging

        from app.providers import ProviderError

        cut = self._reply()[:200]  # a body that stops mid-string
        provider = self._provider(monkeypatch, cut)
        with caplog.at_level(logging.WARNING, logger="app.providers.runware"):
            with pytest.raises(ProviderError, match="could not be parsed"):
                provider.generate(model="google:4@1", prompt="x")
        assert "tail:" in caplog.text
        assert "200 bytes" in caplog.text

    def test_a_salvaged_failure_is_never_retried_on_its_own(self, monkeypatch):
        """The generation was billed; trying again spends that a second time."""
        from app.providers import ProviderError

        provider = self._provider(monkeypatch, self._reply()[:200])
        with pytest.raises(ProviderError) as caught:
            provider.generate(model="google:4@1", prompt="x")
        assert getattr(caught.value, "retryable", False) is False


class TestOpenRouterCost:
    """OpenRouter reports what a call cost in `usage`, when the key may see it."""

    def _reply(self, images=1, **usage):
        import base64
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        reply = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "images": [{"image_url": {"url": uri}} for _ in range(images)],
                    },
                }
            ]
        }
        if usage:
            reply["usage"] = usage
        return reply

    def _provider(self, monkeypatch, payload):
        from app.providers.openrouter import OpenRouterProvider

        monkeypatch.setattr("httpx.post", _Recorder(payload, 200))
        return OpenRouterProvider(api_key="sk-or-x")

    def test_the_reported_cost_reaches_the_image(self, monkeypatch):
        from app.providers import image_cost

        provider = self._provider(monkeypatch, self._reply(cost=0.0125))
        assert image_cost(provider.generate(model="m", prompt="x")[0]) == 0.0125

    def test_one_call_returning_two_images_splits_its_charge(self, monkeypatch):
        """The call is billed once; charging each image the full price doubles it."""
        from app.providers import image_cost

        provider = self._provider(monkeypatch, self._reply(images=2, cost=0.01))
        images = provider.generate(model="m", prompt="x")
        assert [image_cost(image) for image in images] == [0.005, 0.005]

    def test_a_key_that_cannot_see_prices_still_gets_its_image(self, monkeypatch):
        """Usage accounting is optional; an absent price is not a failure."""
        from app.providers import image_cost

        provider = self._provider(monkeypatch, self._reply())
        images = provider.generate(model="m", prompt="x")
        assert len(images) == 1 and image_cost(images[0]) is None

    def test_a_price_that_is_not_a_number_is_ignored(self, monkeypatch):
        from app.providers import image_cost

        provider = self._provider(monkeypatch, self._reply(cost="unavailable"))
        assert image_cost(provider.generate(model="m", prompt="x")[0]) is None


class TestBothProviders:
    def test_each_offers_only_the_filters_it_can_honour(self, tmp_path):
        from app.config import Settings
        from app.providers import ProviderRegistry

        registry = ProviderRegistry(Settings(_env_file=None, state_dir=tmp_path))
        kinds = {entry["id"]: entry["kinds"] for entry in registry.list_providers()}
        assert kinds["openrouter"] == ["image", "text", "all"]
        # Runware hosts image checkpoints, so a text filter there would only
        # ever come back empty; "community" is the mirror behind the paid API.
        assert kinds["runware"] == ["image", "all", "community"]

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
        # Runware's curated catalog is served publicly; only the community
        # mirror and generation need the credential.
        assert entries["runware"]["catalog_is_public"] is True
        assert entries["runware"]["kinds"] == ["image", "all", "community"]

    def test_a_search_that_needs_a_key_answers_409_not_502(self, client):
        # Browsing is free; reaching the community mirror is not, and a missing
        # credential is a step rather than a failure of the provider.
        response = client.get(
            "/v1/providers/runware/models", params={"kind": "community", "q": "sdxl"}
        )
        assert response.status_code == 409
        assert "API key" in response.json()["detail"]

    def test_a_filter_the_provider_cannot_honour_is_refused(self, client):
        response = client.get("/v1/providers/runware/models", params={"kind": "text"})
        assert response.status_code == 422
        response = client.get("/v1/providers/openrouter/models", params={"kind": "community"})
        assert response.status_code == 422

    def test_an_unknown_provider_is_a_404(self, client):
        assert client.get("/v1/providers/nope/models").status_code == 404
        assert client.post("/v1/providers/nope/pin", json={"model_id": "x"}).status_code == 404


class TestKeyChecking:
    """"A key is set" and "a key works" are different claims, and only the
    second one is worth showing someone about to spend money."""

    def test_no_key_is_not_a_failed_check(self, monkeypatch):
        from app.providers.runware import RunwareProvider

        def _refuse(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("a request was made with no key to send")

        monkeypatch.setattr("httpx.post", _refuse)
        result = RunwareProvider(api_key=None).check_key()
        assert result.ok is False
        assert result.detail == "no key set"

    def test_a_working_key_spends_one_cheap_call(self, monkeypatch):
        provider, recorder = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS[:1]))
        result = provider.check_key()
        assert result.ok is True
        body = recorder.sent[0]["body"][0]
        # One row, not a page: the answer is the status code, not the payload.
        assert body["taskType"] == "modelSearch" and body["limit"] == 1

    def test_runware_is_checked_against_the_paid_api_not_the_public_one(self, monkeypatch):
        """The curated catalog answers without a credential, so reading it
        proves nothing about the key."""
        provider, recorder = _paid(monkeypatch, _search_payload([]))
        monkeypatch.setattr("httpx.get", _Recorder(CURATED))
        provider.check_key()
        assert recorder.sent, "the check never touched the paid API"

    def test_a_rejected_key_is_reported_rather_than_raised(self, monkeypatch):
        provider, _ = _paid(
            monkeypatch,
            {"errors": [{"code": "invalidApiKey", "message": "no", "parameter": "apiKey"}]},
            status=401,
        )
        result = provider.check_key()
        assert result.ok is False
        assert "rejected the API key" in result.detail

    def test_an_unreachable_provider_is_a_failed_check_not_an_exception(self, monkeypatch):
        import httpx

        from app.providers.runware import RunwareProvider

        monkeypatch.setattr("httpx.post", _Recorder(None, raises=httpx.ConnectError("down")))
        result = RunwareProvider(api_key="rw-key").check_key()
        assert result.ok is False and "could not reach" in result.detail

    def test_openrouter_asks_the_endpoint_that_costs_no_tokens(self, monkeypatch):
        from app.providers.openrouter import OpenRouterProvider

        recorder = _Recorder({"data": {"label": "a key"}})
        monkeypatch.setattr("httpx.get", recorder)
        result = OpenRouterProvider(api_key="sk-or-x").check_key()
        assert result.ok is True
        assert recorder.sent[0]["url"].endswith("/key")
        assert recorder.sent[0]["headers"]["Authorization"] == "Bearer sk-or-x"


class TestCheckCaching:
    """Every check costs a request to the provider, and the picker asks on
    every opening."""

    def _registry(self, tmp_path, monkeypatch, results):
        from app.config import Settings
        from app.providers import KeyCheck, ProviderRegistry

        registry = ProviderRegistry(Settings(_env_file=None, state_dir=tmp_path))
        registry.set_key("runware", "rw-key")
        calls = {"n": 0}

        def _check(self):
            calls["n"] += 1
            return KeyCheck(*results)

        monkeypatch.setattr("app.providers.base.Provider.check_key", _check)
        return registry, calls

    def test_a_second_ask_is_answered_from_the_first(self, tmp_path, monkeypatch):
        registry, calls = self._registry(tmp_path, monkeypatch, (True, "fine"))
        assert registry.check_key("runware").ok is True
        registry.check_key("runware")
        assert calls["n"] == 1

    def test_force_spends_another_one(self, tmp_path, monkeypatch):
        registry, calls = self._registry(tmp_path, monkeypatch, (True, "fine"))
        registry.check_key("runware")
        registry.check_key("runware", force=True)
        assert calls["n"] == 2

    def test_a_new_key_is_never_answered_with_the_old_verdict(self, tmp_path, monkeypatch):
        registry, calls = self._registry(tmp_path, monkeypatch, (False, "rejected"))
        assert registry.check_key("runware").ok is False
        registry.set_key("runware", "a-better-key")
        registry.check_key("runware")
        assert calls["n"] == 2, "the verdict on the replaced key was reused"

    def test_clearing_the_key_forgets_the_verdict_too(self, tmp_path, monkeypatch):
        registry, calls = self._registry(tmp_path, monkeypatch, (True, "fine"))
        registry.check_key("runware")
        registry.clear_key("runware")
        registry.check_key("runware")
        assert calls["n"] == 2


class TestCreatorShapes:
    """Who made a model is answered in three different shapes across the two
    surfaces, and stringifying the wrong one puts a Python dict on screen."""

    def test_the_curated_slug_is_taken_as_written(self, monkeypatch):
        provider, _ = _curated(monkeypatch)
        assert provider.get_model("bfl:flux@2-dev").creator == "black-forest-labs"

    def test_a_whole_creator_record_is_reduced_to_its_name(self, monkeypatch):
        entry = dict(
            CURATED[0],
            creator={
                "id": "xai",
                "name": "xAI",
                "logo": "https://assets.runware.ai/x.png",
                "description": "a paragraph that does not belong on a list row",
            },
        )
        provider, _ = _curated(monkeypatch, [entry])
        model = provider.search_catalog(kind="all").models[0]
        assert model.creator == "xAI"
        assert "{" not in model.creator

    def test_no_creator_at_all_is_empty_rather_than_none(self, monkeypatch):
        provider, _ = _paid(monkeypatch, _search_payload(COMMUNITY_RESULTS[1:]))
        model = provider.search_catalog(kind="community").models[0]
        assert model.creator == ""


class TestTheRealPayloadShape:
    """Shaped from a live `modelSearch` answer, so the fixtures above cannot
    drift away from what Runware actually sends."""

    LIVE = {
        "name": "Absolute Realistic Vision - SD1.5 Merge",
        "air": "civitai:106886@114852",
        "tags": ["photorealistic", "base model", "photo", "realistic"],
        "heroImage": "https://mim.runware.ai/r/67c4a14056145-800x1067.jpg",
        "category": "checkpoint",
        "private": False,
        "comment": "",
        "version": "v1.0",
        "architecture": "sd1x",
        "capabilities": ["form:checkpoint", "io:text-to-image", "io:image-to-image"],
        "source": "community",
        "defaultWidth": 512,
        "defaultHeight": 512,
        "defaultSteps": 20,
        "defaultCFG": 7.5,
    }

    def test_it_is_read_as_a_generator_that_takes_a_reference(self, monkeypatch):
        provider, _ = _paid(monkeypatch, _search_payload([self.LIVE]))
        model = provider.search_catalog(kind="community").models[0]

        assert model.id == "civitai:106886@114852"
        assert model.makes_images is True
        assert model.reads_images is True
        # The architecture is the first thing an operator looks for, and the
        # community mirror's hero image is a cover like any other.
        assert model.description.startswith("sd1x · photorealistic")
        assert model.cover_image.startswith("https://mim.runware.ai/")
        assert model.creator == ""


class TestDiagnosingAFailure:
    """A real job failed with "OpenRouter refused the request: Provider
    returned error", which says nothing anyone can act on. These pin down the
    three reasons it said so little."""

    def _openrouter(self, monkeypatch, payload, status=200):
        from app.providers.openrouter import OpenRouterProvider

        recorder = _Recorder(payload, status)
        monkeypatch.setattr("httpx.post", recorder)
        return OpenRouterProvider(api_key="sk-or-x"), recorder

    def test_the_upstream_reason_is_carried_out_of_the_metadata(self, monkeypatch):
        """OpenRouter's own message is often just "Provider returned error";
        what actually happened is one level down."""
        from app.providers import ProviderError

        provider, _ = self._openrouter(
            monkeypatch,
            {
                "error": {
                    "code": 400,
                    "message": "Provider returned error",
                    "metadata": {
                        "provider_name": "Google AI Studio",
                        "raw": {"error": {"status": "INVALID_ARGUMENT", "message": "unsupported"}},
                    },
                }
            },
            status=400,
        )
        with pytest.raises(ProviderError) as caught:
            provider.generate(model="google/gemini-3.1-flash-image", prompt="x")

        said = str(caught.value)
        assert "Provider returned error" in said
        assert "Google AI Studio" in said
        assert "INVALID_ARGUMENT" in said

    def test_an_error_arriving_with_a_200_is_still_an_error(self, monkeypatch):
        """The HTTP call succeeded and the inference did not."""
        from app.providers import ProviderError

        provider, _ = self._openrouter(
            monkeypatch, {"error": {"message": "rate limited upstream"}}, status=200
        )
        with pytest.raises(ProviderError, match="rate limited upstream"):
            provider.generate(model="google/gemini-3.1-flash-image", prompt="x")

    def test_the_failing_call_is_named(self, monkeypatch):
        """One job can make two calls to OpenRouter — rewriting the prompt and
        then generating — so "the request" failed is not an answer."""
        from app.providers import ProviderError

        provider, _ = self._openrouter(monkeypatch, {"error": {"message": "no"}}, status=400)
        with pytest.raises(ProviderError, match="a generation"):
            provider.generate(model="m", prompt="x")
        with pytest.raises(ProviderError, match="a prompt rewrite"):
            provider.rewrite_prompt(model="m", prompt="x")

    def test_a_model_that_refuses_in_words_is_quoted(self, monkeypatch):
        """A model that will not draw something usually says why, in the text
        part of a perfectly successful reply."""
        from app.providers import ProviderError

        provider, _ = self._openrouter(
            monkeypatch,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "I can't create images of real football clubs.",
                            "images": [],
                        },
                    }
                ]
            },
        )
        with pytest.raises(ProviderError) as caught:
            provider.generate(model="google/gemini-3.1-flash-image", prompt="x")
        assert "real football clubs" in str(caught.value)

    def test_every_call_is_logged_with_the_model_and_the_outcome(self, monkeypatch, caplog):
        import logging

        provider, _ = self._openrouter(monkeypatch, {"error": {"message": "nope"}}, status=400)
        with caplog.at_level(logging.WARNING, logger="app.providers.openrouter"):
            with pytest.raises(Exception):
                provider.generate(model="google/gemini-3.1-flash-image", prompt="x")

        logged = caplog.text
        assert "google/gemini-3.1-flash-image" in logged
        assert "a generation" in logged
        assert "HTTP 400" in logged

    def test_a_runware_failure_logs_the_whole_error_object(self, monkeypatch, caplog):
        import logging

        provider, _ = _paid(
            monkeypatch,
            {
                "errors": [
                    {
                        "code": "invalidModel",
                        "message": "unknown model",
                        "parameter": "model",
                        "taskType": "imageInference",
                    }
                ]
            },
            status=400,
        )
        with caplog.at_level(logging.WARNING, logger="app.providers.runware"):
            with pytest.raises(Exception):
                provider.generate(model="nobody:nothing@0", prompt="x")

        # The sentence shown to the operator is short on purpose; the log is
        # what a bug report is written from.
        assert "invalidModel" in caplog.text
        assert "imageInference" in caplog.text


class TestRetrying:
    """A real job died on an upstream hiccup that cleared by itself: the same
    request succeeded a minute later. Retrying that is worth it — retrying a
    refusal only spends the failure twice."""

    def _flaky(self, failures, retryable=True):
        from app.providers import ProviderError

        calls = {"n": 0}

        def call():
            calls["n"] += 1
            if calls["n"] <= failures:
                raise ProviderError("Provider returned error", retryable=retryable)
            return "an image"

        return call, calls

    def test_a_transient_failure_is_tried_again(self):
        from app.providers import with_retries

        call, calls = self._flaky(2)
        assert with_retries(call, sleep=lambda _: None) == "an image"
        assert calls["n"] == 3

    def test_a_refusal_is_not(self):
        from app.providers import ProviderError, with_retries

        call, calls = self._flaky(2, retryable=False)
        with pytest.raises(ProviderError):
            with_retries(call, sleep=lambda _: None)
        # Once. A rejected key or a refused prompt fails the same way twice.
        assert calls["n"] == 1

    def test_it_gives_up_and_reports_the_last_failure(self):
        from app.providers import ProviderError, with_retries

        call, calls = self._flaky(99)
        with pytest.raises(ProviderError, match="Provider returned error"):
            with_retries(call, attempts=3, sleep=lambda _: None)
        assert calls["n"] == 3

    def test_the_wait_grows(self):
        from app.providers import with_retries

        waits = []
        call, _ = self._flaky(2)
        with_retries(call, delay=2.0, sleep=waits.append)
        assert waits == [2.0, 4.0]

    def test_a_rejected_key_is_never_retryable(self, monkeypatch):
        from app.providers.openrouter import OpenRouterProvider

        recorder = _Recorder({"error": {"message": "no"}}, 401)
        monkeypatch.setattr("httpx.post", recorder)
        try:
            OpenRouterProvider(api_key="sk-or-x").generate(model="m", prompt="x")
        except Exception as exc:
            assert getattr(exc, "retryable", False) is False

    def test_the_generic_upstream_wrapper_is(self, monkeypatch):
        """"Provider returned error" is what OpenRouter says when the model
        behind it hiccupped, which is exactly the case worth retrying."""
        from app.providers.openrouter import OpenRouterProvider

        recorder = _Recorder({"error": {"message": "Provider returned error"}}, 400)
        monkeypatch.setattr("httpx.post", recorder)
        try:
            OpenRouterProvider(api_key="sk-or-x").generate(model="m", prompt="x")
        except Exception as exc:
            assert getattr(exc, "retryable", False) is True
