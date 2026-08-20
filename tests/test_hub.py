"""Finding something to load on the Hugging Face hub.

The service has always accepted a repo id it does not ship; what was missing
was a way to find one without leaving the studio. Nothing here downloads
anything — a search answers what exists, what is already on this disk, and what
this machine would assume about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Entry:
    """Shaped like huggingface_hub's ModelInfo, with only what is read."""

    def __init__(self, model_id, downloads=0, likes=0, pipeline_tag="text-to-image", gated=False):
        self.id = model_id
        self.downloads = downloads
        self.likes = likes
        self.pipeline_tag = pipeline_tag
        self.gated = gated


RESULTS = [
    _Entry("black-forest-labs/FLUX.1-schnell", downloads=429787, likes=5578),
    _Entry("black-forest-labs/FLUX.2-dev", downloads=825840, pipeline_tag="image-to-image"),
    _Entry("someone/a-sentence-embedder", downloads=99, pipeline_tag="feature-extraction"),
    _Entry("someone/an-unlabelled-checkpoint", downloads=12, pipeline_tag=None),
]


@pytest.fixture()
def hub(monkeypatch):
    """The hub answering with RESULTS, and one repo already on disk."""
    sent = {}

    class _Api:
        def list_models(self, **kwargs):
            sent.update(kwargs)
            return list(RESULTS)

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        "app.hub.cached_repos", lambda: {"black-forest-labs/FLUX.1-schnell"}
    )
    return sent


class TestSearch:
    def test_it_asks_for_diffusers_sorted_by_downloads(self, hub):
        from app.hub import search

        search("flux", limit=5)
        assert hub["search"] == "flux"
        assert hub["filter"] == "diffusers"
        assert hub["sort"] == "downloads"

    def test_models_that_do_not_make_pictures_are_dropped(self, hub):
        from app.hub import search

        found = {model.repo_id for model in search("flux")}
        assert "someone/a-sentence-embedder" not in found
        # An unlabelled entry is kept: the hub's tags are optional, and a
        # missing one is not evidence against the model.
        assert "someone/an-unlabelled-checkpoint" in found

    def test_the_filter_can_be_dropped(self, hub):
        from app.hub import search

        found = {model.repo_id for model in search("flux", only_images=False)}
        assert "someone/a-sentence-embedder" in found

    def test_what_is_already_on_disk_is_marked(self, hub):
        from app.hub import search

        models = {model.repo_id: model for model in search("flux")}
        assert models["black-forest-labs/FLUX.1-schnell"].cached is True
        assert models["black-forest-labs/FLUX.2-dev"].cached is False

    def test_a_catalogued_model_is_recognised_as_one(self, hub):
        from app.hub import search

        models = {model.repo_id: model for model in search("flux")}
        schnell = models["black-forest-labs/FLUX.1-schnell"]
        assert schnell.in_catalog is True
        assert schnell.catalog_id == "flux1-schnell"
        assert schnell.family == "flux1"

        # Outside the catalog the architecture is guessed from the name, and
        # the UI says the footprints behind it are estimates.
        stranger = models["someone/an-unlabelled-checkpoint"]
        assert stranger.in_catalog is False
        assert stranger.catalog_id is None

    def test_the_limit_is_honoured_after_filtering(self, hub):
        from app.hub import search

        assert len(search("flux", limit=2)) == 2

    def test_an_unreachable_hub_is_an_error_with_a_reason(self, monkeypatch):
        from app.hub import HubError, search

        class _Api:
            def list_models(self, **kwargs):
                raise OSError("no route to host")

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)
        with pytest.raises(HubError, match="no route to host"):
            search("flux")

    def test_a_missing_cache_is_not_a_failure(self, monkeypatch):
        """A machine that has never downloaded anything has no cache to scan."""
        from app import hub

        def _explode():
            raise OSError("no cache directory")

        monkeypatch.setattr("huggingface_hub.scan_cache_dir", _explode)
        assert hub.cached_repos() == set()


class TestTheRoute:
    def test_an_unreachable_hub_answers_502(self, client, monkeypatch):
        from app.hub import HubError

        def _fail(*args, **kwargs):
            raise HubError("could not search the Hugging Face hub: timed out")

        monkeypatch.setattr("app.main.search_hub", _fail)
        response = client.get("/v1/models/search", params={"q": "flux"})
        assert response.status_code == 502
        assert "Hugging Face" in response.json()["detail"]

    def test_results_carry_what_the_picker_shows(self, client, monkeypatch):
        from app.hub import HubModel

        monkeypatch.setattr(
            "app.main.search_hub",
            lambda q, limit, only_images=True: [
                HubModel(
                    repo_id="black-forest-labs/FLUX.1-dev",
                    downloads=606804,
                    likes=14193,
                    pipeline_tag="text-to-image",
                    cached=False,
                    in_catalog=True,
                    catalog_id="flux1-dev",
                    family="flux1",
                )
            ],
        )
        entry = client.get("/v1/models/search", params={"q": "flux"}).json()[0]
        assert entry["repo_id"] == "black-forest-labs/FLUX.1-dev"
        assert entry["in_catalog"] is True
        assert entry["cached"] is False
        assert entry["family"] == "flux1"

    def test_the_query_is_passed_through(self, client, monkeypatch):
        seen = {}

        def _search(q, limit, only_images=True):
            seen.update({"q": q, "limit": limit, "only_images": only_images})
            return []

        monkeypatch.setattr("app.main.search_hub", _search)
        client.get("/v1/models/search", params={"q": "qwen", "limit": 7, "only_images": False})
        assert seen == {"q": "qwen", "limit": 7, "only_images": False}
