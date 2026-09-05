"""The HTTP contract the studio depends on."""

from __future__ import annotations

import pytest


def wait_ready(client):
    for _ in range(80):
        body = client.get("/healthz").json()
        if body["status"] == "ok":
            return body
        import time

        time.sleep(0.05)
    raise AssertionError(f"never became ready: {body}")


class TestAuth:
    def test_unauthenticated_requests_are_refused(self, client):
        client.cookies.clear()
        assert client.get("/v1/jobs").status_code == 401
        assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 401

    def test_a_bad_key_says_so(self, client):
        client.cookies.clear()
        response = client.post("/v1/auth", json={"key": "nope"})
        assert response.status_code == 401
        assert "not recognised" in response.json()["detail"]

    def test_the_cookie_never_carries_the_key_itself(self, client):
        client.cookies.clear()
        client.post("/v1/auth", json={"key": "alpha-key"})
        assert "alpha-key" not in str(dict(client.cookies))

    def test_files_are_reachable_with_the_cookie_alone(self, client):
        """<img src> cannot send a header, so the cookie has to be enough."""
        wait_ready(client)
        job = submit(client, "a fox", response_format="url")
        url = job["result"]["images"][0]["url"]
        assert client.get(url).status_code == 200


class TestHealth:
    def test_health_reports_the_machine(self, client):
        body = wait_ready(client)
        assert body["model"]["state"] == "ready"
        assert body["auth_required"] is True
        assert "used_bytes" in body["storage"]
        assert isinstance(body["gpus"], list)


class TestModels:
    def test_catalog_lists_both_families(self, client):
        catalog = client.get("/v1/models/catalog").json()
        families = {entry["family"] for entry in catalog}
        assert families == {"flux1", "flux2"}

    def test_steps_come_from_the_checkpoint_not_the_environment(self, client):
        """OIG_DEFAULT_STEPS=50 must not make schnell, a 4-step model, do 50."""
        catalog = {entry["id"]: entry for entry in client.get("/v1/models/catalog").json()}
        assert catalog["flux1-schnell"]["default_steps"] == 4
        assert catalog["flux2-dev-4bit"]["default_steps"] == 50

    def test_a_model_too_large_is_listed_with_the_reason(self, client):
        catalog = {entry["id"]: entry for entry in client.get("/v1/models/catalog").json()}
        oversized = catalog["flux2-dev"]
        assert oversized["runnable"] is False
        assert "transformer alone needs" in oversized["placement_reason"]

    def test_an_unknown_model_is_a_404(self, client):
        assert client.post("/v1/models/load", json={"model": "nope"}).status_code == 404


def submit(client, prompt, **extra):
    payload = {"prompt": prompt, "width": 512, "height": 512, "num_steps": 2, **extra}
    response = client.post("/v1/images/generations", json=payload)
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    body = client.get(f"/v1/jobs/{job_id}", params={"wait": 30}).json()
    assert body["status"] == "succeeded", body
    return body


class TestGeneration:
    def test_a_job_runs_and_lands_in_the_archive(self, client):
        wait_ready(client)
        submit(client, "a red fox in deep snow", response_format="url")
        page = client.get("/v1/jobs").json()
        assert page["total"] == 1
        row = page["jobs"][0]
        assert row["prompt"] == "a red fox in deep snow"
        assert row["width"] == 512 and row["height"] == 512
        assert row["images"] and row["images"][0]["available"] is True

    def test_openrouter_upsampling_without_a_key_is_refused_up_front(self, client):
        wait_ready(client)
        response = client.post(
            "/v1/images/generations", json={"prompt": "x", "upsample_prompt": "openrouter"}
        )
        assert response.status_code == 400

    def test_the_archive_searches(self, client):
        wait_ready(client)
        submit(client, "an overgrown greenhouse", response_format="url")
        assert client.get("/v1/jobs", params={"search": "greenhouse"}).json()["total"] == 1
        assert client.get("/v1/jobs", params={"search": "submarine"}).json()["total"] == 0


class TestImageProvenance:
    """An image outlives the archive row that describes it, so what produced
    it — and what it billed — is written into the file itself."""

    def _metadata(self, client, url, tmp_path):
        from app.images import read_metadata

        response = client.get(url)
        assert response.status_code == 200
        path = tmp_path / "downloaded.png"
        path.write_bytes(response.content)
        return read_metadata(path)

    def test_a_saved_file_names_the_model_that_made_it(self, client, tmp_path):
        wait_ready(client)
        job = submit(client, "a red fox", response_format="url")
        row = client.get("/v1/jobs").json()["jobs"][0]

        written = self._metadata(client, job["result"]["images"][0]["url"], tmp_path)
        assert written["Model"] == row["model_label"]
        assert written["Model ID"] == row["model_id"]
        assert written["Software"] == "openImageGen"

    def test_a_local_generation_claims_no_cost(self, client, tmp_path):
        """The GPUs bill nothing, and "0.00" would be a claim, not a blank."""
        wait_ready(client)
        job = submit(client, "a red fox", response_format="url")
        written = self._metadata(client, job["result"]["images"][0]["url"], tmp_path)
        assert "Cost" not in written

    def test_an_inline_answer_carries_the_same_metadata(self, client, tmp_path):
        import base64

        from app.images import read_metadata

        wait_ready(client)
        job = submit(client, "a red fox", response_format="b64_json")
        path = tmp_path / "inline.png"
        path.write_bytes(base64.b64decode(job["result"]["images"][0]["b64_json"]))
        assert read_metadata(path)["Model"]


class TestClearingAGpu:
    """Pinned against patched devices, never the machine's own.

    The suite has to pass identically on a box with two cards and on one with
    none, and a test that reaches for a real GPU would sweep the allocator of
    whatever else is running on it.
    """

    def test_it_needs_a_key_like_everything_else(self, client):
        client.cookies.clear()
        assert client.post("/v1/gpus/0/release").status_code == 401

    def test_a_card_this_process_cannot_see_is_a_404(self, client, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        response = client.post("/v1/gpus/0/release")
        assert response.status_code == 404
        assert "CUDA" in response.json()["detail"]

    def test_an_index_past_the_last_card_is_a_404(self, client, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        response = client.post("/v1/gpus/7/release")
        assert response.status_code == 404
        assert "no cuda:7" in response.json()["detail"]

    def _fake_cards(self, monkeypatch, free=(8000, 24000)):
        """Present two cards without touching the machine's own.

        device_count is left alone deliberately: /healthz enumerates every
        card it claims exist, so inventing extras makes mem_get_info fail on
        a real GPU. Only the memory calls are faked, which is enough to keep
        the sweep away from cards this process does not own.
        """
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr("app.model_manager.release_device_memory", lambda index: None)
        monkeypatch.setattr("app.model_manager.release_cuda_memory", lambda: None)
        monkeypatch.setattr("app.model_manager.device_memory_mb", lambda index: free)

    def test_clearing_the_card_the_model_is_on_unloads_it_and_says_so(
        self, client, monkeypatch
    ):
        wait_ready(client)
        self._fake_cards(monkeypatch)

        body = client.post("/v1/gpus/0/release").json()
        assert body["unloaded_model"]
        assert "Unloaded" in body["detail"] and "Load a model" in body["detail"]

        # The rail must stop claiming a model is ready the moment it is not.
        status = client.get("/v1/models/status").json()
        assert status["state"] == "empty"
        assert status["model_id"] is None

    def test_a_job_submitted_after_a_clear_fails_with_the_reason(self, client, monkeypatch):
        """Rather than hanging on a warm-up for a load nobody asked for."""
        wait_ready(client)
        self._fake_cards(monkeypatch)
        client.post("/v1/gpus/0/release")

        response = client.post(
            "/v1/images/generations", json={"prompt": "x", "width": 512, "height": 512}
        )
        assert response.status_code == 202
        body = client.get(f"/v1/jobs/{response.json()['id']}", params={"wait": 30}).json()
        assert body["status"] == "failed"
        # The reason names the clear, not a generic absence: "no model is
        # loaded" would leave the operator wondering what happened to theirs.
        assert "gpu was cleared" in (body["error"] or "").lower()
        assert "load a model" in (body["error"] or "").lower()


class TestCostReachesTheOperator:
    """The price a provider quoted has two destinations, and both matter.

    It is written into the image file, which outlives the archive row, and it
    is carried on the job so the panel beside the picture can show it. A price
    that only reaches the file is a price nobody reads.
    """

    def test_a_quoted_price_rides_the_job_into_the_archive(self, client, monkeypatch):
        from PIL import Image

        from app.providers import tag_cost

        wait_ready(client)
        # A local dry-run generation, with a provider's price pinned onto the
        # image the way Runware's adapter does.
        monkeypatch.setattr(
            "app.engines.base.BaseEngine.generate",
            lambda self, **kwargs: _priced_result(kwargs, tag_cost, Image),
        )
        job = submit(client, "a billed heron", response_format="url")
        image = job["result"]["images"][0]
        assert image["cost"] == pytest.approx(0.0021)

        # And it survives into the archive row, which is what the panel reads
        # once the job has left the queue.
        row = client.get("/v1/jobs").json()["jobs"][0]
        assert row["images"][0]["cost"] == pytest.approx(0.0021)

    def test_a_local_generation_quotes_no_price(self, client):
        wait_ready(client)
        job = submit(client, "an unbilled heron", response_format="url")
        assert job["result"]["images"][0]["cost"] is None


def _priced_result(kwargs, tag_cost, Image):
    """One image carrying a provider's price, shaped like an EngineResult."""
    from app.engines.base import EngineResult

    width = kwargs.get("width") or 512
    height = kwargs.get("height") or 512
    image = tag_cost(Image.new("RGB", (width, height), "teal"), 0.0021)
    return EngineResult(
        images=[image],
        seeds=[7],
        width=width,
        height=height,
        timings={"denoise_s": 0.0},
    )


class TestOwnership:
    def test_one_key_cannot_see_another_keys_work(self, client):
        wait_ready(client)
        job = submit(client, "private work", response_format="url")

        client.cookies.clear()
        client.post("/v1/auth", json={"key": "beta-key"})
        assert client.get("/v1/jobs").json()["total"] == 0
        assert client.get(f"/v1/jobs/{job['id']}").status_code == 404


class TestExposure:
    def test_a_public_bind_without_keys_refuses_to_start(self, settings_factory):
        from app.main import _check_exposure

        with pytest.raises(RuntimeError, match="OIG_API_KEYS"):
            _check_exposure(settings_factory(host="0.0.0.0", api_keys=[]))

    def test_loopback_without_keys_is_fine(self, settings_factory):
        from app.main import _check_exposure

        _check_exposure(settings_factory(host="127.0.0.1", api_keys=[]))

    def test_an_explicit_override_is_honoured(self, settings_factory):
        from app.main import _check_exposure

        _check_exposure(
            settings_factory(host="0.0.0.0", api_keys=[], allow_open_access=True)
        )

    def test_comma_separated_keys_parse(self, settings_factory):
        assert settings_factory(api_keys="a,b , c").api_keys == ["a", "b", "c"]


class TestSpa:
    def test_an_unknown_api_path_is_not_swallowed_by_the_spa(self, client):
        assert client.get("/v1/nonexistent").status_code == 404
