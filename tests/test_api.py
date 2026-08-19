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
