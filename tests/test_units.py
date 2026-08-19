"""Units where the arithmetic, not the wiring, is what can be wrong."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.images import fit_to_budget
from app.models_registry import CATALOG, by_id, spec_for_repo
from app.store import JobRecord, JobStore, enforce_retention


class TestPixelBudget:
    def test_a_request_inside_the_budget_is_left_alone(self):
        assert fit_to_budget(1024, 1024, 1024 * 1024) == (1024, 1024)

    def test_an_oversized_request_is_scaled_and_stays_aligned(self):
        width, height = fit_to_budget(2048, 2048, 1024 * 1024)
        assert width * height <= 1024 * 1024
        assert width % 16 == 0 and height % 16 == 0

    def test_aspect_ratio_survives_the_scaling(self):
        width, height = fit_to_budget(1920, 1080, 1024 * 1024)
        assert abs((width / height) - (1920 / 1080)) < 0.05


class TestRegistry:
    def test_every_entry_has_a_licence_and_a_footprint(self):
        for spec in CATALOG:
            assert spec.licence
            assert spec.transformer_vram_gb > 0
            assert spec.text_encoder_vram_gb > 0

    def test_ids_are_unique(self):
        ids = [spec.id for spec in CATALOG]
        assert len(ids) == len(set(ids))

    def test_flux1_can_be_quantized_and_flux2_mirror_cannot(self):
        assert by_id("flux1-dev").can_quantize is True
        assert by_id("flux2-dev-4bit").can_quantize is False

    def test_quantized_footprints_are_smaller(self):
        spec = by_id("flux1-dev")
        assert sum(spec.footprints("nf4")) < sum(spec.footprints("bf16"))

    def test_schnell_ignores_guidance(self):
        spec = by_id("flux1-schnell")
        assert spec.guidance_range == (0.0, 0.0)

    def test_an_unknown_repo_is_described_rather_than_refused(self):
        spec = spec_for_repo("someone/FLUX.1-experimental")
        assert spec.custom is True
        assert spec.family == "flux1"


class TestStore:
    def _record(self, **overrides):
        base = dict(id="a", owner="me", kind="generation", status="succeeded", created=1)
        base.update(overrides)
        return JobRecord(**base)

    def test_a_restart_closes_jobs_that_can_never_finish(self, tmp_path):
        store = JobStore(tmp_path / "jobs.db")
        store.upsert(self._record(id="running", status="running"))
        store.upsert(self._record(id="done", status="succeeded"))
        assert store.mark_interrupted() == 1
        assert store.get("running").status == "failed"
        assert store.get("done").status == "succeeded"

    def test_history_is_scoped_by_owner(self, tmp_path):
        store = JobStore(tmp_path / "jobs.db")
        store.upsert(self._record(id="mine", owner="me"))
        store.upsert(self._record(id="theirs", owner="them"))
        assert store.count(owner="me") == 1
        assert [r.id for r in store.list(owner="me")] == ["mine"]


class TestRetention:
    def test_the_oldest_files_go_first_when_over_budget(self, tmp_path):
        import os
        import time

        for index in range(5):
            path = tmp_path / f"{index}.png"
            path.write_bytes(b"x" * 1024 * 1024)
            os.utime(path, (time.time() - (10 - index) * 60,) * 2)

        removed = enforce_retention(tmp_path, max_gb=3 / 1024, max_age_days=None)
        assert removed == ["0.png", "1.png"]
        assert not (tmp_path / "0.png").exists()
        assert (tmp_path / "4.png").exists()

    def test_no_budget_means_nothing_is_touched(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        assert enforce_retention(tmp_path, max_gb=None, max_age_days=None) == []
        assert (tmp_path / "a.png").exists()
