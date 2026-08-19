"""Test fixtures.

Everything runs against the real application in dry-run mode: the HTTP layer,
the queue, the archive and the placement planner are all exercised; only the
weights are absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OIG_DRY_RUN", "true")
    monkeypatch.setenv("OIG_DRY_RUN_STEP_SECONDS", "0")
    monkeypatch.setenv("OIG_HOST", "127.0.0.1")
    monkeypatch.setenv("OIG_API_KEYS", "alpha-key,beta-key")
    monkeypatch.setenv("OIG_ENABLE_NSFW_FILTER", "false")
    monkeypatch.setenv("OIG_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("OIG_STATE_DIR", str(tmp_path / "state"))

    from app.config import get_settings

    get_settings.cache_clear()
    for module in [m for m in list(sys.modules) if m.startswith("app.")]:
        del sys.modules[module]

    from app.main import app

    with TestClient(app) as test_client:
        test_client.post("/v1/auth", json={"key": "alpha-key"})
        yield test_client

    get_settings.cache_clear()


@pytest.fixture()
def settings_factory(monkeypatch):
    """Build a Settings object with overrides, ignoring the developer's .env."""

    def build(**overrides):
        for key in list(os.environ):
            if key.startswith("OIG_"):
                monkeypatch.delenv(key, raising=False)
        from app.config import Settings

        return Settings(_env_file=None, **overrides)

    return build
