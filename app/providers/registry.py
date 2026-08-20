"""Which providers exist, their credentials, and the models pinned from them.

Credentials are held server-side and never sent back to the browser: the API
answers "configured: true" and nothing more. The store is a single JSON file
under the state directory, written 0600, so a key set through the UI survives a
restart without ever entering the repository.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from .base import KeyCheck, Provider, ProviderError, RemoteModel
from .openrouter import OpenRouterProvider
from .runware import RunwareProvider

logger = logging.getLogger(__name__)

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    OpenRouterProvider.id: OpenRouterProvider,
    RunwareProvider.id: RunwareProvider,
}

# Where each provider's credential and endpoint live in Settings, for the
# operators who would rather keep them in the environment than in the UI.
# How long a key check is trusted before it is spent again.
CHECK_TTL_S = 120.0

ENV_SETTINGS: dict[str, tuple[str, str]] = {
    OpenRouterProvider.id: ("openrouter_api_key", "openrouter_base_url"),
    RunwareProvider.id: ("runware_api_key", "runware_base_url"),
}


@dataclass(frozen=True)
class PinnedModel:
    """A remote model the operator chose to keep on the platform."""

    provider: str
    model_id: str
    label: str
    makes_images: bool = True
    reads_images: bool = False
    price_image: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_id}"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "provider": self.provider,
            "model_id": self.model_id,
            "label": self.label,
            "makes_images": self.makes_images,
            "reads_images": self.reads_images,
            "price_image": self.price_image,
        }


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._path = settings.state_dir / "providers.json"
        self._lock = threading.Lock()
        self._state = self._read()
        # A checked key stays checked for a couple of minutes. The picker asks
        # every time it opens, and each answer costs a request to the provider.
        self._checks: dict[str, tuple[float, KeyCheck]] = {}

    # ------------------------------------------------------------- persistence
    def _read(self) -> dict:
        if not self._path.is_file():
            return {"keys": {}, "pinned": []}
        try:
            with open(self._path) as handle:
                data = json.load(handle)
        except Exception:  # noqa: BLE001 - a corrupt file must not stop startup
            logger.exception("could not read %s; starting from empty", self._path)
            return {"keys": {}, "pinned": []}
        data.setdefault("keys", {})
        data.setdefault("pinned", [])
        return data

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with open(temporary, "w") as handle:
            json.dump(self._state, handle, indent=2)
        os.chmod(temporary, 0o600)
        temporary.replace(self._path)

    # ------------------------------------------------------------- credentials
    def _key_for(self, provider_id: str) -> tuple[str | None, str]:
        """(key, where it came from). A key set here wins over the environment."""
        stored = (self._state.get("keys") or {}).get(provider_id)
        if stored:
            return stored, "stored"
        key_attr, _ = ENV_SETTINGS.get(provider_id, ("", ""))
        from_env = getattr(self.settings, key_attr, None) if key_attr else None
        if from_env:
            return from_env, "env"
        return None, "none"

    def set_key(self, provider_id: str, key: str) -> None:
        if provider_id not in PROVIDER_CLASSES:
            raise ProviderError(f"unknown provider {provider_id!r}")
        with self._lock:
            self._state.setdefault("keys", {})[provider_id] = key.strip()
            self._checks.pop(provider_id, None)
            self._write()
        logger.info("stored an API key for %s", provider_id)

    def clear_key(self, provider_id: str) -> None:
        with self._lock:
            (self._state.get("keys") or {}).pop(provider_id, None)
            self._checks.pop(provider_id, None)
            self._write()
        logger.info("cleared the stored API key for %s", provider_id)

    # ---------------------------------------------------------------- providers
    def get(self, provider_id: str) -> Provider:
        cls = PROVIDER_CLASSES.get(provider_id)
        if cls is None:
            raise ProviderError(f"unknown provider {provider_id!r}")
        key, source = self._key_for(provider_id)
        _, base_attr = ENV_SETTINGS.get(provider_id, ("", ""))
        base_url = getattr(self.settings, base_attr, None) if base_attr else None
        if base_url:
            return cls(key, key_source=source, base_url=base_url)
        return cls(key, key_source=source)

    def check_key(self, provider_id: str, force: bool = False) -> KeyCheck:
        """Whether the credential actually works, remembered briefly."""
        cached = self._checks.get(provider_id)
        if cached and not force and time.monotonic() - cached[0] < CHECK_TTL_S:
            return cached[1]
        result = self.get(provider_id).check_key()
        with self._lock:
            self._checks[provider_id] = (time.monotonic(), result)
        return result

    def list_providers(self) -> list[dict]:
        return [self.get(provider_id).info().as_dict() for provider_id in PROVIDER_CLASSES]

    # ------------------------------------------------------------------ pinned
    def pinned(self, provider_id: str | None = None) -> list[PinnedModel]:
        entries = [PinnedModel(**entry) for entry in self._state.get("pinned", [])]
        if provider_id:
            entries = [entry for entry in entries if entry.provider == provider_id]
        return entries

    def find_pinned(self, key: str) -> PinnedModel | None:
        return next((entry for entry in self.pinned() if entry.key == key), None)

    def pin(self, provider_id: str, model: RemoteModel) -> PinnedModel:
        entry = PinnedModel(
            provider=provider_id,
            model_id=model.id,
            label=model.name,
            makes_images=model.makes_images,
            reads_images=model.reads_images,
            price_image=model.price_image,
        )
        with self._lock:
            pinned = [p for p in self._state.get("pinned", []) if
                      f"{p['provider']}:{p['model_id']}" != entry.key]
            pinned.append(
                {
                    "provider": entry.provider,
                    "model_id": entry.model_id,
                    "label": entry.label,
                    "makes_images": entry.makes_images,
                    "reads_images": entry.reads_images,
                    "price_image": entry.price_image,
                }
            )
            self._state["pinned"] = pinned
            self._write()
        return entry

    def unpin(self, key: str) -> bool:
        with self._lock:
            before = len(self._state.get("pinned", []))
            self._state["pinned"] = [
                p
                for p in self._state.get("pinned", [])
                if f"{p['provider']}:{p['model_id']}" != key
            ]
            changed = len(self._state["pinned"]) != before
            if changed:
                self._write()
        return changed
