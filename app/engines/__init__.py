"""Model backends, one per architecture family.

``create_engine`` is the only thing the rest of the service needs: hand it a
spec from the registry and it returns something that loads, describes itself
and generates. Adding a family is a new module plus one line here.
"""

from __future__ import annotations

from ..config import Settings
from ..models_registry import ModelSpec
from .base import BaseEngine, EngineResult, ProgressCallback


def create_engine(settings: Settings, spec: ModelSpec, choice=None) -> BaseEngine:
    if spec.family == "flux2":
        from .flux2 import Flux2Engine

        return Flux2Engine(settings, spec, choice=choice)
    if spec.family == "flux1":
        from .flux1 import Flux1Engine

        return Flux1Engine(settings, spec, choice=choice)
    raise ValueError(
        f"no backend for model family {spec.family!r}; supported families are flux2 and flux1"
    )


__all__ = ["BaseEngine", "EngineResult", "ProgressCallback", "create_engine"]
