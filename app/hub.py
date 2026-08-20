"""Searching the Hugging Face hub for something to load.

The service has always accepted a repo id it does not ship — `POST
/v1/models/load` takes one, guesses the architecture from the name and
downloads on demand. What was missing was a way to *find* one without leaving
the studio and pasting a string back in.

Nothing is downloaded here. A search answers what exists, what is already in
the local cache, and what this machine would make of it; loading is what pulls
the weights, and it already reports its own progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .models_registry import by_repo_id, spec_for_repo

logger = logging.getLogger(__name__)

# What a diffusion checkpoint looks like on the hub. Anything else in the
# search results is a text model, an adapter or a dataset card.
IMAGE_PIPELINES = frozenset(
    {"text-to-image", "image-to-image", "inpainting", "image-to-video", "text-to-video"}
)


class HubError(RuntimeError):
    """The hub could not be searched."""


@dataclass(frozen=True)
class HubModel:
    repo_id: str
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str | None = None
    gated: bool = False
    # Already in the local Hugging Face cache: loading it will not re-download.
    cached: bool = False
    # One of the checkpoints this service ships a tested profile for.
    in_catalog: bool = False
    catalog_id: str | None = None
    # What the loader would assume. For anything outside the catalog this is a
    # guess from the name, and the footprints behind it are estimates.
    family: str = "flux2"

    def as_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "downloads": self.downloads,
            "likes": self.likes,
            "pipeline_tag": self.pipeline_tag,
            "gated": self.gated,
            "cached": self.cached,
            "in_catalog": self.in_catalog,
            "catalog_id": self.catalog_id,
            "family": self.family,
        }


def cached_repos() -> set[str]:
    """Repo ids already on this disk, so a search can say what is free to load."""
    try:
        from huggingface_hub import scan_cache_dir

        return {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception as exc:  # noqa: BLE001 - an empty or missing cache is normal
        logger.debug("could not scan the huggingface cache: %s", exc)
        return set()


def search(query: str, limit: int = 30, *, only_images: bool = True) -> list[HubModel]:
    """Diffusion checkpoints on the hub matching `query`, most downloaded first."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - huggingface_hub is a dependency
        raise HubError("huggingface_hub is not installed") from exc

    try:
        found = list(
            HfApi().list_models(
                search=query.strip() or None,
                filter="diffusers",
                sort="downloads",
                limit=max(1, min(100, limit)) * (3 if only_images else 1),
            )
        )
    except Exception as exc:  # noqa: BLE001 - the hub is a network away
        raise HubError(f"could not search the Hugging Face hub: {exc}") from exc

    on_disk = cached_repos()
    models: list[HubModel] = []
    for entry in found:
        pipeline = getattr(entry, "pipeline_tag", None)
        if only_images and pipeline is not None and pipeline not in IMAGE_PIPELINES:
            continue
        known = by_repo_id(entry.id)
        models.append(
            HubModel(
                repo_id=entry.id,
                downloads=int(getattr(entry, "downloads", 0) or 0),
                likes=int(getattr(entry, "likes", 0) or 0),
                pipeline_tag=pipeline,
                gated=bool(getattr(entry, "gated", False)),
                cached=entry.id in on_disk,
                in_catalog=known is not None,
                catalog_id=known.id if known is not None else None,
                family=spec_for_repo(entry.id).family,
            )
        )
        if len(models) >= limit:
            break
    return models
