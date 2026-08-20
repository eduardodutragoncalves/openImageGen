"""Remote model providers.

A provider is a catalog of models this server does not host: it can be
searched, models can be pinned for later use, and — where the provider
supports it — a pinned model can generate an image through its API instead of
through the local GPUs.

The shape here is deliberately small. A provider answers three questions:
what models do you have, which of them make images, and can you make one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class RemoteModel:
    """One model in a provider's catalog."""

    id: str
    name: str
    description: str = ""
    # What the model takes and returns, which is how image generators are told
    # apart from everything else.
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    context_length: int | None = None
    # Provider-quoted prices, as strings because they are per-unit and vary in
    # unit between providers. Displayed, never used in arithmetic.
    price_image: str | None = None
    price_prompt: str | None = None
    # A router entry rather than a model: useful, but not a checkpoint.
    is_router: bool = False

    @property
    def makes_images(self) -> bool:
        return "image" in self.output_modalities

    @property
    def reads_images(self) -> bool:
        return "image" in self.input_modalities

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "input_modalities": list(self.input_modalities),
            "output_modalities": list(self.output_modalities),
            "context_length": self.context_length,
            "price_image": self.price_image,
            "price_prompt": self.price_prompt,
            "is_router": self.is_router,
            "makes_images": self.makes_images,
            "reads_images": self.reads_images,
        }


class ProviderError(RuntimeError):
    """Anything the provider refused or could not answer."""


@dataclass
class ProviderInfo:
    id: str
    label: str
    summary: str
    docs_url: str = ""
    key_url: str = ""
    supports_generation: bool = True
    configured: bool = False
    # Where the credential came from, so the UI can say whether editing it
    # here will have any effect.
    key_source: str = "none"  # none | env | stored
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "summary": self.summary,
            "docs_url": self.docs_url,
            "key_url": self.key_url,
            "supports_generation": self.supports_generation,
            "configured": self.configured,
            "key_source": self.key_source,
            **self.extra,
        }


class Provider(ABC):
    """One remote catalog."""

    id: str
    label: str
    summary: str
    docs_url: str = ""
    key_url: str = ""
    supports_generation: bool = True
    # True when the catalog can be listed without a credential, which lets the
    # UI show what is on offer before anyone has pasted a key.
    catalog_is_public: bool = False

    def __init__(self, api_key: str | None, key_source: str = "none") -> None:
        self.api_key = api_key
        self.key_source = key_source

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            label=self.label,
            summary=self.summary,
            docs_url=self.docs_url,
            key_url=self.key_url,
            supports_generation=self.supports_generation,
            configured=self.configured,
            key_source=self.key_source,
            extra={"catalog_is_public": self.catalog_is_public},
        )

    @abstractmethod
    def list_models(self) -> list[RemoteModel]:
        """Every model the provider offers."""

    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        references: list[Image.Image] | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images: int = 1,
    ) -> list[Image.Image]:
        """Produce images through the provider's API."""

    def rewrite_prompt(
        self,
        *,
        model: str,
        prompt: str,
        references: list[Image.Image] | None = None,
    ) -> str:
        """Improve a prompt using one of the provider's language models."""
        raise ProviderError(f"{self.label} cannot rewrite prompts")


def search(models: list[RemoteModel], query: str) -> list[RemoteModel]:
    """Substring match over id, name and description."""
    needle = query.strip().lower()
    if not needle:
        return models
    return [
        model
        for model in models
        if needle in model.id.lower()
        or needle in model.name.lower()
        or needle in model.description.lower()
    ]
