"""OpenRouter: one API in front of many vendors' models.

Two things make it a good fit here. Its catalog is public, so the Web models
tab can show what is on offer before a key exists; and every model declares its
input and output modalities, so image generators can be told apart from the
several hundred text models without guessing from names.
"""

from __future__ import annotations

import base64
import io
import logging
import re

import httpx
from PIL import Image

from .base import Provider, ProviderError, RemoteModel

logger = logging.getLogger(__name__)

_DATA_URI = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

SYSTEM_REWRITE = (
    "You rewrite image prompts. Return one vivid, concrete prompt describing "
    "subject, composition, lighting and medium. Keep the user's intent and "
    "every specific they named. Reply with the prompt only: no preamble, no "
    "quotes, no commentary."
)


def _to_data_uri(image: Image.Image, max_side: int = 1024) -> str:
    copy = image.convert("RGB")
    if max(copy.size) > max_side:
        copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode(url: str) -> Image.Image:
    raw = _DATA_URI.sub("", url)
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


class OpenRouterProvider(Provider):
    id = "openrouter"
    label = "OpenRouter"
    summary = (
        "One API in front of Google, OpenAI, Anthropic and others. Its catalog "
        "is public, so you can see what is on offer before adding a key."
    )
    docs_url = "https://openrouter.ai/docs"
    key_url = "https://openrouter.ai/keys"
    catalog_is_public = True

    def __init__(
        self,
        api_key: str | None,
        key_source: str = "none",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 180.0,
    ) -> None:
        super().__init__(api_key, key_source)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ---------------------------------------------------------------- catalog
    def list_models(self) -> list[RemoteModel]:
        try:
            response = httpx.get(f"{self._base_url}/models", timeout=30.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not reach OpenRouter: {exc}") from exc

        models = []
        for entry in payload.get("data", []):
            architecture = entry.get("architecture") or {}
            pricing = entry.get("pricing") or {}
            identifier = entry.get("id", "")
            models.append(
                RemoteModel(
                    id=identifier,
                    name=entry.get("name") or identifier,
                    description=(entry.get("description") or "").strip(),
                    input_modalities=tuple(architecture.get("input_modalities") or ()),
                    output_modalities=tuple(architecture.get("output_modalities") or ()),
                    context_length=entry.get("context_length"),
                    price_image=pricing.get("image"),
                    price_prompt=pricing.get("prompt"),
                    # `openrouter/auto` picks a model per request; it belongs in
                    # the list but is not a checkpoint you can pin meaningfully.
                    is_router=identifier.startswith("openrouter/"),
                )
            )
        models.sort(key=lambda model: model.id)
        return models

    # ------------------------------------------------------------- generation
    def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise ProviderError(
                "OpenRouter needs an API key. Add one on the Web models tab or set "
                "OIG_OPENROUTER_API_KEY."
            )
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/eduardodutragoncalves/openImageGen",
                    "X-Title": "openImageGen",
                },
                json=payload,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not reach OpenRouter: {exc}") from exc

        if response.status_code == 401:
            raise ProviderError("OpenRouter rejected the API key.")
        if response.status_code == 402:
            raise ProviderError("OpenRouter reports insufficient credit for that model.")
        if response.status_code >= 400:
            detail = ""
            try:
                detail = (response.json().get("error") or {}).get("message", "")
            except Exception:  # noqa: BLE001
                detail = response.text[:200]
            raise ProviderError(f"OpenRouter refused the request: {detail or response.status_code}")
        return response.json()

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
        content: list[dict] = [{"type": "text", "text": prompt}]
        for reference in references or []:
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(reference)}})

        images: list[Image.Image] = []
        # OpenRouter returns one image per completion, so several images means
        # several calls. Each is billed, which is why the count is capped
        # upstream by the same 1..4 the local engine uses.
        for _ in range(max(1, num_images)):
            data = self._post(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "modalities": ["image", "text"],
                }
            )
            images.extend(self._images_from(data))
        if not images:
            raise ProviderError(
                f"{model} returned no image. Not every model on OpenRouter can produce "
                "one, even when its catalog entry says so."
            )
        return images

    @staticmethod
    def _images_from(data: dict) -> list[Image.Image]:
        choices = data.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        out = []
        for item in message.get("images") or []:
            url = (item.get("image_url") or {}).get("url") or item.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                out.append(_decode(url))
        return out

    # ---------------------------------------------------------------- prompts
    def rewrite_prompt(
        self,
        *,
        model: str,
        prompt: str,
        references: list[Image.Image] | None = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for reference in references or []:
            content.append({"type": "image_url", "image_url": {"url": _to_data_uri(reference)}})

        data = self._post(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_REWRITE},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
            }
        )
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        text = text.strip()
        if not text:
            raise ProviderError(f"{model} returned an empty prompt")
        return text
