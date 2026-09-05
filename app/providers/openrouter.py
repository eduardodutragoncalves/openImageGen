"""OpenRouter: one API in front of many vendors' models.

Two things make it a good fit here. Its catalog is public, so the Web models
tab can show what is on offer before a key exists; and every model declares its
input and output modalities, so image generators can be told apart from the
several hundred text models without guessing from names.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time

import httpx
from PIL import Image

from .base import Provider, ProviderError, RemoteModel, tag_cost

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

    def _check_key(self) -> None:
        """`/key` describes the credential itself, and costs no tokens."""
        try:
            response = httpx.get(
                f"{self._base_url}/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=20.0,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not reach OpenRouter: {exc}") from exc
        if response.status_code in (401, 403):
            raise ProviderError("OpenRouter rejected the key")
        if response.status_code >= 400:
            logger.warning("openrouter: the key check answered HTTP %s", response.status_code)
            raise ProviderError(f"OpenRouter answered {response.status_code}")

    # ------------------------------------------------------------- generation
    def _post(self, payload: dict, what: str = "a request") -> dict:
        """One call to OpenRouter, logged either way.

        `what` names the operation because a single job can make two of these —
        rewriting the prompt and then generating — and "OpenRouter refused the
        request" is not an answer when you cannot tell which request.
        """
        if not self.api_key:
            raise ProviderError(
                "OpenRouter needs an API key. Add one on the Web models tab or set "
                "OIG_OPENROUTER_API_KEY."
            )
        model = payload.get("model", "?")
        started = time.perf_counter()
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
            logger.warning("openrouter: %s with %s could not be sent: %s", what, model, exc)
            raise ProviderError(f"could not reach OpenRouter: {exc}", retryable=True) from exc

        elapsed = time.perf_counter() - started
        if response.status_code == 429:
            logger.warning("openrouter: %s with %s was rate limited", what, model)
            raise ProviderError("OpenRouter is rate limiting this key.", retryable=True)
        if response.status_code == 401:
            logger.warning("openrouter: %s with %s rejected the key", what, model)
            raise ProviderError("OpenRouter rejected the API key.")
        if response.status_code == 402:
            logger.warning("openrouter: %s with %s has no credit", what, model)
            raise ProviderError("OpenRouter reports insufficient credit for that model.")

        try:
            data = response.json()
        except ValueError:
            logger.warning(
                "openrouter: %s with %s answered HTTP %s with non-JSON: %s",
                what, model, response.status_code, response.text[:400],
            )
            raise ProviderError(
                f"OpenRouter answered {response.status_code} with something that was not JSON"
            ) from None

        # An error can arrive with a 200: the HTTP call succeeded and the
        # inference did not.
        if response.status_code >= 400 or data.get("error"):
            detail = self._describe(data, response.status_code)
            logger.warning(
                "openrouter: %s with %s failed after %.1fs — HTTP %s: %s",
                what, model, elapsed, response.status_code, detail,
            )
            raise ProviderError(
                f"{model} refused {what}: {detail}",
                # A 5xx, or the generic wrapper OpenRouter puts around an
                # upstream hiccup, is worth one more attempt. A 400 is not.
                retryable=response.status_code >= 500
                or "provider returned error" in detail.lower(),
            )

        logger.info("openrouter: %s with %s in %.1fs", what, model, elapsed)
        return data

    @staticmethod
    def _describe(data: dict, status: int) -> str:
        """The reason, dug out of where OpenRouter puts it.

        Its own `message` is frequently just "Provider returned error"; what
        actually happened — a content policy refusal, an unsupported parameter,
        the upstream quota — is in `error.metadata`, and dropping that leaves
        the operator with nothing to act on.
        """
        error = data.get("error") or {}
        parts: list[str] = []
        message = str(error.get("message") or "").strip()
        if message:
            parts.append(message)

        metadata = error.get("metadata") or {}
        upstream = metadata.get("provider_name")
        raw = metadata.get("raw")
        if raw is not None and not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        raw = (raw or "").strip()
        if raw:
            parts.append(f"{upstream or 'the provider'} said: {raw[:600]}")
        elif upstream:
            parts.append(f"upstream: {upstream}")

        if reasons := metadata.get("reasons"):
            parts.append(f"reasons: {reasons}")
        return " — ".join(parts) or f"HTTP {status}"

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
        said: str = ""
        for _ in range(max(1, num_images)):
            data = self._post(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "modalities": ["image", "text"],
                },
                what="a generation",
            )
            produced = self._images_from(data)
            # OpenRouter bills the whole completion, not each image in it, so a
            # reply carrying more than one splits the charge evenly. Anything
            # else would bill each image for the whole call.
            if (call_cost := self._cost_from(data)) and produced:
                for image in produced:
                    tag_cost(image, call_cost / len(produced))
            if not produced:
                # A model that will not draw something usually says why, in the
                # text part of the same reply. Throwing that away is how
                # "returned no image" becomes the only thing anyone ever learns.
                said = self._text_from(data)
                logger.warning(
                    "openrouter: %s returned no image (finish_reason=%s)%s",
                    model,
                    self._finish_reason(data),
                    f" and said: {said[:400]}" if said else "",
                )
            images.extend(produced)

        if not images:
            reason = f" It replied: “{said[:300]}”" if said else ""
            raise ProviderError(
                f"{model} returned no image.{reason} Not every model on OpenRouter can "
                "produce one, even when its catalog entry says so — and some refuse a "
                "particular prompt rather than the whole job."
            )
        return images

    @staticmethod
    def _cost_from(data: dict) -> float:
        """What OpenRouter says the call cost, in USD credits.

        Reported in `usage`, and only for keys allowed to see it — an account
        with usage accounting off gets images and no price, which is why this
        answers 0.0 rather than raising.
        """
        usage = data.get("usage") or {}
        try:
            return float(usage.get("cost") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _text_from(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "").strip()

    @staticmethod
    def _finish_reason(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return "?"
        return str(choices[0].get("finish_reason") or choices[0].get("native_finish_reason") or "?")

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
            },
            what="a prompt rewrite",
        )
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        text = text.strip()
        if not text:
            raise ProviderError(f"{model} returned an empty prompt")
        return text
