"""Runware: a GPU marketplace behind one task-shaped API.

It differs from OpenRouter in the two ways that matter to this module.

Its catalog is enormous — it mirrors civitai on top of its own featured
models — so it cannot be fetched and filtered here; every query goes to
`modelSearch` and comes back paginated. And nothing is readable without a
credential, so the tab has to say "add a key" rather than showing a list.

What it gives back is a real diffusion host rather than a chat model that
happens to emit an image: `imageInference` takes a prompt, a size and a
checkpoint, and returns the picture.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import uuid

import httpx
from PIL import Image

from .base import ModelPage, Provider, ProviderError, RemoteModel

logger = logging.getLogger(__name__)

# Runware quotes capabilities in more than one register across its catalog
# ("text-to-image", "textToImage", "Text to Image"), so they are compared with
# the punctuation and case taken out.
_SQUASH = re.compile(r"[^a-z0-9]")

# Capabilities that mean "give it a prompt and it produces a picture".
# Deliberately excludes the utility heads — upscale, background removal,
# vectorise — which return an image but ignore what you asked for.
MAKES_IMAGES = frozenset(
    {"texttoimage", "imagetoimage", "inpainting", "outpainting", "edit"}
)
# Capabilities that mean the model takes an image in.
READS_IMAGES = frozenset(
    {
        "imagetoimage",
        "inpainting",
        "outpainting",
        "edit",
        "upscale",
        "removebackground",
        "vectorize",
        "controlnet",
    }
)

# Everything else in the catalog is an ingredient rather than a model you can
# generate with: LoRAs, VAEs, embeddings.
GENERATOR_CATEGORY = "checkpoint"


def _squash(values) -> set[str]:
    return {_SQUASH.sub("", str(value).lower()) for value in values or ()}


def _to_data_uri(image: Image.Image, max_side: int = 1536) -> str:
    copy = image.convert("RGB")
    if max(copy.size) > max_side:
        copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _fit(value: int | None, fallback: int) -> int:
    """Runware takes sizes in steps of 64 within 128..2048.

    The studio's own budget is a multiple of 16 and derived from local VRAM,
    neither of which Runware cares about, so the number is squared up here
    rather than refusing a request over a rounding difference.
    """
    pixels = int(value or fallback)
    pixels = max(128, min(2048, pixels))
    return int(round(pixels / 64)) * 64


class RunwareProvider(Provider):
    id = "runware"
    label = "Runware"
    summary = (
        "A GPU marketplace: FLUX, SDXL, Qwen and Seedream alongside a mirror of "
        "civitai. The catalog needs a key to read, and is searched rather than "
        "listed — there are far too many models to show at once."
    )
    docs_url = "https://runware.ai/docs"
    key_url = "https://my.runware.ai/keys"
    catalog_is_public = False
    # Runware hosts image checkpoints. There is no text catalog to offer, and a
    # filter that returned nothing would only look broken.
    kinds = ("image", "all")

    def __init__(
        self,
        api_key: str | None,
        key_source: str = "none",
        base_url: str = "https://api.runware.ai/v1",
        timeout: float = 180.0,
    ) -> None:
        super().__init__(api_key, key_source)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # --------------------------------------------------------------- requests
    def _run(self, task: dict, timeout: float | None = None) -> list[dict]:
        """Post one task and return the `data` entries it produced."""
        if not self.api_key:
            raise ProviderError(
                "Runware needs an API key — its catalog is not public. Add one on the "
                "Web models tab or set OIG_RUNWARE_API_KEY."
            )
        body = [{"taskUUID": str(uuid.uuid4()), **task}]
        try:
            response = httpx.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout or self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"could not reach Runware: {exc}") from exc

        try:
            payload = response.json()
        except ValueError:
            raise ProviderError(
                f"Runware answered {response.status_code} with something that was not JSON"
            ) from None

        errors = payload.get("errors") or []
        if errors:
            raise ProviderError(self._explain(errors[0], response.status_code))
        if response.status_code >= 400:
            raise ProviderError(f"Runware refused the request ({response.status_code})")
        return payload.get("data") or []

    @staticmethod
    def _explain(error: dict, status: int) -> str:
        """Runware's own message, which is usually the most useful thing here."""
        code = str(error.get("code") or "")
        message = str(error.get("message") or "").strip()
        if code == "invalidApiKey" or status == 401:
            return "Runware rejected the API key."
        if code == "insufficientCredits" or status == 402:
            return "Runware reports no credit left on this account."
        if error.get("parameter") == "search":
            # The docs mark `search` as required. If a blank one is refused,
            # say what to do about it rather than repeating the field name.
            return "Runware needs something to search for. Type a model name or an AIR id."
        return f"Runware refused the request: {message or code or status}"

    # ---------------------------------------------------------------- catalog
    def search_catalog(
        self,
        *,
        query: str = "",
        kind: str = "image",
        limit: int = 60,
        include_routers: bool = False,
    ) -> ModelPage:
        task: dict = {
            "taskType": "modelSearch",
            "search": query.strip(),
            "limit": max(1, min(100, limit)),
            "offset": 0,
            "sort": "popularity",
        }
        if kind == "image":
            # Only checkpoints generate; the rest of the catalog is LoRAs and
            # VAEs that attach to one.
            task["category"] = GENERATOR_CATEGORY

        data = self._run(task, timeout=60.0)
        entry = data[0] if data else {}
        models = [self._model(result) for result in entry.get("results") or []]
        if kind == "image":
            models = [model for model in models if model.makes_images]
        return ModelPage(
            models=models,
            total=int(entry.get("totalResults") or len(models)),
            # A searchable catalog has no total to report: what "all of Runware"
            # holds is not a number the operator could act on.
            catalog_total=0,
        )

    def get_model(self, model_id: str) -> RemoteModel | None:
        """Resolve one AIR identifier. Runware's search matches it directly."""
        data = self._run(
            {
                "taskType": "modelSearch",
                "search": model_id,
                "visibility": "all",
                "limit": 25,
                "offset": 0,
            },
            timeout=60.0,
        )
        entry = data[0] if data else {}
        for result in entry.get("results") or []:
            if str(result.get("air") or "") == model_id:
                return self._model(result)
        return None

    @staticmethod
    def _model(entry: dict) -> RemoteModel:
        air = str(entry.get("air") or "")
        category = str(entry.get("category") or "").lower()
        capabilities = _squash(entry.get("capabilities"))

        if capabilities:
            makes = bool(capabilities & MAKES_IMAGES)
            reads = bool(capabilities & READS_IMAGES)
        else:
            # Older entries declare no capabilities. A checkpoint generates from
            # a prompt by definition; whether it also accepts a reference is not
            # something to assume, so an edit is refused rather than attempted.
            makes = category == GENERATOR_CATEGORY
            reads = False

        description = str(entry.get("shortDescription") or entry.get("comment") or "").strip()
        architecture = entry.get("architecture")
        tags = [str(tag) for tag in entry.get("tags") or []]
        # The architecture is what tells FLUX from SDXL from Qwen, and it is the
        # first thing an operator looks for. The catalog list shows the
        # description, so it goes in there rather than into a field the UI would
        # have to learn about.
        prefix = " · ".join(part for part in [architecture, ", ".join(tags[:4])] if part)
        if prefix:
            description = f"{prefix} — {description}" if description else str(prefix)

        return RemoteModel(
            id=air,
            name=str(entry.get("name") or air),
            description=description,
            input_modalities=("text",) + (("image",) if reads else ()),
            output_modalities=("image",) if makes else (),
            # Runware prices per generation, from the model and the size asked
            # for; there is no per-image figure in the catalog to quote here.
            price_image=None,
        )

    # ------------------------------------------------------------- generation
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
        task: dict = {
            "taskType": "imageInference",
            "model": model,
            "positivePrompt": prompt,
            "width": _fit(width, 1024),
            "height": _fit(height, 1024),
            "numberResults": max(1, num_images),
            # Asking for the bytes rather than a URL: Runware keeps generated
            # images for seven days and this archive is meant to outlive that.
            "outputType": "base64Data",
            "outputFormat": "PNG",
            "includeCost": True,
        }
        if references:
            task["referenceImages"] = [_to_data_uri(reference) for reference in references]

        data = self._run(task)
        images: list[Image.Image] = []
        cost = 0.0
        for entry in data:
            raw = entry.get("imageBase64Data") or entry.get("imageDataURI")
            if not raw:
                continue
            if isinstance(raw, str) and raw.startswith("data:"):
                raw = raw.split(",", 1)[-1]
            images.append(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB"))
            cost += float(entry.get("cost") or 0.0)

        if not images:
            raise ProviderError(f"{model} returned no image")
        if cost:
            logger.info("runware billed $%.4f for %d image(s) from %s", cost, len(images), model)
        return images
