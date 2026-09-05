"""Runware: a GPU marketplace behind one task-shaped API.

It offers its catalog twice over, and this module uses both.

`content.runware.ai` publishes the curated set — a few hundred models with a
cover image, a headline and a declared capability list, served to anyone
without a credential. That is what the operator browses, and it is the same
catalog Runware's own model picker is built on.

`modelSearch`, on the paid API, reaches everything else: the civitai mirror,
hundreds of thousands of community checkpoints. It needs a key, so it is what a
typed query falls through to once one exists.

What it generates with is a real diffusion host rather than a chat model that
happens to emit an image: `imageInference` takes a prompt, a size and a
checkpoint, and returns the picture.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
import uuid

import httpx
from PIL import Image

from .base import ModelPage, Provider, ProviderError, RemoteModel, tag_cost

logger = logging.getLogger(__name__)

# Runware's capability taxonomy is namespaced — `io:` for what a model takes
# and returns, `op:` for what it does, `form:` for what kind of artefact it is
# (https://content.runware.ai/capabilities). Not every surface quotes the
# namespace, and the spelling varies between them ("text-to-image",
# "textToImage"), so a capability is reduced to its bare suffix before it is
# compared: both `io:text-to-image` and `textToImage` become `texttoimage`.
_NAMESPACE = re.compile(r"^[a-z]+:")
_SQUASH = re.compile(r"[^a-z0-9]")

# The `io:` capabilities say what comes out. Only these two end in a picture;
# `io:text-to-video` and `io:image-to-3d` end in something this application
# cannot show.
OUTPUTS_IMAGE = frozenset({"texttoimage", "imagetoimage"})
# What a model does to an image it is given. `io:image-to-image` is not on this
# list on purpose: a background remover and an upscaler both declare it, and
# neither one draws what you asked for. Only an editing op does.
EDITS_IMAGES = frozenset(
    {"edit", "inpaint", "inpainting", "extend", "outpaint", "outpainting"}
)
# Capabilities that mean the model takes an image in.
READS_IMAGES = frozenset(
    {
        "imagetoimage",
        "imagetotext",
        "imagetovideo",
        "imageto3d",
        "edit",
        "inpaint",
        "inpainting",
        "extend",
        "outpaint",
        "outpainting",
        "upscale",
        "removebackground",
        "vectorize",
        "controlnet",
    }
)

# Everything else in the catalog is an ingredient rather than a model you can
# generate with: LoRAs, VAEs, embeddings.
GENERATOR_CATEGORY = "checkpoint"

# The curated catalog is one 400KB document that changes when Runware ships a
# model, so it is fetched once and held. A provider instance is built per
# request, so the cache has to outlive it.
CURATED_URL = "https://content.runware.ai/models"
_CURATED_TTL_S = 900.0
_curated_lock = threading.Lock()
_curated_cache: tuple[float, list["RemoteModel"]] = (0.0, [])


def _squash(values) -> set[str]:
    """Capabilities as bare comparable tokens, namespace and punctuation gone."""
    return {
        _SQUASH.sub("", _NAMESPACE.sub("", str(value).lower().strip()))
        for value in values or ()
    }


def _to_data_uri(image: Image.Image, max_side: int = 1536) -> str:
    copy = image.convert("RGB")
    if max(copy.size) > max_side:
        copy.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _creator(entry: dict) -> str:
    """Who made it, from whichever shape this surface uses.

    The curated document names the creator with a slug; `modelSearch` answers
    its featured entries with the whole creator record — id, name, logo, a
    paragraph of description — and its community ones with nothing at all.
    """
    value = entry.get("creator") or entry.get("provider") or ""
    if isinstance(value, dict):
        value = value.get("name") or value.get("id") or ""
    return str(value)


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
        "A GPU marketplace: FLUX, SDXL, Qwen and Seedream, plus a mirror of "
        "civitai. The curated catalog browses without a key; searching the "
        "whole of it, and generating, needs one."
    )
    docs_url = "https://runware.ai/docs"
    key_url = "https://my.runware.ai/keys"
    # The curated catalog is served publicly, so there is something to look at
    # before anyone pastes a credential.
    catalog_is_public = True
    # Runware hosts image checkpoints, so there is no text catalog to offer.
    # "community" is the third step: the civitai mirror behind the paid API,
    # which only a typed query can reach.
    kinds = ("image", "all", "community")

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
                "Runware needs an API key for this. Browsing the curated catalog is "
                "free; searching the community mirror and generating are not. Add a "
                "key on the Web models tab or set OIG_RUNWARE_API_KEY."
            )
        body = [{"taskUUID": str(uuid.uuid4()), **task}]
        what = str(task.get("taskType") or "a request")
        started = time.perf_counter()
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
            logger.warning("runware: %s could not be sent: %s", what, exc)
            raise ProviderError(f"could not reach Runware: {exc}", retryable=True) from exc

        elapsed = time.perf_counter() - started
        payload = self._decode(response, what)

        errors = payload.get("errors") or []
        if errors:
            # Runware's own error object carries more than the sentence shown
            # to the operator, and the rest of it is what a bug report needs.
            logger.warning(
                "runware: %s failed after %.1fs — HTTP %s: %s",
                what, elapsed, response.status_code, json.dumps(errors[0], ensure_ascii=False)[:600],
            )
            raise ProviderError(
                self._explain(errors[0], response.status_code),
                retryable=str(errors[0].get("code") or "")
                in {"timeoutProvider", "providerRateLimitExceeded"},
            )
        if response.status_code >= 400:
            logger.warning("runware: %s answered HTTP %s", what, response.status_code)
            raise ProviderError(f"Runware refused the request ({response.status_code})")

        logger.info("runware: %s in %.1fs", what, elapsed)
        return payload.get("data") or []

    @staticmethod
    def _decode(response: httpx.Response, what: str) -> dict:
        """Runware's answer, recovered where it can be.

        Twice on a live server a 200 came back carrying a finished image *and
        its price*, and the whole body was rejected as not-JSON — so the
        operator was billed for a picture this process then threw away. The
        head of both bodies was well-formed, which rules out an error page and
        points at whatever follows the first document.

        So the first complete JSON value is parsed and anything after it is
        ignored, which salvages the image whenever the prefix is whole. Where
        it is not, the log now carries the byte length, the parser's own
        complaint and the tail — which is what tells a truncated body from
        trailing bytes, and is precisely what the first two of these could not
        say, because only the first 400 characters were ever recorded.
        """
        try:
            return response.json()
        except ValueError as exc:
            text = response.text
            salvaged = None
            try:
                value, _end = json.JSONDecoder().raw_decode(text.lstrip())
                if isinstance(value, dict):
                    salvaged = value
            except ValueError:
                pass

            if salvaged is not None:
                logger.warning(
                    "runware: %s answered HTTP %s with %d bytes that would not parse "
                    "whole (%s); recovered the first document and used it",
                    what, response.status_code, len(text), exc,
                )
                return salvaged

            logger.warning(
                "runware: %s answered HTTP %s with %d bytes of non-JSON (%s)\n"
                "  head: %s\n  tail: %s",
                what, response.status_code, len(text), exc, text[:300], text[-300:],
            )
            # Deliberately not retryable: a generation that got this far was
            # billed, and trying again spends that a second time. Whether to
            # pay twice is the operator's call, not this loop's.
            raise ProviderError(
                f"Runware answered {response.status_code} with {len(text)} bytes "
                f"that could not be parsed ({exc}). If it billed for this, the image "
                "was made and lost in transit rather than never made."
            ) from None

    @staticmethod
    def _explain(error: dict, status: int) -> str:
        """Runware's own message, which is usually the most useful thing here."""
        code = str(error.get("code") or "")
        message = str(error.get("message") or "").strip()
        if code == "invalidApiKey" or status == 401:
            return "Runware rejected the API key."
        if code == "insufficientCredits" or status == 402:
            return "Runware reports no credit left on this account."
        if code in {"timeoutProvider", "providerRateLimitExceeded"}:
            return f"Runware's upstream is busy ({code}); it is worth trying again."
        if error.get("parameter") == "search":
            # The docs mark `search` as required and the API accepts a blank
            # one anyway. Kept for the day that changes, and phrased as what to
            # do rather than as the name of a field.
            return "Runware needs something to search for. Type a model name or an AIR id."
        return f"Runware refused the request: {message or code or status}"

    # ---------------------------------------------------------------- catalog
    def list_models(self) -> list[RemoteModel]:
        """The curated catalog, which is public.

        One document of a few hundred models, held for a while rather than
        re-fetched per keystroke. Filtering and searching it is the base
        class's job.
        """
        global _curated_cache
        fetched_at, cached = _curated_cache
        if cached and time.monotonic() - fetched_at < _CURATED_TTL_S:
            return cached

        try:
            response = httpx.get(CURATED_URL, timeout=30.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            if cached:
                # Stale beats empty: the catalog changes when Runware ships a
                # model, not by the minute.
                logger.warning("could not refresh Runware's catalog (%s); serving the held copy", exc)
                return cached
            raise ProviderError(f"could not reach Runware's catalog: {exc}") from exc

        entries = payload if isinstance(payload, list) else payload.get("results") or []
        # `weight` is the catalog's own prominence, and it is the order Runware
        # shows these in. Sorting by name instead would open the list on six
        # background removers.
        entries = sorted(
            entries,
            key=lambda entry: (-float(entry.get("weight") or 0), str(entry.get("name") or "")),
        )
        models = [self._model(entry) for entry in entries]
        with _curated_lock:
            _curated_cache = (time.monotonic(), models)
        return models

    def search_catalog(
        self,
        *,
        query: str = "",
        kind: str = "image",
        limit: int = 60,
        include_routers: bool = False,
    ) -> ModelPage:
        if kind != "community":
            return super().search_catalog(
                query=query, kind=kind, limit=limit, include_routers=include_routers
            )

        # The community mirror is the rest of civitai: hundreds of thousands of
        # checkpoints, reachable only through the paid API and only by
        # searching. It is a deliberate second step, not the default view.
        task: dict = {
            "taskType": "modelSearch",
            "search": query.strip(),
            "category": GENERATOR_CATEGORY,
            "limit": max(1, min(100, limit)),
            "offset": 0,
            "sort": "popularity",
        }
        data = self._run(task, timeout=60.0)
        entry = data[0] if data else {}
        models = [self._model(result) for result in entry.get("results") or []]
        models = [model for model in models if model.makes_images]
        return ModelPage(
            models=models,
            total=int(entry.get("totalResults") or len(models)),
            # A searched catalog has no total to report: what the whole mirror
            # holds is not a number the operator could act on.
            catalog_total=0,
        )

    def _check_key(self) -> None:
        """The curated catalog answers without a credential, so it proves
        nothing here. One search against the paid API does."""
        self._run(
            {
                "taskType": "modelSearch",
                "search": "flux",
                "category": GENERATOR_CATEGORY,
                "limit": 1,
                "offset": 0,
            },
            timeout=30.0,
        )

    def get_model(self, model_id: str) -> RemoteModel | None:
        """Resolve one AIR identifier.

        The curated catalog answers for free and covers everything the tab
        shows by default. Only a community model has to be looked up on the
        paid API, where the search matches an AIR id directly.
        """
        found = next((m for m in self.list_models() if m.id == model_id), None)
        if found is not None or not self.api_key:
            return found

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
        """One catalog entry, from either surface.

        The curated document and `modelSearch` describe the same thing with
        different field names — `headline` against `shortDescription`,
        `coverImage` against `heroImage`, `creator` against `provider` — so
        both spellings are read here rather than in two mapping functions that
        would drift apart.
        """
        air = str(entry.get("air") or "")
        category = str(entry.get("category") or "").lower()
        capabilities = _squash(entry.get("capabilities"))

        if capabilities:
            # Two conditions, and both are needed. It has to end in a picture —
            # a video model declaring `op:edit` edits video — and it has to
            # draw rather than post-process, which an `op:upscale` head does
            # not, however faithfully it returns an image.
            makes = bool(capabilities & OUTPUTS_IMAGE) and (
                "texttoimage" in capabilities or bool(capabilities & EDITS_IMAGES)
            )
            reads = bool(capabilities & READS_IMAGES)
        else:
            # Much of the community mirror declares no capabilities at all. A
            # checkpoint generates from a prompt by definition; whether it also
            # accepts a reference is not something to assume, so an edit is
            # refused rather than attempted and billed.
            makes = category == GENERATOR_CATEGORY
            reads = False

        description = str(
            entry.get("headline")
            or entry.get("shortDescription")
            or entry.get("comment")
            or ""
        ).strip()
        # The architecture is what tells FLUX from SDXL from Qwen, and it is the
        # first thing an operator looks for. The list shows the description, so
        # it goes in there rather than into a field the UI would have to learn.
        tags = [str(tag) for tag in entry.get("tags") or []]
        prefix = " · ".join(
            part for part in [entry.get("architecture"), ", ".join(tags[:4])] if part
        )
        if prefix:
            description = f"{prefix} — {description}" if description else str(prefix)

        return RemoteModel(
            id=air,
            name=str(entry.get("name") or air),
            description=description,
            input_modalities=("text",) + (("image",) if reads else ()),
            output_modalities=("image",) if makes else (),
            # Runware prices per generation, from the model and the size asked
            # for. The curated catalog quotes that as a sentence rather than a
            # per-unit number, so it is carried as one.
            price_note=(str(entry.get("pricingOverview") or "").strip() or None),
            cover_image=(str(entry.get("coverImage") or entry.get("heroImage") or "") or None),
            creator=_creator(entry),
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
            image = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
            # Runware prices each result separately, so the price stays with
            # the image it belongs to rather than being averaged over a batch.
            entry_cost = float(entry.get("cost") or 0.0)
            images.append(tag_cost(image, entry_cost))
            cost += entry_cost

        if not images:
            raise ProviderError(f"{model} returned no image")
        if cost:
            logger.info("runware billed $%.4f for %d image(s) from %s", cost, len(images), model)
        return images
