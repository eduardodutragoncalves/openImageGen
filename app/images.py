"""Image encoding/decoding helpers for the HTTP layer."""

from __future__ import annotations

import base64
import binascii
import io
import re
from pathlib import Path

from PIL import Image, PngImagePlugin

_DATA_URI = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

_PIL_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}

SOFTWARE = "openImageGen"

# EXIF tag numbers, for the formats that have no text chunks.
_EXIF_IMAGE_DESCRIPTION = 0x010E
_EXIF_SOFTWARE = 0x0131


class InvalidImage(ValueError):
    """Raised when a supplied reference image cannot be decoded."""


def decode_image(payload: str, *, max_bytes: int = 32 * 1024 * 1024) -> Image.Image:
    """Decode a base64 payload or data URI into an RGB image."""
    raw = _DATA_URI.sub("", payload.strip())
    try:
        data = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImage("reference image is not valid base64") from exc

    if len(data) > max_bytes:
        raise InvalidImage(f"reference image exceeds {max_bytes // (1024 * 1024)}MB")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001 - PIL raises a zoo of errors
        raise InvalidImage(f"could not decode reference image: {exc}") from exc

    return image.convert("RGB")


def build_metadata(
    *,
    model_id: str | None = None,
    model_label: str | None = None,
    cost: float | None = None,
    currency: str = "USD",
) -> dict[str, str]:
    """What a generated file should say about itself.

    An image outlives the job row that describes it — it gets downloaded,
    copied, mailed on — so what produced it and what it billed travel inside
    the file rather than only in the archive. Cost is included only when the
    provider actually quoted one: an absent price and a free generation are
    different claims, and writing "0.00" would turn the first into the second.
    """
    metadata: dict[str, str] = {"Software": SOFTWARE}
    if model_label:
        metadata["Model"] = str(model_label)
    if model_id and model_id != model_label:
        # The label is what an operator recognises; the id is what reproduces
        # the generation. Both, unless they are the same string.
        metadata["Model ID"] = str(model_id)
    if cost:
        # Six places because a single image can bill fractions of a cent, and
        # rounding those away is how a per-image price becomes zero.
        metadata["Cost"] = f"{float(cost):.6f}"
        metadata["Cost Currency"] = currency
    return metadata


def _save_kwargs(
    pil_format: str, quality: int, metadata: dict[str, str] | None
) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if pil_format in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    if pil_format == "JPEG":
        kwargs["subsampling"] = 0
    if not metadata:
        return kwargs

    if pil_format == "PNG":
        info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            info.add_text(key, value)
        kwargs["pnginfo"] = info
    else:
        # JPEG and WebP have no text chunks, so the same facts go into the two
        # EXIF fields every viewer already shows.
        exif = Image.Exif()
        exif[_EXIF_SOFTWARE] = SOFTWARE
        exif[_EXIF_IMAGE_DESCRIPTION] = "; ".join(
            f"{key}: {value}" for key, value in metadata.items() if key != "Software"
        )
        kwargs["exif"] = exif.tobytes()
    return kwargs


def _format_for(fmt: str) -> str:
    pil_format = _PIL_FORMAT.get(fmt.lower())
    if pil_format is None:
        raise ValueError(f"unsupported output format: {fmt}")
    return pil_format


def encode_image(
    image: Image.Image,
    fmt: str = "png",
    *,
    quality: int = 95,
    metadata: dict[str, str] | None = None,
) -> str:
    """Serialize a PIL image to base64 in the requested format."""
    pil_format = _format_for(fmt)
    buffer = io.BytesIO()
    image.save(buffer, format=pil_format, **_save_kwargs(pil_format, quality, metadata))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def save_image(
    image: Image.Image,
    path: Path,
    fmt: str = "png",
    *,
    quality: int = 95,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a PIL image to disk with its metadata attached."""
    pil_format = _format_for(fmt)
    image.save(path, format=pil_format, **_save_kwargs(pil_format, quality, metadata))


def read_metadata(path: Path) -> dict[str, str]:
    """The metadata openImageGen wrote into a file, as far as it can be read.

    Used by the tests, and by anyone asking a saved image what made it.
    """
    with Image.open(path) as image:
        if image.format == "PNG":
            return {
                key: value
                for key, value in image.info.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        exif = image.getexif()
        described = str(exif.get(_EXIF_IMAGE_DESCRIPTION) or "")
        metadata = {"Software": str(exif.get(_EXIF_SOFTWARE) or "")} if exif else {}
        for part in filter(None, (piece.strip() for piece in described.split(";"))):
            key, _, value = part.partition(":")
            metadata[key.strip()] = value.strip()
        return {key: value for key, value in metadata.items() if value}


def mime_type(fmt: str) -> str:
    return _MIME.get(fmt.lower(), "application/octet-stream")


def fit_to_budget(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Scale a request down to the pixel budget, keeping multiples of 16."""
    width = max(16, 16 * (width // 16))
    height = max(16, 16 * (height // 16))

    if width * height <= max_pixels:
        return width, height

    scale = (max_pixels / (width * height)) ** 0.5
    width = max(16, 16 * (int(width * scale) // 16))
    height = max(16, 16 * (int(height * scale) // 16))
    return width, height
