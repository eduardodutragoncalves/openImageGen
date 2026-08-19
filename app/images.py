"""Image encoding/decoding helpers for the HTTP layer."""

from __future__ import annotations

import base64
import binascii
import io
import re

from PIL import Image

_DATA_URI = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

_PIL_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


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


def encode_image(image: Image.Image, fmt: str = "png", *, quality: int = 95) -> str:
    """Serialize a PIL image to base64 in the requested format."""
    pil_format = _PIL_FORMAT.get(fmt.lower())
    if pil_format is None:
        raise ValueError(f"unsupported output format: {fmt}")

    buffer = io.BytesIO()
    save_kwargs: dict[str, object] = {}
    if pil_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    if pil_format == "JPEG":
        save_kwargs["subsampling"] = 0

    image.save(buffer, format=pil_format, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


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
