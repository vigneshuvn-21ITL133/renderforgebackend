"""Validation of uploaded photos before they are queued for generation."""

import io
from typing import Final

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

# Pillow format name -> (file extension, MIME type)
SUPPORTED_FORMATS: Final[dict[str, tuple[str, str]]] = {
    "PNG": ("png", "image/png"),
    "JPEG": ("jpg", "image/jpeg"),
    "WEBP": ("webp", "image/webp"),
}


class InvalidImageError(ValueError):
    """The bytes are not a complete image in a supported format."""


def detect_image(data: bytes) -> tuple[str, str]:
    """Return ``(extension, mime_type)`` for a valid image, or raise InvalidImageError.

    The client-supplied Content-Type is not trusted. Pillow decides the real format.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("Uploaded file is not a readable image") from exc

    if image_format not in SUPPORTED_FORMATS:
        raise InvalidImageError("Only PNG, JPEG and WebP images are supported")
    return SUPPORTED_FORMATS[image_format]


async def read_image_upload(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, enforcing the size limit without buffering more than one byte past it."""
    data = await upload.read(max_bytes + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Image is larger than {max_bytes // (1024 * 1024)} MB",
        )
    return data
