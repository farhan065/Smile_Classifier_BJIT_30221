"""Reusable upload helpers: validation and JPG conversion."""
from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from app.config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    UPLOAD_DIR,
)


class UploadError(Exception):
    """Raised when an uploaded file fails a validation rule."""


def _extension_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_and_read(file: UploadFile) -> bytes:
    """Validate one uploaded file and return its raw bytes."""
    if file is None or not file.filename:
        raise UploadError("No file was selected. Please choose an image.")

    if not _extension_ok(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadError(
            f"'{file.filename}' is not a supported type. "
            f"Allowed formats: {allowed}."
        )

    data = file.file.read()

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise UploadError(
            f"'{file.filename}' is too large. "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    if len(data) == 0:
        raise UploadError(f"'{file.filename}' is empty.")

    return data


def save_as_jpg(data: bytes) -> Path:
    """Convert image bytes to JPG and save to the uploads folder."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(BytesIO(data))
    except Exception:
        raise UploadError("The file is not a valid image or is corrupted.")

    image = image.convert("RGB")
    filename = f"{uuid.uuid4().hex}.jpg"
    path = UPLOAD_DIR / filename
    image.save(path, "JPEG", quality=90)
    return path