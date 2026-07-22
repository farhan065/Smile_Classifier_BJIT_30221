"""Central configuration, loaded from environment variables (.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def require_env(key: str) -> str:
    """Return an environment variable, or fail loudly if it's missing.

    Secrets must never silently fall back to a default value, so a missing
    one is a configuration error we want to catch immediately at startup.
    """
    value = os.getenv(key)
    if value is None or value == "":
        raise RuntimeError(
            f"Required environment variable '{key}' is not set. "
            f"Check that your .env file exists and defines it."
        )
    return value


# --- Database settings (required — no fallback) ---
POSTGRES_USER = require_env("POSTGRES_USER")
POSTGRES_PASSWORD = require_env("POSTGRES_PASSWORD")
POSTGRES_DB = require_env("POSTGRES_DB")
POSTGRES_HOST = require_env("POSTGRES_HOST")
POSTGRES_PORT = require_env("POSTGRES_PORT")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# --- Upload validation rules (safe, non-secret defaults are fine here) ---
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TRAIN_FILES = int(os.getenv("MAX_TRAIN_FILES", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --- Filesystem paths ---
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "smile_model.pkl"
DATA_DIR = BASE_DIR / "data"

# Staging folder where Train-page uploads accumulate by class before training.
TRAIN_UPLOAD_DIR = BASE_DIR / "app" / "static" / "train_uploads"
TRAIN_CLASS_FOLDERS = {"Smiling": "smile", "Not Smiling": "non_smile"}