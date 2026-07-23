"""Smile Classifier - 30221
FastAPI application entry point.

Routes:
  GET  /                -> Home page (explains model & framework)
  GET  /classify        -> show upload form
  POST /classify        -> validate, predict, save to DB, show result
  GET  /train           -> show training page
  POST /train/upload    -> validate & stage training images by label
  POST /train/run       -> train on staged images, save model, delete uploads
  GET  /history         -> table of all past classifications
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from sqlalchemy.orm import Session

from app.config import (
    DISPLAY_TIMEZONE,
    MAX_FILE_SIZE_MB,
    MAX_TRAIN_FILES,
    TRAIN_CLASS_FOLDERS,
    TRAIN_UPLOAD_DIR,
    UPLOAD_DIR,
)
from app.database import get_db
from app.ml import inference
from app.ml.train import train_from_directory
from app.models import History
from app.utils import UploadError, save_as_jpg, validate_and_read

# --------------------------------------------------------------------------- #
# Application setup
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Smile Classifier - 30221")

# Serve static files (CSS, uploaded images) directly to the browser.
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Tell Jinja2 where the HTML templates live.
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# --------------------------------------------------------------------------- #
# Template filters
#
# Timestamps are STORED in UTC (unambiguous and portable). These filters
# only affect how they are DISPLAYED, using the configured timezone.
# --------------------------------------------------------------------------- #
def to_local_time(value, fmt: str = "%Y-%m-%d %I:%M:%S %p") -> str:
    """Convert a datetime (from the database) to the display timezone."""
    if value is None:
        return ""
    # Values coming back from the database may be naive; treat them as UTC.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime(fmt)


def iso_to_local_time(value, fmt: str = "%Y-%m-%d %I:%M:%S %p") -> str:
    """Convert an ISO-8601 UTC string (from model metadata) to local time."""
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return value  # fall back to showing the raw value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime(fmt)


templates.env.filters["localtime"] = to_local_time
templates.env.filters["isolocaltime"] = iso_to_local_time

# Ensure the uploads folder exists at startup.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _count_staged() -> dict:
    """Count how many images are staged in each training class folder."""
    counts = {"smile": 0, "non_smile": 0}
    for folder in counts:
        path = TRAIN_UPLOAD_DIR / folder
        if path.exists():
            counts[folder] = sum(
                1 for f in path.iterdir() if f.suffix.lower() == ".jpg"
            )
    return counts


def _train_context() -> dict:
    """Base template context shared by all Train page responses."""
    return {
        "active": "train",
        "max_size_mb": MAX_FILE_SIZE_MB,
        "max_files": MAX_TRAIN_FILES,
        "model_info": inference.get_model_info(),
    }


# --------------------------------------------------------------------------- #
# Home (requirement #7)
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page — explains the model and framework."""
    return templates.TemplateResponse(
        request, "home.html",
        {"active": "home", "model_info": inference.get_model_info()},
    )


# --------------------------------------------------------------------------- #
# Classify (requirements #1, #3, #4, #11, #13, #15)
# --------------------------------------------------------------------------- #
@app.get("/classify", response_class=HTMLResponse)
def classify_form(request: Request):
    """Show the classify upload form."""
    return templates.TemplateResponse(
        request, "classify.html",
        {"active": "classify", "max_size_mb": MAX_FILE_SIZE_MB},
    )


@app.post("/classify", response_class=HTMLResponse)
def classify_submit(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Process an uploaded image: validate, predict, save, show result."""

    # Requirement #4 — model must exist before we can predict.
    if not inference.model_exists():
        return templates.TemplateResponse(
            request, "classify.html",
            {
                "active": "classify",
                "max_size_mb": MAX_FILE_SIZE_MB,
                "error": "The model has not been trained yet. "
                         "Please train the model on the Train page first.",
            },
        )

    # Requirements #1, #3 — validate the upload; convert & save as JPG (#11).
    try:
        data = validate_and_read(file)
        saved_path = save_as_jpg(data)
    except UploadError as exc:
        return templates.TemplateResponse(
            request, "classify.html",
            {"active": "classify", "max_size_mb": MAX_FILE_SIZE_MB,
             "error": str(exc)},
        )

    # Run inference on the saved image.
    with Image.open(saved_path) as img:
        predicted_class, confidence = inference.predict(img)

    # Requirements #13, #15 — save the result to PostgreSQL.
    record = History(
        image_path=str(saved_path.relative_to(BASE_DIR)),
        predicted_class=predicted_class,
    )
    db.add(record)
    db.commit()

    # Requirement #13 — show the result on a separate page.
    image_url = f"/static/uploads/{saved_path.name}"
    return templates.TemplateResponse(
        request, "result.html",
        {
            "active": "classify",
            "image_url": image_url,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 1),
        },
    )


# --------------------------------------------------------------------------- #
# Train (requirements #2, #8, #11, #12)
# --------------------------------------------------------------------------- #
@app.get("/train", response_class=HTMLResponse)
def train_form(request: Request):
    """Show the training upload page."""
    ctx = _train_context()
    ctx["staged"] = _count_staged()
    return templates.TemplateResponse(request, "train.html", ctx)


@app.post("/train/upload", response_class=HTMLResponse)
def train_upload(
    request: Request,
    label: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Validate and stage uploaded training images under their class folder."""
    ctx = _train_context()

    # Requirement #2 — limit the number of files per upload.
    if len(files) > MAX_TRAIN_FILES:
        ctx["staged"] = _count_staged()
        ctx["error"] = (
            f"You selected {len(files)} files. "
            f"Please upload at most {MAX_TRAIN_FILES} at a time."
        )
        return templates.TemplateResponse(request, "train.html", ctx)

    # Map the friendly label to its folder (e.g. "Smiling" -> "smile").
    folder_name = TRAIN_CLASS_FOLDERS.get(label)
    if folder_name is None:
        ctx["staged"] = _count_staged()
        ctx["error"] = "Invalid label selected."
        return templates.TemplateResponse(request, "train.html", ctx)

    dest_dir = TRAIN_UPLOAD_DIR / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    try:
        for file in files:
            data = validate_and_read(file)      # reqs #1, #3
            jpg_path = save_as_jpg(data)         # req #11 (saved into UPLOAD_DIR)
            # Move the converted JPG into the correct class staging folder.
            final_path = dest_dir / jpg_path.name
            shutil.move(str(jpg_path), str(final_path))
            saved += 1
    except UploadError as exc:
        ctx["staged"] = _count_staged()
        ctx["error"] = str(exc)
        return templates.TemplateResponse(request, "train.html", ctx)

    ctx["staged"] = _count_staged()
    ctx["success"] = f"Uploaded {saved} image(s) labelled '{label}'."
    return templates.TemplateResponse(request, "train.html", ctx)


@app.post("/train/run", response_class=HTMLResponse)
def train_run(request: Request):
    """Train on all staged images, save the model, then delete the uploads."""
    try:
        summary = train_from_directory(TRAIN_UPLOAD_DIR)   # req #12 (train + save)
    except ValueError as exc:
        ctx = _train_context()
        ctx["staged"] = _count_staged()
        ctx["error"] = str(exc)
        return templates.TemplateResponse(request, "train.html", ctx)

    # Requirement #12 — delete the uploaded training images after training.
    if TRAIN_UPLOAD_DIR.exists():
        shutil.rmtree(TRAIN_UPLOAD_DIR)

    # Built AFTER training so the model info reflects the new model.
    ctx = _train_context()
    ctx["staged"] = _count_staged()
    ctx["success"] = (
        f"Model trained on {summary['total_images']} image(s) and saved. "
        + (f"Estimated accuracy: {summary['accuracy']}%. "
           if summary["accuracy"] is not None else "")
        + "Staged uploads cleared."
    )
    return templates.TemplateResponse(request, "train.html", ctx)


# --------------------------------------------------------------------------- #
# History (requirement #14)
# --------------------------------------------------------------------------- #
@app.get("/history", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)):
    """Show all past classifications, newest first."""
    records = db.query(History).order_by(History.created_at.desc()).all()
    return templates.TemplateResponse(
        request, "history.html", {"active": "history", "records": records}
    )