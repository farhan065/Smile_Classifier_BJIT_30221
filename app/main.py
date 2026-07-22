"""Smile Classifier - 30221
FastAPI application entry point.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from sqlalchemy.orm import Session

from app.config import MAX_FILE_SIZE_MB, UPLOAD_DIR
from app.database import get_db
from app.ml import inference
from app.models import History
from app.utils import UploadError, save_as_jpg, validate_and_read

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Smile Classifier - 30221")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page (requirement #7)."""
    return templates.TemplateResponse(request, "home.html", {"active": "home"})


@app.get("/classify", response_class=HTMLResponse)
def classify_form(request: Request):
    """Show the upload form."""
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