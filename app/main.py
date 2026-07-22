"""Smile Classifier - 30221
FastAPI application entry point.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import UPLOAD_DIR

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Smile Classifier - 30221")

# Serve static files (CSS, uploaded images) directly to the browser.
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Tell Jinja2 where the HTML templates live.
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Make sure the uploads folder exists when the app starts.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Home page — explains the model and framework (requirement #7)."""
    return templates.TemplateResponse(
        request, "home.html", {"active": "home"}
    )