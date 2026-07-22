"""Inference: predict smiling / not smiling for a single image."""
from __future__ import annotations

import pickle
from pathlib import Path

from PIL import Image

from app.ml.preprocessing import image_to_features

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "model" / "smile_model.pkl"

CLASS_NAMES = {0: "Not Smiling", 1: "Smiling"}


def model_exists() -> bool:
    """Requirement #4: check the trained model is present before predicting."""
    return MODEL_PATH.exists()


def load_model():
    if not model_exists():
        raise FileNotFoundError(
            "Model has not been trained yet. Please train the model first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(image: Image.Image):
    """Return (label_name, confidence 0-1) for one Pillow image."""
    model = load_model()
    features = image_to_features(image).reshape(1, -1)  # model wants 2-D input
    label = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][label])
    return CLASS_NAMES[label], confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.ml.inference <path-to-image>")
        sys.exit(1)
    with Image.open(sys.argv[1]) as img:
        name, conf = predict(img)
    print(f"Prediction: {name} ({conf:.2%} confidence)")