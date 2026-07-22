"""Training script for the smile classifier.

Run from the project root with:  python -m app.ml.train
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.preprocessing import image_to_features

# --- Paths (all relative to the project root, so it works anywhere) ---
BASE_DIR = Path(__file__).resolve().parents[2]   # .../smile-classifier-30221
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "smile_model.pkl"

# Folder name -> numeric label the model learns.
CLASS_FOLDERS = {"non_smile": 0, "smile": 1}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_dataset(data_dir: Path):
    """Read every image, convert to features, and collect its label."""
    features, labels = [], []
    for folder_name, label in CLASS_FOLDERS.items():
        folder = data_dir / folder_name
        if not folder.exists():
            continue
        for image_path in folder.iterdir():
            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as img:
                    features.append(image_to_features(img))
                    labels.append(label)
            except Exception as exc:  # a corrupt image shouldn't stop training
                print(f"  Skipping {image_path.name}: {exc}")
    return np.array(features), np.array(labels)


def build_model() -> Pipeline:
    """Scaler + SVM bundled together and saved as one object."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])


def train_and_save():
    print("Loading images and extracting features...")
    X, y = load_dataset(DATA_DIR)
    if len(X) == 0:
        raise RuntimeError("No images found in data/. Check the folder names.")
    print(f"Loaded {len(X)} images "
          f"({int((y == 1).sum())} smiling, {int((y == 0).sum())} not smiling).")

    # Hold back 20% to honestly measure accuracy on unseen images.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training the model (this can take a little while)...")
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(
        y_test, y_pred, target_names=["Not Smiling", "Smiling"]
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}")
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    train_and_save()