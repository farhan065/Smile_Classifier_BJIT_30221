"""Training script for the smile classifier.

Run from the project root with:  python -m app.ml.train
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.preprocessing import image_to_features

# --- Paths (all relative to the project root, so it works anywhere) ---
BASE_DIR = Path(__file__).resolve().parents[2]   # .../smile-classifier-30221
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "smile_model.pkl"
MODEL_META_PATH = MODEL_DIR / "model_meta.json"

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


def estimate_accuracy(X: np.ndarray, y: np.ndarray):
    """Estimate accuracy using stratified k-fold cross-validation.

    On small datasets a single train/test split is unreliable, so we average
    across folds instead. Returns (accuracy, evaluation_description) or
    (None, reason) when there is too little data to evaluate at all.
    """
    smallest_class = int(min(np.sum(y == 0), np.sum(y == 1)))

    # Need at least 2 samples in the smaller class to hold one out per fold.
    if smallest_class < 2:
        return None, "Not enough images per class to evaluate"

    n_splits = min(5, smallest_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(), X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), f"{n_splits}-fold cross-validation"


def save_metadata(accuracy: float | None, total_images: int, per_class: dict,
                  source: str, evaluation: str) -> None:
    """Write model metrics to a JSON file so the web app can display them.

    Kept separate from the pickle so metadata can be read cheaply without
    unpickling the whole model.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
        "total_images": total_images,
        "per_class": per_class,
        "source": source,
        "evaluation": evaluation,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "SVM (RBF kernel) on HOG features",
        "framework": "scikit-learn",
    }
    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_and_save():
    """Train on the full dataset in data/ with a held-out test split."""
    print("Loading images and extracting features...")
    X, y = load_dataset(DATA_DIR)
    if len(X) == 0:
        raise RuntimeError("No images found in data/. Check the folder names.")
    print(f"Loaded {len(X)} images "
          f"({int((y == 1).sum())} smiling, {int((y == 0).sum())} not smiling).")

    # Plenty of data here, so a single held-out test split is reliable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training the model (this can take a little while)...")
    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.2%}\n")
    print(classification_report(
        y_test, y_pred, target_names=["Not Smiling", "Smiling"]
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}")

    save_metadata(
        accuracy=accuracy,
        total_images=len(X),
        per_class={"smile": int((y == 1).sum()),
                   "non_smile": int((y == 0).sum())},
        source="Full dataset (data/)",
        evaluation="20% held-out test set",
    )
    print(f"Metadata saved to: {MODEL_META_PATH}")
    return accuracy


def train_from_directory(source_dir: Path) -> dict:
    """Train the model from a directory containing class subfolders.

    Expects subfolders named like CLASS_FOLDERS (e.g. 'smile', 'non_smile').
    Accuracy is estimated with cross-validation (reliable on small batches),
    then the final model is fitted on ALL images so none are wasted.
    """
    features, labels = [], []
    per_class_counts = {}

    for folder_name, label in CLASS_FOLDERS.items():
        folder = source_dir / folder_name
        count = 0
        if folder.exists():
            for image_path in folder.iterdir():
                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                try:
                    with Image.open(image_path) as img:
                        features.append(image_to_features(img))
                        labels.append(label)
                        count += 1
                except Exception as exc:
                    print(f"  Skipping {image_path.name}: {exc}")
        per_class_counts[folder_name] = count

    # We need at least one image in EACH class to train a 2-class model.
    present_classes = [c for c, n in per_class_counts.items() if n > 0]
    if len(present_classes) < 2:
        raise ValueError(
            "Training needs images in BOTH classes (smiling and not smiling). "
            f"Currently staged: {per_class_counts}."
        )

    X = np.array(features)
    y = np.array(labels)

    # 1. Measure accuracy with cross-validation.
    accuracy, evaluation = estimate_accuracy(X, y)

    # 2. Fit the final model on every image (nothing wasted).
    model = build_model()
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    save_metadata(
        accuracy=accuracy,
        total_images=len(X),
        per_class=per_class_counts,
        source="Uploaded images (Train page)",
        evaluation=evaluation,
    )

    return {
        "total_images": len(X),
        "per_class": per_class_counts,
        "accuracy": round(accuracy * 100, 2) if accuracy is not None else None,
    }


if __name__ == "__main__":
    train_and_save()