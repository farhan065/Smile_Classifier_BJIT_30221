"""Shared image preprocessing.

Both training and inference MUST turn an image into a feature vector in
exactly the same way. If they differed, the model would receive numbers
in a format it never learned, and predictions would be garbage. Keeping
this logic in one place is the single source of truth that prevents that.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog

# Every image is standardized to this size before features are extracted.
# Same size for every image => feature vectors of identical length.
IMAGE_SIZE = (64, 64)

# HOG settings. These describe edge directions across the image.
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def image_to_features(image: Image.Image) -> np.ndarray:
    """Convert a Pillow image into a 1-D feature vector for the model."""
    # 1. Grayscale: a smile is about shape, not color.
    gray = image.convert("L")
    # 2. Resize so every image has identical dimensions.
    gray = gray.resize(IMAGE_SIZE)
    # 3. Scale pixel values from 0-255 down to 0-1 for stable training.
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # 4. HOG: describe edge directions (the shape of the mouth, cheeks...).
    features = hog(arr, **HOG_PARAMS)
    return features