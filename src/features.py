"""Classical feature extraction for preprocessed signature images."""

from __future__ import annotations

import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.measure import moments_hu

from config import (
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
    LBP_POINTS,
    LBP_RADIUS,
)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Extract HOG, geometry, Hu moments, and an LBP histogram."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D preprocessed image, got {image.shape}.")
    gray = np.clip(image.astype(np.float32), 0.0, 1.0)
    ink = gray > 0.5
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )
    rows, columns = np.nonzero(ink)
    if rows.size:
        min_row, max_row = int(rows.min()), int(rows.max())
        min_col, max_col = int(columns.min()), int(columns.max())
        aspect_ratio = (max_col - min_col + 1) / max(max_row - min_row + 1, 1)
        centroid_x = float(columns.mean()) / max(IMAGE_SIZE - 1, 1)
        centroid_y = float(rows.mean()) / max(IMAGE_SIZE - 1, 1)
    else:
        aspect_ratio = centroid_x = centroid_y = 0.0
    density = float(ink.mean())
    hu = moments_hu(ink.astype(np.float64))
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    lbp = local_binary_pattern(
        ink.astype(np.uint8), LBP_POINTS, LBP_RADIUS, method="uniform"
    )
    lbp_histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
        density=True,
    )
    return np.concatenate(
        [
            np.asarray(hog_features, dtype=np.float32),
            np.asarray([aspect_ratio, density, centroid_x, centroid_y], dtype=np.float32),
            hu.astype(np.float32),
            lbp_histogram.astype(np.float32),
        ]
    )


def extract_feature_matrix(images: np.ndarray) -> np.ndarray:
    """Extract one feature vector per 2-D image."""
    if len(images) == 0:
        raise ValueError("Cannot extract features from an empty image collection.")
    return np.stack([extract_features(image) for image in images])
