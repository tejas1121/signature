"""Image preprocessing shared by model training and inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import IMAGE_SIZE


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV or RGB image array to 8-bit grayscale."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The supplied image is empty or invalid.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}.")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported channel count: {image.shape[2]}.")


def preprocess_with_steps(
    image: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Preprocess an image and return the normalized image plus visual steps.

    Ink is represented as white pixels on a black background. The input should
    be an OpenCV BGR/BGRA image or a grayscale array.
    """
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")

    original = image.copy()
    gray = _as_grayscale(image)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    coordinates = cv2.findNonZero(binary)
    if coordinates is None:
        cropped = np.zeros((1, 1), dtype=np.uint8)
    else:
        x, y, width, height = cv2.boundingRect(coordinates)
        cropped = binary[y : y + height, x : x + width]

    height, width = cropped.shape
    scale = min(image_size / max(width, 1), image_size / max(height, 1))
    resized_width = max(1, min(image_size, int(round(width * scale))))
    resized_height = max(1, min(image_size, int(round(height * scale))))
    resized = cv2.resize(
        cropped, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    padded = np.zeros((image_size, image_size), dtype=np.uint8)
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    padded[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    normalized = padded.astype(np.float32) / 255.0
    steps = {
        "original": original,
        "grayscale": gray,
        "denoised": denoised,
        "binary (ink white)": binary,
        "cropped": cropped,
        "resized and padded": padded,
        "normalized": normalized,
    }
    return normalized, steps


def preprocess(image: np.ndarray) -> np.ndarray:
    """Return a centered 128x128 float32 image with values in [0, 1]."""
    processed, _ = preprocess_with_steps(image)
    return processed
