"""Load saved models and predict the identity of one signature image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from config import MODELS_DIR
from src.features import extract_features
from src.preprocess import preprocess


def load_model_bundle(model_type: str, models_dir: Path | str = MODELS_DIR) -> dict[str, Any]:
    """Load one model and the label encoder and, when needed, feature scaler."""
    normalized_type = model_type.lower()
    if normalized_type not in {"cnn", "svm", "rf"}:
        raise ValueError("model_type must be one of: cnn, svm, rf.")
    directory = Path(models_dir)
    encoder_path = directory / "label_encoder.joblib"
    model_path = {
        "cnn": directory / "cnn.keras",
        "svm": directory / "svm.joblib",
        "rf": directory / "random_forest.joblib",
    }[normalized_type]
    missing = [path for path in (encoder_path, model_path) if not path.exists()]
    if normalized_type in {"svm", "rf"}:
        scaler_path = directory / "scaler.joblib"
        if not scaler_path.exists():
            missing.append(scaler_path)
    if missing:
        raise FileNotFoundError(
            "Required model artifacts are missing: "
            + ", ".join(str(path) for path in missing)
            + ". Train the selected model first."
        )
    bundle: dict[str, Any] = {"label_encoder": joblib.load(encoder_path)}
    if normalized_type == "cnn":
        import tensorflow as tf

        bundle["model"] = tf.keras.models.load_model(model_path)
    else:
        bundle["model"] = joblib.load(model_path)
        bundle["scaler"] = joblib.load(directory / "scaler.joblib")
    bundle["model_type"] = normalized_type
    return bundle


def predict_image(
    image: np.ndarray,
    bundle: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Predict a signature identity and return the highest-scoring classes."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    processed = preprocess(image)
    model_type = bundle.get("model_type")
    if model_type == "cnn":
        probabilities = np.asarray(
            bundle["model"].predict(processed[np.newaxis, ..., np.newaxis], verbose=0)[0]
        )
    elif model_type in {"svm", "rf"}:
        vector = extract_features(processed).reshape(1, -1)
        scaled = bundle["scaler"].transform(vector)
        probabilities = np.asarray(bundle["model"].predict_proba(scaled)[0])
    else:
        raise ValueError("The model bundle has an unsupported model_type.")
    encoder = bundle["label_encoder"]
    count = min(top_k, len(probabilities))
    indices = np.argsort(probabilities)[::-1][:count]
    return {
        "identity": str(encoder.inverse_transform([int(indices[0])])[0]),
        "confidence": float(probabilities[indices[0]]),
        "top_predictions": [
            {
                "identity": str(encoder.inverse_transform([int(index)])[0]),
                "probability": float(probabilities[index]),
            }
            for index in indices
        ],
        "processed_image": processed,
    }


def predict_image_file(image_path: Path | str, bundle: dict[str, Any]) -> dict[str, Any]:
    """Read an image from disk and return its predicted identity."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return predict_image(image, bundle)
