"""Streamlit interface for signature identity prediction and evaluation review."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import LOW_CONFIDENCE_THRESHOLD, MODELS_DIR, REPORTS_DIR
from src.predict import load_model_bundle, predict_image
from src.preprocess import preprocess_with_steps

st.set_page_config(page_title="Signature Identification", page_icon="✍️", layout="wide")


@st.cache_resource
def cached_model_bundle(model_type: str) -> dict:
    """Cache a loaded model for repeated Streamlit reruns."""
    return load_model_bundle(model_type)


def _read_upload(upload: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """Decode an uploaded image as a BGR array for OpenCV preprocessing."""
    encoded = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def main() -> None:
    """Render upload, prediction, preprocessing, and evaluation panels."""
    st.title("Signature Identification Using Machine Learning")
    st.write(
        "Upload a handwritten signature to identify the most likely person among "
        "the identities represented by the trained dataset."
    )
    with st.sidebar:
        st.header("Prediction settings")
        model_label = st.selectbox(
                      "Model",
                      ["RF", "SVM", "CNN"],
                      index=0
                      )
        model_type = {"CNN": "cnn", "SVM": "svm", "RF": "rf"}[model_label]
        st.caption("Predictions are closed-set: the model always ranks known identities.")
        st.divider()
        comparison_path = REPORTS_DIR / "model_comparison.csv"
        metrics_tab, confusion_tab = st.tabs(["Metrics", "Confusion matrix"])
        with metrics_tab:
            if comparison_path.exists():
                comparison = pd.read_csv(comparison_path)
                st.dataframe(comparison, hide_index=True, use_container_width=True)
            else:
                st.info("Run `python src/evaluate.py` to create model metrics.")
        with confusion_tab:
            matrix_path = REPORTS_DIR / f"{model_type}_confusion_matrix.png"
            if matrix_path.exists():
                st.image(
                    str(matrix_path),
                    caption=f"{model_label} test confusion matrix",
                )
            else:
                st.caption("The selected model's confusion matrix is not available yet.")

    uploaded = st.file_uploader(
        "Choose a signature image", type=["png", "jpg", "jpeg"]
    )
    if uploaded is None:
        st.info("Upload a PNG or JPEG image to begin.")
        return

    try:
        raw_image = _read_upload(uploaded)
        processed, steps = preprocess_with_steps(raw_image)
        original_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) if raw_image.ndim == 3 else raw_image
        image_columns = st.columns(2)
        image_columns[0].image(original_rgb, caption="Original image", use_container_width=True)
        image_columns[1].image(processed, caption="Preprocessed (ink on black)", clamp=True, use_container_width=True)
        with st.expander("Inspect preprocessing steps"):
            step_tabs = st.tabs(list(steps.keys()))
            for tab, (step_name, step_image) in zip(step_tabs, steps.items()):
                with tab:
                    if step_name == "original" and step_image.ndim == 3:
                        display_image = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
                    else:
                        display_image = step_image
                    tab.image(display_image, caption=step_name, clamp=True, use_container_width=True)

        try:
            bundle = cached_model_bundle(model_type)
            prediction = predict_image(raw_image, bundle)
        except FileNotFoundError as exc:
            st.warning(f"No trained {model_label} model is ready. {exc}")
            return
        st.subheader("Prediction")
        st.metric("Predicted identity", prediction["identity"])
        confidence = prediction["confidence"]
        st.metric("Confidence", f"{confidence:.1%}")
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            st.warning("Signature may not belong to any known person")
        top_predictions = prediction["top_predictions"]
        chart_data = pd.DataFrame(
            {
                "Identity": [item["identity"] for item in top_predictions],
                "Probability": [item["probability"] for item in top_predictions],
            }
        ).set_index("Identity")
        st.bar_chart(chart_data)
    except (ValueError, cv2.error, OSError) as exc:
        st.error(f"Could not process this image: {exc}")


if __name__ == "__main__":
    main()
