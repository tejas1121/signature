"""Evaluate saved models on a deterministic, held-out stratified test set."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

from config import MODELS_DIR, REPORTS_DIR, ensure_directories
from src.dataset import load_dataset, stratified_split
from src.features import extract_feature_matrix


def _write_model_evaluation(
    name: str,
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Save model metrics and a labeled confusion-matrix heatmap."""
    report = classification_report(
        y_true,
        y_prediction,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    print(f"\n{name} classification report:\n{report}")
    (REPORTS_DIR / f"{name}_classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    matrix = confusion_matrix(y_true, y_prediction, labels=labels)
    figure_size = max(8, min(18, len(class_names) * 0.45))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=len(class_names) <= 20,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(title=f"{name} confusion matrix", xlabel="Predicted identity", ylabel="True identity")
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(figure)

    row: dict[str, float | str] = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_prediction),
        "precision_macro": precision_score(
            y_true, y_prediction, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_prediction, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_prediction, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_prediction, average="weighted", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_prediction, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_prediction, average="weighted", zero_division=0
        ),
    }
    if probabilities is not None:
        row["top_3_accuracy"] = top_k_accuracy_score(
            y_true, probabilities, k=min(3, probabilities.shape[1]), labels=labels
        )
    return row


def main() -> None:
    """Evaluate each available model and persist reports, plots, and CSV."""
    ensure_directories()
    dataset = load_dataset()
    split = stratified_split(dataset.labels)
    y_test = dataset.labels[split.test]
    class_names = [str(value) for value in dataset.label_encoder.classes_]
    labels = np.arange(len(class_names))
    rows: list[dict[str, float | str]] = []

    svm_path = MODELS_DIR / "svm.joblib"
    forest_path = MODELS_DIR / "random_forest.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    if svm_path.exists() and forest_path.exists() and scaler_path.exists():
        features = extract_feature_matrix(dataset.images[split.test])
        scaler = joblib.load(scaler_path)
        x_test = scaler.transform(features)
        for name, path in (("svm", svm_path), ("random_forest", forest_path)):
            model = joblib.load(path)
            prediction = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
            rows.append(
                _write_model_evaluation(
                    name, y_test, prediction, labels, class_names, probabilities
                )
            )
    else:
        print("Classical models unavailable; run `python src/train_classical.py` first.")

    cnn_path = MODELS_DIR / "cnn.keras"
    if cnn_path.exists():
        cnn = tf.keras.models.load_model(cnn_path)
        probabilities = cnn.predict(dataset.images[split.test, ..., np.newaxis], verbose=0)
        prediction = np.argmax(probabilities, axis=1)
        rows.append(
            _write_model_evaluation(
                "cnn", y_test, prediction, labels, class_names, probabilities
            )
        )
        history_path = MODELS_DIR / "cnn_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            for metric, axis, title in (
                ("loss", axes[0], "Loss"),
                ("accuracy", axes[1], "Accuracy"),
            ):
                if metric in history:
                    axis.plot(history[metric], label=f"training {metric}")
                if f"val_{metric}" in history:
                    axis.plot(history[f"val_{metric}"], label=f"validation {metric}")
                axis.set(title=title, xlabel="Epoch", ylabel=title)
                axis.legend()
            figure.tight_layout()
            figure.savefig(REPORTS_DIR / "cnn_training_curves.png", dpi=160)
            plt.close(figure)
    else:
        print("CNN unavailable; run `python src/train_cnn.py` first.")

    if rows:
        comparison = pd.DataFrame(rows).sort_values("model")
        comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        print("\nHeld-out test comparison:\n")
        print(comparison.to_string(index=False))
    else:
        raise FileNotFoundError(
            "No trained models were found. Run the classical and/or CNN training script."
        )
    (REPORTS_DIR / "RESULTS_GUIDE.md").write_text(
        "# Reading the results\n\n"
        "`model_comparison.csv` reports held-out test metrics; macro scores weigh "
        "each identity equally, while weighted scores account for test-set support. "
        "The confusion-matrix rows are actual identities and columns are predictions. "
        "CNN top-3 accuracy is the share of examples whose identity appears in its "
        "three highest-probability classes. Diagonal counts are correct predictions.\n\n"
        "Similar writing styles, few training samples, scan quality or crop changes, "
        "large pen/pressure differences, and class imbalance can cause "
        "misclassification. Synthetic-demo scores do not estimate performance on "
        "real signatures.\n",
        encoding="utf-8",
    )
    print(f"\nSaved evaluation artifacts in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
