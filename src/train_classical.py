"""Train and compare SVM and Random Forest classical classifiers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CLASSICAL_CV_FOLDS, MODELS_DIR, RANDOM_SEED, ensure_directories
from src.dataset import load_dataset, stratified_split
from src.features import extract_feature_matrix


def main() -> None:
    """Train both classical models and persist the best SVM and the forest."""
    ensure_directories()
    dataset = load_dataset()
    split = stratified_split(dataset.labels)
    print("Extracting HOG, geometry, Hu-moment, and LBP features...")
    features = extract_feature_matrix(dataset.images)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[split.train])
    x_validation = scaler.transform(features[split.validation])
    y_train = dataset.labels[split.train]
    y_validation = dataset.labels[split.validation]

    minimum_train_count = int(np.bincount(y_train).min())
    folds = min(CLASSICAL_CV_FOLDS, minimum_train_count)
    if folds < 2:
        raise ValueError("Not enough training samples per identity for SVM CV.")
    print(f"Searching SVM hyperparameters with stratified {folds}-fold CV...")
    search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True),
        param_grid={"C": [1, 10], "gamma": ["scale", 0.001, 0.01]},
        scoring="f1_macro",
        cv=folds,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)

    forest = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    print("Training Random Forest...")
    forest.fit(x_train, y_train)

    comparison = []
    for name, model in (("svm", search.best_estimator_), ("random_forest", forest)):
        prediction = model.predict(x_validation)
        accuracy = accuracy_score(y_validation, prediction)
        macro_f1 = f1_score(y_validation, prediction, average="macro", zero_division=0)
        comparison.append(
            {"model": name, "validation_accuracy": accuracy, "validation_macro_f1": macro_f1}
        )
        print(
            f"{name}: validation accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}"
        )

    joblib.dump(search.best_estimator_, MODELS_DIR / "svm.joblib")
    joblib.dump(forest, MODELS_DIR / "random_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(dataset.label_encoder, MODELS_DIR / "label_encoder.joblib")
    pd.DataFrame(comparison).to_csv(MODELS_DIR / "classical_validation.csv", index=False)
    print(f"Best SVM parameters: {search.best_params_}")
    print(f"Saved SVM, Random Forest, scaler, and label encoder to {MODELS_DIR}")


if __name__ == "__main__":
    main()
