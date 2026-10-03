from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from config import MODELS_DIR
from src.preprocess import preprocess
from src.features import extract_features


# Kaggle official test directory
TEST_DIR = Path("data/kaggle-download/sign_data/sign_data/test")

# Load trained artifacts
model = joblib.load(MODELS_DIR / "random_forest.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")


y_true = []
y_pred = []
paths = []

print("Evaluating Random Forest on Kaggle test set...")
print(f"Test directory: {TEST_DIR}")

for identity_dir in sorted(TEST_DIR.iterdir()):

    # Ignore forged folders such as 049_forg
    if not identity_dir.is_dir():
        continue

    if "forg" in identity_dir.name.lower():
        continue

    identity = identity_dir.name

    for image_path in sorted(identity_dir.iterdir()):

        if image_path.suffix.lower() not in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        }:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        try:
            processed = preprocess(image)

            features = extract_features(processed).reshape(1, -1)

            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]

            predicted_identity = encoder.inverse_transform(
                [int(prediction)]
            )[0]

            y_true.append(identity)
            y_pred.append(str(predicted_identity))
            paths.append(image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


# Results
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print("KAGGLE TEST SET RESULTS")
print("=" * 60)

print(f"Total genuine test images : {len(y_true)}")
print(f"Correct predictions       : {sum(a == b for a, b in zip(y_true, y_pred))}")
print(f"Incorrect predictions     : {sum(a != b for a, b in zip(y_true, y_pred))}")
print(f"Accuracy                  : {accuracy:.4%}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

# Show incorrect predictions
print("\n" + "=" * 60)
print("INCORRECT PREDICTIONS")
print("=" * 60)

errors = 0

for path, actual, predicted in zip(paths, y_true, y_pred):
    if actual != predicted:
        print(
            f"{path.name}: actual={actual}, predicted={predicted}"
        )
        errors += 1

        if errors >= 20:
            print("... showing first 20 errors only")
            break