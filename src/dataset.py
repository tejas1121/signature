"""Dataset discovery, preprocessing, label encoding, and stratified splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_DIR,
    RANDOM_SEED,
    TEST_FRACTION,
    VALIDATION_FRACTION,
)
from src.preprocess import preprocess

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FORGERY_MARKERS = ("forg", "negative")


@dataclass
class SignatureDataset:
    """Loaded images and encoded identities, with paths retained for auditing."""

    images: np.ndarray
    labels: np.ndarray
    label_encoder: LabelEncoder
    paths: list[Path]


@dataclass
class DatasetSplit:
    """Indices for deterministic train, validation, and held-out test subsets."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _is_forgery_path(relative_path: Path) -> bool:
    """Identify common forged/negative folder names and Kaggle suffixes."""
    for part in relative_path.parts[:-1]:
        lowered = part.lower()
        if any(marker in lowered for marker in FORGERY_MARKERS):
            return True
    return False


def _identity_for_path(relative_path: Path) -> str:
    """Use the first person directory as the class, excluding genuine subfolders."""
    parts = relative_path.parts[:-1]
    if not parts:
        raise ValueError(f"Image is not inside an identity folder: {relative_path}")
    identity = parts[0]
    if identity.lower() in {"genuine", "real", "train", "test", "images"}:
        if len(parts) < 2:
            raise ValueError(f"Could not infer identity from {relative_path}.")
        identity = parts[1]
    if identity.lower().endswith(("_forg", "_forged")):
        identity = identity.rsplit("_", 1)[0]
    return identity


def load_dataset(data_dir: Path | str = DATA_DIR) -> SignatureDataset:
    """Load `person/image` folders, retaining only genuine signatures.

    Forgery folders named `forg`, `forged`, `forgery`, `forgeries`,
    `negative`, or with `_forg`/`_forged` suffixes are excluded.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Run `python make_demo_data.py` "
            "or set SIGNATURE_DATA_DIR to a folder-per-person dataset."
        )
    paths: list[Path] = []
    labels: list[str] = []
    processed_images: list[np.ndarray] = []
    failures: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(root)
        if _is_forgery_path(relative_path):
            continue
        try:
            label = _identity_for_path(relative_path)
            raw_image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw_image is None:
                raise ValueError("OpenCV could not decode the file.")
            processed_images.append(preprocess(raw_image))
            paths.append(path)
            labels.append(label)
        except (ValueError, cv2.error) as exc:
            failures.append(f"{path}: {exc}")
    if not processed_images:
        raise ValueError(
            f"No readable genuine signature images were found under {root}."
        )
    if failures:
        print(f"Warning: skipped {len(failures)} unreadable/invalid image(s).")
        for message in failures[:5]:
            print(f"  {message}")
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    counts = np.bincount(encoded_labels)
    if len(encoder.classes_) < 2:
        raise ValueError("At least two identity folders are required.")
    if int(counts.min()) < 3:
        sparse = {
            str(encoder.classes_[index]): int(count)
            for index, count in enumerate(counts)
            if count < 3
        }
        raise ValueError(
            "Each identity needs at least 3 genuine images for a stratified "
            f"train/validation/test split. Underrepresented identities: {sparse}"
        )
    print(
        f"Loaded {len(processed_images)} genuine images from "
        f"{len(encoder.classes_)} identities; excluded forgeries by folder name."
    )
    return SignatureDataset(
        images=np.stack(processed_images),
        labels=encoded_labels.astype(np.int64),
        label_encoder=encoder,
        paths=paths,
    )


def stratified_split(
    labels: np.ndarray,
    seed: int = RANDOM_SEED,
    test_fraction: float = TEST_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> DatasetSplit:
    """Create a deterministic stratified 70/15/15-style split."""
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Test and validation fractions must be between 0 and 1.")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("Test and validation fractions must sum to less than 1.")
    number_of_classes = len(np.unique(labels))
    test_size = int(np.ceil(len(labels) * test_fraction))
    validation_size = int(
        np.ceil(len(labels) * validation_fraction / (1.0 - test_fraction))
    )
    if test_size < number_of_classes or validation_size < number_of_classes:
        raise ValueError(
            "The dataset is too small for every identity to appear in both "
            "validation and test sets. Add more samples per identity."
        )
    indices = np.arange(len(labels))
    train_val, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=labels
    )
    relative_val_fraction = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val_fraction,
        random_state=seed,
        stratify=labels[train_val],
    )
    return DatasetSplit(train=train, validation=validation, test=test)
