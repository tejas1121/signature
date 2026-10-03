"""Shared project configuration and reproducibility settings."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SIGNATURE_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGE_SIZE = 128
RANDOM_SEED = 42
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 32
CNN_EPOCHS = 35
CNN_LEARNING_RATE = 3e-4
CLASSICAL_CV_FOLDS = 3
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
LBP_POINTS = 24
LBP_RADIUS = 3
LOW_CONFIDENCE_THRESHOLD = 0.50


def ensure_directories() -> None:
    """Create the output directories used by training and evaluation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
