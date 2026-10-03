"""Train a compact augmented TensorFlow/Keras signature classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    IMAGE_SIZE,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.dataset import load_dataset, stratified_split


def build_model(number_of_classes: int) -> tf.keras.Model:
    """Create a compact CNN that preserves spatial stroke information."""
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = inputs
    for filters, dropout_rate in ((32, 0.05), (64, 0.10), (128, 0.15)):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)   # faster-settling stats
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout_rate)(x)
    # 16x16x128 -> 4x4x128 = 2048 features (instead of 32768)
    x = layers.AveragePooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(number_of_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="signature_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(
    epochs: int = CNN_EPOCHS,
    output_dir: Path | str = MODELS_DIR,
) -> None:
    """Train, checkpoint by validation loss, and report held-out CNN metrics."""
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    dataset = load_dataset()
    split = stratified_split(dataset.labels)
    images = dataset.images[..., np.newaxis].astype("float32")
    print("Image range:", images.min(), images.max(), "| shape:", images.shape)

    x_train, y_train = images[split.train], dataset.labels[split.train]
    x_val, y_val = images[split.validation], dataset.labels[split.validation]
    x_test, y_test = images[split.test], dataset.labels[split.test]

    class_counts = np.bincount(dataset.labels)
    print(
        f"Identities={len(dataset.label_encoder.classes_)}, "
        f"samples/identity min/median/max="
        f"{class_counts.min()}/{np.median(class_counts):.0f}/{class_counts.max()}, "
        f"chance top-1={1 / len(class_counts):.2%}"
    )
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_model(len(dataset.label_encoder.classes_))
    augmentation = ImageDataGenerator(
        rotation_range=4,
        shear_range=0.05,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.06,
        horizontal_flip=False,
        fill_mode="constant",
        cval=0.0,
    )
    checkpoint_path = str(model_dir / "cnn.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-5, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min",
                        save_best_only=True, verbose=1),
    ]
    steps = max(1, int(np.ceil(len(x_train) / BATCH_SIZE)))
    print(f"Training on {len(x_train)} images, val={len(x_val)}, test={len(x_test)}, "
          f"steps/epoch={steps}")

    history = model.fit(
        augmentation.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED),
        validation_data=(x_val, y_val),
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    joblib.dump(dataset.label_encoder, model_dir / "label_encoder.joblib")
    pd.DataFrame(history.history).to_csv(model_dir / "cnn_history.csv", index=False)

    best = tf.keras.models.load_model(checkpoint_path)
    probabilities = best.predict(x_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    top_k = min(3, probabilities.shape[1])
    top_k_accuracy = top_k_accuracy_score(
        y_test,
        probabilities,
        k=top_k,
        labels=np.arange(probabilities.shape[1]),
    )
    print(
        f"Held-out test accuracy={accuracy_score(y_test, predictions):.4f}; "
        f"top-{top_k} accuracy={top_k_accuracy:.4f}; "
        f"predicted identities={len(np.unique(predictions))}/"
        f"{probabilities.shape[1]}"
    )
    print(f"Saved CNN checkpoint and history to {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    arguments = parser.parse_args()
    main(epochs=arguments.epochs, output_dir=arguments.output_dir)