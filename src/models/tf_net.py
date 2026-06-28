"""Small TensorFlow/Keras 3 MLP for tabular fraud detection (CPU-friendly).

Mirrors torch_net.py for an apples-to-apples GBM-vs-PyTorch-vs-Keras benchmark.
Imbalance is handled with Keras `class_weight=` (the Keras-idiomatic approach).
`tensorflow` is optional — `is_available()` lets the benchmark skip this model.

Note (per blueprint): TF 2.21 bundles Keras 3 (`tf.keras`). Native-Windows GPU is
unsupported since TF 2.11 — use WSL2 or `tensorflow-cpu`. CPU is fine here.
"""
from __future__ import annotations

import numpy as np

from src.models.imbalance import class_weight_dict


def is_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def train_keras_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 12,
    batch_size: int = 2048,
    hidden: tuple[int, ...] = (64, 32),
    random_state: int = 42,
):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xs = (X - mu) / sigma

    layers = [keras.layers.Input(shape=(Xs.shape[1],))]
    for h in hidden:
        layers += [keras.layers.Dense(h, activation="relu"), keras.layers.Dropout(0.2)]
    layers += [keras.layers.Dense(1, activation="sigmoid")]
    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="auprc")],
    )
    model.fit(
        Xs, y,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict(y),
        verbose=0,
    )
    return _KerasWrapper(model, mu, sigma)


class _KerasWrapper:
    def __init__(self, model, mu, sigma):
        self.model = model
        self.mu = mu
        self.sigma = sigma
        self._creditlens_backend = "tensorflow-keras"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = ((np.asarray(X, dtype=np.float32) - self.mu) / self.sigma).astype(np.float32)
        p = self.model.predict(Xs, verbose=0).ravel()
        return np.column_stack([1 - p, p])
