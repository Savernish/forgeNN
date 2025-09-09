"""
forgeNN MNIST Classification with Sequential and compile/fit
===========================================================

End-to-end MNIST training using the Sequential API with @ activation wiring,
driven by a lightweight Keras-like trainer (compile/fit/evaluate/predict).

Features:
- MNIST handwritten digit classification
- Sequential model definition with Dense @ 'relu'
- compile(loss='cross_entropy', metrics=['accuracy'])
- fit(..., validation_data=(X_test, y_test)) and evaluate/predict
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import forgeNN as fnn


def load_mnist(n_samples: int = 5000):
    """Load and preprocess MNIST for Sequential training.

    Returns:
        X_train, X_test, y_train, y_test (all numpy arrays)
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    # Subsample for a quick demo
    print(f"Using {n_samples} samples for training...")
    X, y = X[:n_samples], y[:n_samples]

    # Normalize to [0,1]
    X /= 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    return X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("MNIST CLASSIFICATION WITH forgeNN Sequential")
    print("=" * 60)

    # Data
    X_train, X_test, y_train, y_test = load_mnist(n_samples=5000)

    # Model (lazy-init Dense; run a dummy forward before compiling)
    print("\nBuilding Sequential model...")
    model = fnn.Sequential([
        fnn.Input((784,)),
        fnn.Dense(128) @ 'relu',
        fnn.Dense(64) @ 'relu',
        fnn.Dense(10) @ 'linear',  # logits
    ])

    # # Initialize lazily constructed params for all layers
    # _ = model(fnn.Tensor(np.zeros((1, 784), dtype=np.float32)))
    model.summary()
    time.sleep(4)

    # Compile with built-in loss/metric and optimizer config
    compiled = fnn.compile(
        model,
        optimizer={"lr": 0.005, "momentum": 0.9},
        loss="cross_entropy",
        metrics=["accuracy"],
    )

    # Training config
    epochs = 10
    batch_size = 64
    print("\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {0.01}")
    print(f"  Momentum: {0.9}")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start = time.time()
    compiled.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_test, y_test), verbose=1)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_loss, final_metrics = compiled.evaluate(X_test, y_test, batch_size)
    final_acc = final_metrics.get("accuracy", 0.0)
    print(f"Final Test Accuracy: {final_acc*100:.2f}%")
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Training Time: {elapsed:.1f} seconds")
    print(f"Samples per Second: {len(X_train) * epochs / elapsed:.0f}")

    # Show sample predictions
    print("\nSample Predictions:")
    sample_idx = np.random.choice(len(X_test), 5, replace=False)
    for i, idx in enumerate(sample_idx, start=1):
        logits = compiled.predict(X_test[idx:idx+1])
        probs = fnn.Tensor(logits, requires_grad=False).softmax(axis=1).data[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        print(f"  Sample {i}: True = {y_test[idx]}, Predicted = {pred}, Confidence = {conf:.3f}")


if __name__ == "__main__":
    main()
