"""
MNIST CNN example using forgeNN with history plots and timing.

Workflow
- Downloads MNIST from OpenML on first run (cached by sklearn)
- Builds a small ConvNet (valid conv):
  28x28 -> Conv(32,3x3) -> 26x26 -> Pool2x2 -> 13x13
        -> Conv(64,3x3) -> 11x11 -> Pool2x2 -> 5x5
        -> Flatten -> Dense(70) -> Dense(10)
- Trains with compile/fit, collects history, and saves plots

Usage
  python -c "import sys, os; sys.path.insert(0, os.getcwd()); import examples.mnist_cnn as m; m.main()"
Environment knobs
  EPOCHS (default 10), BATCH (default 128), MNIST_LIMIT (default 60000)
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import forgeNN as fnn
from forgeNN.core.tensor import Tensor


def load_mnist(limit: int | None = None, seed: int = 42) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # Fetches 70k (28x28) as float64 0..255; convert to float32 0..1
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = (mnist["data"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = mnist["target"].astype(np.int64)

    if limit is not None and limit < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=limit, random_state=seed, stratify=y)

    # Train/Test split (hold out 10k for test if possible)
    test_size = min(10000, int(len(X) * 0.15))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model() -> fnn.Sequential:
    return fnn.Sequential([
        fnn.Input((1, 28, 28)),
        fnn.Conv2D(cin=1, cout=32, kernel_size=(3, 3)) @ 'relu',
        fnn.MaxPool2D(kernel_size=(2, 2)),
        fnn.Conv2D(cin=32, cout=64, kernel_size=(3, 3)) @ 'relu',
        fnn.MaxPool2D(kernel_size=(2, 2)),
        fnn.Flatten(),
        fnn.Dense(70) @ 'relu',
        fnn.Dense(10)
    ])


def plot_history(history, epoch_times, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history['loss']) + 1)
    # Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='train')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, np.array(history.get('accuracy', [])) * 100, label='train')
    if 'val_accuracy' in history:
        plt.plot(epochs, np.array(history['val_accuracy']) * 100, label='val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, epoch_times, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Seconds'); plt.title('Epoch Time (s)')
    plt.tight_layout()
    path = os.path.join(out_dir, 'mnist_training_summary.png')
    plt.savefig(path)
    print(f"Saved curves to {path}")


def show_confusion(compiled, X_test, y_test, out_dir: str):
    logits = compiled.predict(X_test, batch_size=512)
    y_pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    path = os.path.join(out_dir, 'mnist_confusion_matrix.png')
    plt.tight_layout(); plt.savefig(path)
    print(f"Saved confusion matrix to {path}")


def main():
    limit = os.environ.get('MNIST_LIMIT', '')
    limit_int = int(limit) if (limit.strip().isdigit()) else 60000
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(limit=limit_int)

    model = build_model()
    opt = fnn.AdamW(lr=1e-3, weight_decay=1e-4)
    compiled = fnn.compile(model, optimizer=opt, loss='cross_entropy', metrics=['accuracy'])

    epochs = int(os.environ.get('EPOCHS', '10'))
    batch_size = int(os.environ.get('BATCH', '256'))
    val_every = int(os.environ.get('VAL_EVERY', '0'))  # 0 = no val during training

    # Warmup (JIT compile numba kernels & BLAS path) on a tiny batch
    warm_n = min(64, len(X_train))
    bx = Tensor(X_train[:warm_n])
    by = y_train[:warm_n]
    logits = model(bx)
    loss = compiled.loss_fn(logits, by)
    compiled.optimizer.zero_grad(); loss.backward(); compiled.optimizer.step()

    # Train one epoch at a time to collect per-epoch timing while using history
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    epoch_times = []
    for e in range(1, epochs + 1):
        t0 = time.time()
        vdata = (X_val, y_val) if (val_every and (e % val_every == 0)) else None
        h = compiled.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=vdata, verbose=0)
        epoch_times.append(time.time() - t0)
        for k in history:
            if k in h:
                history[k].extend(h[k])
    print(f"Total training time: {sum(epoch_times):.2f}s  (avg {np.mean(epoch_times):.2f}s/epoch)")

    test_loss, test_metrics = compiled.evaluate(X_test, y_test, batch_size=512)
    print(f"Test: loss={test_loss:.4f}, acc={test_metrics.get('accuracy', 0.0)*100:.2f}%")

    out_dir = os.path.join(os.path.dirname(__file__), 'artifacts_mnist_cnn')
    plot_history(history, epoch_times, out_dir)
    show_confusion(compiled, X_test, y_test, out_dir)


if __name__ == '__main__':
    main()


