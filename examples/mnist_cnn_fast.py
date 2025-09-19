"""
MNIST fast CNN example (minimal conv) for quick runs on CPU.

Model
  28x28 -> Conv(8, 3x3, valid) -> 26x26 -> Flatten -> Dense(10)

Usage
  python -c "import sys, os; sys.path.insert(0, os.getcwd()); import examples.mnist_cnn_fast as m; m.main()"

Env knobs
  EPOCHS (default 5), BATCH (default 512), MNIST_LIMIT (default 60000)
"""

from __future__ import annotations

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import forgeNN as fnn
from forgeNN.core.tensor import Tensor


def load_mnist(limit: int | None = None, seed: int = 42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = (mnist["data"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = mnist["target"].astype(np.int64)
    if limit is not None and limit < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=limit, random_state=seed, stratify=y)
    # Simple split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=10000, random_state=seed, stratify=y)
    return X_tr, y_tr, X_te, y_te


def build_model():
    return fnn.Sequential([
        fnn.Input((1, 28, 28)),
        fnn.Conv2D(cin=1, cout=8, kernel_size=(3, 3)) @ 'relu',
        fnn.Flatten(),
        fnn.Dense(10)
    ])


def main():
    limit = os.environ.get('MNIST_LIMIT', '')
    limit_int = int(limit) if (limit.strip().isdigit()) else 60000
    X_tr, y_tr, X_te, y_te = load_mnist(limit_int)

    model = build_model()
    opt = fnn.AdamW(lr=1e-3, weight_decay=1e-4)
    compiled = fnn.compile(model, optimizer=opt, loss='cross_entropy', metrics=['accuracy'])

    epochs = int(os.environ.get('EPOCHS', '5'))
    batch_size = int(os.environ.get('BATCH', '512'))

    # Warmup to JIT & warm BLAS
    n = min(64, len(X_tr))
    t = Tensor(X_tr[:n])
    l = compiled.loss_fn(model(t), y_tr[:n])
    compiled.optimizer.zero_grad(); l.backward(); compiled.optimizer.step()

    t0 = time.time()
    hist = compiled.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=1)
    total_s = time.time() - t0
    loss, metrics = compiled.evaluate(X_te, y_te, batch_size=1024)
    acc = metrics.get('accuracy', 0.0)
    print(f"Time: {total_s:.2f}s, Test acc: {acc*100:.2f}%")

    # Plot
    e = np.arange(1, len(hist['loss']) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(e, hist['loss'], label='loss')
    if 'accuracy' in hist:
        plt.plot(e, np.array(hist['accuracy']) * 100, label='acc (%)')
    plt.xlabel('Epoch'); plt.title('MNIST fast CNN'); plt.legend(); plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'artifacts_mnist_fast.png')
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()


