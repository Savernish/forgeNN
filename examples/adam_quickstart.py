"""Quickstart: Train a tiny model with Adam (direct optimizer instance)
=======================================================================

Demonstrates:
    * Building a small network via Sequential + activation wrapper (@ syntax)
    * Using the new deferred-parameter Adam: create it BEFORE the model has params
    * Compiling with compile(model, optimizer=opt, ...)
    * Fitting / evaluating on a synthetic 2‑class dataset

Run:
        python examples/adam_quickstart.py

Key points:
    - You can now do ``opt = fnn.Adam(lr=1e-3)`` (no params yet) and pass it to ``compile``.
        Parameters are bound lazily on first access.
    - Alternative (config dict) still works:
                compiled = fnn.compile(model, optimizer={"type": "adam", "lr": 1e-3}, ...)
    - Final layer has no activation (raw logits) for cross-entropy.
    - Dataset is synthetic (Gaussian blobs) so it converges quickly.
"""


from __future__ import annotations

# Running from source tree: if forgeNN not installed, ensure parent on sys.path (optional)
import sys, os  # noqa
if 'forgeNN' not in sys.modules:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import forgeNN as fnn

# ---------------------------------------------------------------------------
# 1. Create a tiny synthetic classification dataset (two 2D blobs)
# ---------------------------------------------------------------------------

def make_blobs(n_per_class: int = 256, spread: float = 0.6, seed: int = 42):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=(-1.5, -1.0), scale=spread, size=(n_per_class, 2))
    c1 = rng.normal(loc=(1.2, 1.3), scale=spread, size=(n_per_class, 2))
    X = np.vstack([c0, c1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)])
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

X, y = make_blobs()

# Simple train/val split (80/20)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# ---------------------------------------------------------------------------
# 2. Define a Sequential model
#    - Input shape (2,) is implicit; first Dense will infer on first forward.
#    - Hidden layer with ReLU, output layer (2 logits for 2 classes)
# ---------------------------------------------------------------------------
model = fnn.Sequential([
    fnn.Dense(32) @ 'relu',
    fnn.Dense(2),  # logits layer (no activation)
])

# Optional: show a symbolic summary (provide input shape since no Input layer used)
print("Model summary:\n")
model.summary(input_shape=(2,))
print()

# ---------------------------------------------------------------------------
# 3. Create optimizer (deferred param binding) & compile
#    We purposely do NOT pass model.parameters() here; they bind lazily inside compile.
# ---------------------------------------------------------------------------
opt = fnn.Adam(lr=1e-3)  # You could also tweak betas=(0.9, 0.999)

compiled = fnn.compile(
    model,
    optimizer=opt,
    loss="cross_entropy",
    metrics=["accuracy"],
)

# (Alternative dict form kept for reference)
# compiled = fnn.compile(model, optimizer={"type": "adam", "lr": 1e-3}, loss="cross_entropy", metrics=["accuracy"]) 

# ---------------------------------------------------------------------------
# 4. Train (a few epochs is enough for linearly separable blobs)
# ---------------------------------------------------------------------------
compiled.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# ---------------------------------------------------------------------------
# 5. Final evaluation
# ---------------------------------------------------------------------------
loss, metrics = compiled.evaluate(X_val, y_val)
print(f"\nValidation loss: {loss:.4f}")
print(f"Validation accuracy: {metrics['accuracy']*100:.2f}%")

# ---------------------------------------------------------------------------
# 6. Predict probabilities (optional)
# ---------------------------------------------------------------------------
logits = compiled.predict(X_val[:5])  # shape (5,2)
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)
print("\nSample probs (first 5):\n", probs)

if __name__ == "__main__":
    # Already executed when run as a script (above). This guard keeps structure clear.
    pass
