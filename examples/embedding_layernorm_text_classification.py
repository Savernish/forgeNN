"""
Minimal text classification example using Embedding + GlobalAvgPool1D + LayerNorm + Dense.

This is a fastText-style classifier:
  - token indices -> Embedding (N, L, D)
  - transpose to N, D, L -> GlobalAvgPool1D over L -> (N, D)
  - LayerNorm, optional Dropout -> Dense to classes

It demonstrates the newly added layers without relying on convs.
"""
from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import forgeNN as fnn

# ----------------------
# Synthetic dataset
# ----------------------
# Rule: class 1 if any of the "trigger" tokens appear, else class 0
# This is easily learnable by averaging embeddings and a linear head.

def make_dataset(n_samples: int = 2000, seq_len: int = 16, vocab_size: int = 200,
                 triggers=(1, 2, 3), seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.integers(low=0, high=vocab_size, size=(n_samples, seq_len), dtype=np.int64)
    y = np.zeros(n_samples, dtype=np.int64)
    # Mark positive if any trigger token appears in the sequence
    trigger_set = set(triggers)
    for i in range(n_samples):
        if any(t in trigger_set for t in X[i]):
            y[i] = 1
    return X, y


def train_val_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_val = int(len(X) * val_ratio)
    return (X[n_val:], y[n_val:]), (X[:n_val], y[:n_val])


# ----------------------
# Model
# ----------------------
class AvgEmbClassifier:
    """Embedding -> Transpose to NDL -> GlobalAvgPool1D -> LayerNorm -> Dropout -> Dense."""
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, dropout: float = 0.1):
        self.emb = fnn.Embedding(vocab_size, embed_dim)
        self.gap = fnn.GlobalAvgPool1D(keepdims=False)
        self.ln = fnn.LayerNorm(embed_dim)
        self.drop = fnn.Dropout(dropout)
        # Specify in_features to avoid lazy init issues with optimizer param capture
        self.head = fnn.Dense(num_classes, in_features=embed_dim)

    def __call__(self, x: fnn.Tensor) -> fnn.Tensor:
        # x: (N, L) integer indices (wrapped as Tensor by the training loop)
        h = self.emb(x)                 # (N, L, D)
        h = h.transpose(0, 2, 1)        # (N, D, L) for GlobalAvgPool1D
        h = self.gap(h)                 # (N, D)
        h = self.ln(h)                  # (N, D)
        h = self.drop(h)                # (N, D)
        logits = self.head(h)           # (N, C)
        return logits

    def parameters(self):
        # Collect params from sub-layers
        params = []
        params.extend(self.emb.parameters())
        params.extend(self.ln.parameters())
        params.extend(self.head.parameters())
        return params


def main():
    # Data
    X, y = make_dataset(n_samples=3000, seq_len=24, vocab_size=500, triggers=(7, 13, 29))
    (X_train, y_train), (X_val, y_val) = train_val_split(X, y, val_ratio=0.2, seed=1)

    # Model
    model = AvgEmbClassifier(vocab_size=500, embed_dim=64, num_classes=2, dropout=0.1)

    # Compile & train
    compiled = fnn.compile(
        model,
        optimizer={"type": "adam", "lr": 0.01},
        loss="cross_entropy",
        metrics=["accuracy"],
    )

    compiled.fit(
        X_train, y_train,
        epochs=8,
        batch_size=64,
        shuffle=True,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    # Evaluate
    loss, metrics = compiled.evaluate(X_val, y_val, batch_size=128)
    print(f"Validation: loss={loss:.4f}, accuracy={metrics['accuracy']*100:.2f}%")

    # Quick sanity predictions
    test_seqs = np.array([
        [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # should be class 1 (trigger 7)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # likely class 0
        [13, 3, 1, 2, 29, 4, 5, 6, 7, 8, 9, 10],   # class 1
    ], dtype=np.int64)
    logits = model(fnn.Tensor(test_seqs, requires_grad=False))
    preds = np.argmax(logits.data, axis=1)
    print("Test preds:", preds.tolist())


if __name__ == "__main__":
    main()
