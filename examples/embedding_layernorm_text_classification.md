# Embedding + GlobalAvgPool1D + LayerNorm Text Classification

This example demonstrates a minimal text classifier built with the new layers:
- Embedding: maps token indices to dense vectors
- GlobalAvgPool1D: averages across sequence length
- LayerNorm: normalizes per-sample across features
- Dropout: regularization
- Dense: linear classifier head

It follows a fastText-style architecture and trains on a synthetic dataset so it runs out-of-the-box.

---

## What this example shows

- How to use the `Embedding` layer with integer token indices.
- How to reshape for `GlobalAvgPool1D` (expecting N, C, L) by transposing from (N, L, D) to (N, D, L).
- Where `LayerNorm` fits in the pipeline and why it’s shape-preserving.
- How to train using the Keras-like `compile(...).fit(...)` workflow.
- How to validate and make predictions with your model.

---

## File layout

- `examples/embedding_layernorm_text_classification.py` — the runnable script.
- `examples/embedding_layernorm_text_classification.md` — this README.

---

## Prerequisites

- Python 3.12 (or 3.10+ should work)
- NumPy (installed via the project’s requirements)
- Local `forgeNN` import (run from repo root so `import forgeNN as fnn` resolves)

---

## How to run

From the repository root:

```powershell
# Windows PowerShell
python examples/embedding_layernorm_text_classification.py
```

The script trains for a few epochs and prints training/validation metrics per epoch, then a final validation score and a few test predictions.

Example output (your numbers may vary):

```
Epoch 1/8, loss=0.51, accuracy=83.4%  val_loss=0.33, val_accuracy=86.3%
...
Epoch 8/8, loss=0.01, accuracy=99.6%  val_loss=0.02, val_accuracy=99.3%
Validation: loss=0.0189, accuracy=99.33%
Test preds: [1, 1, 1]
```

---

## Dataset generation (synthetic)

We generate `n_samples` sequences of integer token IDs with a fixed `seq_len` and `vocab_size`.

- Each sample is labeled as class 1 if it contains any token from a small set of “trigger” IDs; otherwise class 0.
- This creates an easy but non-trivial task for a model that averages embeddings.

Key parameters:
- `n_samples`: number of sequences
- `seq_len`: tokens per sequence
- `vocab_size`: size of the token vocabulary
- `triggers`: token IDs that flip the label to 1 when present

Train/validation split is a simple shuffle + slice (`val_ratio`, e.g., 0.2).

---

## Model architecture

We implement a small classifier as a plain Python class exposing `__call__` and `parameters()` to work with `forgeNN.training.compile`.

Pipeline (with default shapes):

1) Embedding
- Input: `(N, L)` — batch of token indices
- Output: `(N, L, D)` — per-token embedding vectors

2) Transpose for pooling
- `transpose(0, 2, 1)` → `(N, D, L)` to match `GlobalAvgPool1D`’s expected `(N, C, L)` layout

3) GlobalAvgPool1D
- Reduces over the sequence dimension `L`
- Output: `(N, D)` when `keepdims=False` (default)

4) LayerNorm
- Normalizes across the last dimension per-sample
- Shape-preserving: `(N, D)` → `(N, D)`

5) Dropout
- Regularization during training; no change in shape

6) Dense (linear head)
- Input: `(N, D)`
- Output: `(N, C)` where `C` is number of classes (2 in this example)

Notes:
- We specify `Dense(num_classes, in_features=embed_dim)` so weights initialize immediately, ensuring the optimizer binds params without needing an initial forward.
- `LayerNorm(embed_dim)` lazily initializes if you omit the dimension.

---

## Training loop (compile/fit)

We use the high-level training API:

```python
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
)
```

- Optimizer: Adam with a learning rate of 0.01 (tunable).
- Loss: Cross-entropy over integer labels.
- Metric: Accuracy is aggregated across batches for an exact dataset accuracy.

`CompiledModel` automatically:
- Batches the data, shuffles per epoch if requested.
- Runs forward, computes loss, backpropagates, and steps the optimizer.
- Aggregates sample-weighted loss and metrics for consistent reporting.

---

## Expected results

- Because the labeling rule is simple, the model typically reaches > 98% validation accuracy within a few epochs.
- Final printed predictions on a few hand-crafted sequences should align with whether a trigger token is present.

If you see unstable training:
- Try reducing the learning rate (e.g., `lr=0.005` or `0.001`).
- Increase `embed_dim` a bit (e.g., 128).
- Increase training epochs.

---

## Troubleshooting

- ImportError: Make sure you run from the project root so `import forgeNN as fnn` resolves to the local package.
- dtype issues: Inputs to `Embedding` must be integer indices. The training loop wraps arrays into `Tensor` automatically.
- Shape mismatch in pooling: Ensure the transpose before `GlobalAvgPool1D` so your input is `(N, D, L)`.
- If you edited layer stubs (e.g., Conv/Pool/BatchNorm), avoid raising errors at import time; only raise in `__init__`/`forward` to keep the package importable.

---

## Extending this example

- Positional information: Add/learn positional embeddings and sum with token embeddings before pooling.
- Other pooling: Replace global avg with attention pooling or max pooling.
- Deeper head: Add a small MLP (e.g., `Dense(4*D) @ 'gelu'` → Dropout → `Dense(C)`).
- Regularization: Increase Dropout or add `LayerNorm` in more places.
- Multi-class: Increase `num_classes` and adjust the labeling rule accordingly.

---

## ONNX export (optional)

This architecture can be exported in stages:
- `Embedding` → `Gather`
- `GlobalAvgPool1D` → `ReduceMean` over the sequence axis
- `LayerNorm` → primitives: `ReduceMean`, `Sub`, `Pow`, `ReduceMean`, `Add`, `Sqrt`, `Div`, then affine `Mul`/`Add`
- `Dense` → `Gemm`

If you plan to export, prefer `keepdims=True` where helpful and ensure ops are mapped in your exporter.

---

## Reference: key APIs used

- `fnn.Embedding(vocab_size, embed_dim, padding_idx=None)`
- `Tensor.transpose(*axes)` to rearrange NLD → NDL
- `fnn.GlobalAvgPool1D(keepdims=False)`
- `fnn.LayerNorm(normalized_shape=embed_dim, eps=1e-5)`
- `fnn.Dropout(rate=0.1)`
- `fnn.Dense(out_features, in_features=embed_dim)`
- `fnn.compile(model, optimizer={...}, loss='cross_entropy', metrics=['accuracy'])`
- `CompiledModel.fit(...), .evaluate(...), .predict(...)`
