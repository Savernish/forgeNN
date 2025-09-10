"""
Deprecated (v2) compatibility shim for forgeNN.vectorized
=========================================================

This module was removed in v2. Please migrate to the v2 API:
 - Build models with `forgeNN.Sequential([Input(...), Dense(...) @ 'relu', ...])`
 - Train with `forgeNN.compile(model, optimizer=..., loss=..., metrics=[...]).fit(...)`

For a transitional period, we keep the following minimal exports:
 - cross_entropy_loss, mse (from forgeNN.nn.losses)
 - accuracy (from forgeNN.nn.metrics)
 - VectorizedOptimizer (alias of forgeNN.optimizers.SGD)

Attempting to construct VectorizedMLP now raises with guidance.
"""

from __future__ import annotations

from .nn.losses import cross_entropy_loss, mse
from .nn.metrics import accuracy
from .optimizers import SGD as VectorizedOptimizer


class VectorizedMLP:  # pragma: no cover - deprecation path
  def __init__(self, *args, **kwargs):
    raise ImportError(
      "forgeNN.vectorized.VectorizedMLP was removed in v2. "
      "Use forgeNN.Sequential with Dense/Flatten and activations (e.g., Dense(64) @ 'relu')."
    )


__all__ = [
  'cross_entropy_loss',
  'mse',
  'accuracy',
  'VectorizedOptimizer',
  'VectorizedMLP',
]

"""forgeNN.vectorized has been removed in v2.

Please migrate to the v2 API:
  - Build models with forgeNN.Sequential([...]) and layers like Dense/Flatten/Input
  - Attach activations via the @ operator, e.g., Dense(64) @ 'relu'
  - Train via the Keras-like workflow: compiled = forgeNN.compile(model, ...); compiled.fit(...)

See README Quick Start for examples. This module intentionally raises at import.
"""

raise ImportError(
    "forgeNN.vectorized was removed in v2. Use Sequential + compile/fit instead. "
    "See README Quick Start for migration examples."
)
