"""
Metrics for forgeNN v2

Docstrings kept consistent with previous formatting.
"""
from __future__ import annotations
import numpy as np
from ..core.tensor import Tensor


def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits (Tensor): Model predictions (batch_size, num_classes)
        targets (np.ndarray): True class indices (batch_size,)
        
    Returns:
        float: Accuracy as fraction between 0 and 1

    Example:
        >>> logits = Tensor([[1.0, 3.0], [2.0, 0.1]])
        >>> targets = np.array([1, 0])
        >>> round(accuracy(logits, targets), 2)
        1.0
    """
    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == targets))
