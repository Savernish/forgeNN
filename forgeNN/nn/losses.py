"""
Loss functions for forgeNN v2

Docstrings kept consistent with previous formatting.
"""
from __future__ import annotations
import numpy as np
from ..core.tensor import Tensor


def mse(logits: Tensor, targets: np.ndarray | Tensor) -> Tensor:
    """
    Compute Mean Squared Error (MSE) loss.

    This returns the mean over all elements, i.e., mean((logits - targets)^2).
    Gradients are handled by the existing Tensor operations, ensuring
    correct scaling over batch and feature dimensions and supporting broadcasting.
    
        Args:
                logits (Tensor): Predictions of shape (N, D, ...)
                targets (ndarray | Tensor):
                        - If same shape as logits (or broadcastable), used directly.
                        - If 1D integer class indices and logits has shape (N, C) with C>1,
                            targets are automatically one-hot encoded to (N, C).
                        - If 1D floating values and logits has shape (N, 1) they are reshaped to (N,1).
        
    Returns:
        Tensor: Scalar loss value connected to logits for backprop.

    Example:
        >>> preds = Tensor([[0.5, 0.2], [0.1, 0.4]])
        >>> y = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> loss = mse(preds, y)
        >>> loss.backward()
        >>> preds.grad.shape
        (2, 2)
    """
    t = targets if isinstance(targets, Tensor) else Tensor(np.asarray(targets), requires_grad=False)

    if isinstance(t, Tensor) and t.data.ndim == 1 and logits.data.ndim >= 2:
        batch = logits.data.shape[0]
        if all(d == 1 for d in logits.data.shape[1:]):
            t = Tensor(t.data.reshape((batch,) + logits.data.shape[1:]), requires_grad=False)
        elif len(logits.data.shape) == 2:
            C = logits.data.shape[1]
            labels = t.data.astype(int)
            if labels.min() >= 0 and labels.max() < C:
                one_hot = np.zeros((batch, C), dtype=logits.data.dtype)
                one_hot[np.arange(batch), labels] = 1.0
                t = Tensor(one_hot, requires_grad=False)
    diff = logits - t
    return (diff * diff).mean()


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Compute cross-entropy loss for classification with numerical stability.
    
    Args:
        logits (Tensor): Raw model outputs (batch_size, num_classes)
        targets (np.ndarray): Class indices (batch_size,)
        
    Returns:
        Tensor: Scalar loss value connected to logits for backprop.

    Example:
        >>> logits = Tensor([[1., 0.5], [0.2, 0.8]])
        >>> y = np.array([0, 1])
        >>> loss = cross_entropy_loss(logits, y)
        >>> loss.backward()
        >>> logits.grad.shape
        (2, 2)
    """
    targets = np.asarray(targets, dtype=np.int64)
    data = logits.data
    if data.dtype != np.float32:
        data32 = data.astype(np.float32, copy=False)
        logits.data = data32  # keep graph linkage

    batch_size = data.shape[0]
    max_per_row = logits.max(axis=1, keepdims=True)
    shifted = logits - max_per_row
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    probs = exp_shifted / sum_exp
    log_sum_exp = sum_exp.log()
    log_probs = shifted - log_sum_exp

    batch_idx = np.arange(batch_size)
    selected = log_probs.data[batch_idx, targets]
    loss_value = -float(np.mean(selected))
    loss = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op='cross_entropy')

    if logits.requires_grad:
        probs_data = probs.data
        targets_local = targets
        bsz = batch_size

        def _backward():
            grad = probs_data.copy()
            grad[batch_idx, targets_local] -= 1.0
            grad /= bsz
            logits.grad += grad * loss.grad

        loss._backward = _backward

    return loss
