"""forgeNN Optimizers
=======================

Implementation of core optimization algorithms used during training.

Design Goals:
 - Keep state handling simple and explicit (lists parallel to parameter list)
 - Provide minimal, well-understood algorithms first (SGD, Adam, AdamW)
 - Avoid silent parameter list mutation (capture list on construction)
 - Keep public API tiny: step(), zero_grad()

Backward Compatibility:
 - ``VectorizedOptimizer`` from ``vectorized.py`` is now an alias of ``SGD``.

Usage Examples:
    >>> import forgeNN as fnn
    >>> model = fnn.Sequential([fnn.Dense(32) @ 'relu', fnn.Dense(10)])
    >>> opt = fnn.Adam(model.parameters(), lr=1e-3)
    >>> # inside a training loop
    >>> logits = model(fnn.Tensor(X_batch))
    >>> loss = logits.cross_entropy_loss(y_batch)
    >>> loss.backward(); opt.step(); opt.zero_grad()
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Optional
import numpy as np


class Optimizer:
    """Base optimizer.

    Parameters
    ----------
    params: iterable of Tensors exposing .data (np.ndarray) and .grad (np.ndarray|None)
    """

    def __init__(self, params: Optional[Iterable] = None):  # type: ignore[Any]
        # Allow deferred binding: params may be provided later via set_params()
        self.params = list(params) if params is not None else []

    def step(self):  # pragma: no cover - interface
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    # Deferred parameter binding -------------------------------------------------
    def set_params(self, params: Iterable):  # type: ignore[Any]
        """Bind parameters after optimizer construction.

        Allows usage pattern:
            opt = Adam(lr=1e-3)
            compiled = compile(model, optimizer=opt)

        The training loop will call set_params automatically if needed.
        Subclasses override _init_state() to allocate per-parameter buffers.
        """
        # Replace params only if previously empty or identical length
        self.params = list(params)
        self._init_state()

    def _init_state(self):  # pragma: no cover - overridden in subclasses
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum.

    Args:
        params: parameter list
        lr: learning rate
        momentum: momentum factor (0 disables)
        weight_decay: L2 penalty added to gradient (classic formulation)
    """

    def __init__(self, params: Optional[Iterable] = None, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):  # type: ignore[Any]
        super().__init__(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._bufs = [np.zeros_like(p.data) for p in self.params] if (self.momentum > 0 and self.params) else None
        # Backward compatibility attribute expected by older tests
        self.momentum_buffers = self._bufs

    def _init_state(self):  # allocate momentum buffers when params set
        self._bufs = [np.zeros_like(p.data) for p in self.params] if self.momentum > 0 else None
        self.momentum_buffers = self._bufs

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self._bufs is not None:
                self._bufs[i] = self.momentum * self._bufs[i] + g
                g = self._bufs[i]
            # keep alias updated
            self.momentum_buffers = self._bufs
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2015).

    Implements bias-corrected first and second moment estimates.

    Args:
        params: parameters
        lr: learning rate (default 1e-3)
        betas: (beta1, beta2) coefficients
        eps: numerical stability term
        weight_decay: L2 penalty (classic Adam variant; NOT decoupled)
    """

    def __init__(
        self,
        params: Optional[Iterable] = None,  # type: ignore[Any]
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        # State (may be empty if params deferred)
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def _init_state(self):
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            m = self.m[i] = b1 * self.m[i] + (1 - b1) * g
            v = self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)
            # Bias correction
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW (Decoupled Weight Decay) optimizer.

    Differs from Adam by applying weight decay directly to parameters
    (decoupled) instead of adding to the gradient.
    """

    def __init__(
        self,
        params: Optional[Iterable] = None,  # type: ignore[Any]
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
        self.decoupled_weight_decay = float(weight_decay)

    def step(self):
        # Decoupled weight decay
        if self.decoupled_weight_decay:
            for p in self.params:
                p.data -= self.lr * self.decoupled_weight_decay * p.data
        super().step()


__all__ = [
    'Optimizer', 'SGD', 'Adam', 'AdamW'
]