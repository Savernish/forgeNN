"""
Training utilities providing a simple compile/fit style API.

This module adds a minimal Keras-like workflow on top of forgeNN models
without modifying the model classes. Works with any model exposing:
 - __call__(Tensor) -> Tensor
 - parameters() -> List[Tensor]

Usage:
    import forgeNN as fnn
    model = fnn.Sequential([fnn.Dense(64) @ 'relu', fnn.Dense(10)])
    compiled = fnn.compile(model, optimizer={"lr": 0.01, "momentum": 0.9},
                           loss="cross_entropy", metrics=["accuracy"])
    compiled.fit(X_train, y_train, epochs=5, batch_size=64,
                 validation_data=(X_test, y_test))
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .core.tensor import Tensor
from .nn.losses import cross_entropy_loss, mse
from .nn.metrics import accuracy
from .optimizers import SGD, Adam, AdamW, Optimizer


# Registries for built-in losses and metrics
LossFn = Callable[[Tensor, np.ndarray], Tensor]
MetricFn = Callable[[Tensor, np.ndarray], float]

OPTIMIZERS = {"sgd": SGD, "adam": Adam, "adamw": AdamW}


LOSSES: Dict[str, LossFn] = {
    "cross_entropy": cross_entropy_loss,
    "mse": mse,
}

METRICS: Dict[str, MetricFn] = {
    "accuracy": accuracy,
}


class CompiledModel:
    """A compiled model with fit/evaluate/predict helpers.

    Parameters
    ----------
    model: Any
        A model with __call__(Tensor) and parameters() methods.
    optimizer: Optimizer | Dict[str, Any] | None
        If an optimizer instance is provided, it's used as-is. If a dict
    is provided, an optimizer will be constructed lazily from
        model.parameters() on first use. If None, defaults to {"lr": 0.01}.
    loss: str | Callable
        Either 'cross_entropy', 'mse', or a custom callable loss(logits, targets)->Tensor.
    metrics: Sequence[str | Callable]
        Metric names ('accuracy') or callables taking (logits, targets)->float.
    """

    def __init__(
        self,
        model: Any,
    optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
        loss: Union[str, LossFn] = "cross_entropy",
        metrics: Optional[Sequence[Union[str, MetricFn]]] = None,
    ) -> None:
        self.model = model
        self._optimizer: Optional[Optimizer] = None
        if optimizer is None or isinstance(optimizer, dict):
            self._opt_config: Dict[str, Any] = {"lr": 0.01}
            if isinstance(optimizer, dict):
                self._opt_config.update(optimizer)
        else:
            self._optimizer = optimizer
            self._opt_config = {}

        # Loss function
        if isinstance(loss, str):
            if loss not in LOSSES:
                raise ValueError(f"Unknown loss '{loss}'. Available: {list(LOSSES.keys())}")
            self.loss_fn = LOSSES[loss]
        else:
            self.loss_fn = loss

        # Metrics
        metrics = metrics or []
        self.metric_fns: List[Tuple[str, MetricFn]] = []
        for m in metrics:
            if isinstance(m, str):
                if m not in METRICS:
                    raise ValueError(f"Unknown metric '{m}'. Available: {list(METRICS.keys())}")
                self.metric_fns.append((m, METRICS[m]))
            else:
                # best effort name
                self.metric_fns.append((getattr(m, "__name__", "metric"), m))

    @property
    def optimizer(self) -> Optimizer:
        """Return the underlying optimizer, constructing it lazily if needed.

        Notes:
            For lazily initialized layers (e.g., Dense without in_features),
            ensure a dummy forward is run before first optimizer access so that
            parameters() is populated.
        """
        if self._optimizer is None:
            params = self.model.parameters()
            # Build optimizer via factory config path
            opt_type = self._opt_config.pop('type', 'sgd').lower()
            cls = OPTIMIZERS.get(opt_type)
            if cls is None:
                raise ValueError(f"Unknown optimizer type '{opt_type}'. Available: {list(OPTIMIZERS.keys())}")
            self._optimizer = cls(params, **self._opt_config)
        elif getattr(self._optimizer, 'params', None) == []:
            # Deferred binding case: user passed opt = Adam(lr=...) without params
            self._optimizer.set_params(self.model.parameters())
        return self._optimizer

    def _data_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        """Yield mini-batches from arrays X and y.

        Args:
            X: Input features of shape (N, ...).
            y: Targets of shape (N,).
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle before batching.

        Yields:
            Tuple[np.ndarray, np.ndarray]: (batch_X, batch_y)
        """
        n = len(X)
        if shuffle:
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            yield X[i:j], y[i:j]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        shuffle: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: int = 1,
    ) -> None:
        """Train the model for a fixed number of epochs.

        Args:
            X: Training features of shape (N, D).
            y: Training labels of shape (N,).
            epochs: Number of epochs to train.
            batch_size: Batch size.
            shuffle: Shuffle data each epoch.
            validation_data: Optional tuple (X_val, y_val) for per-epoch validation.
            verbose: 0 = silent, 1 = print epoch summary.

        Notes:
            Loss and metrics are aggregated sample-weighted across batches.
        """
        for epoch in range(1, epochs + 1):
            # On-the-fly aggregation without post-epoch extra forward pass
            loss_sum = 0.0
            weight_sum = 0
            metric_sums = {name: 0.0 for name, _ in self.metric_fns}
            correct_total = 0  # special path for accuracy to avoid recompute
            for bx, by in self._data_loader(X, y, batch_size, shuffle):
                logits = self.model(Tensor(bx))
                loss = self.loss_fn(logits, by)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bsz = len(by)
                weight_sum += bsz
                loss_sum += float(loss.data) * bsz
                for name, fn in self.metric_fns:
                    if name == 'accuracy':
                        preds = np.argmax(logits.data, axis=1)
                        correct_total += int(np.sum(preds == by))
                    else:
                        metric_sums[name] += float(fn(logits, by)) * bsz

            avg_loss = loss_sum / max(weight_sum, 1)
            avg_metrics = {}
            for name, _ in self.metric_fns:
                if name == 'accuracy':
                    avg_metrics[name] = correct_total / max(weight_sum, 1)
                else:
                    avg_metrics[name] = metric_sums[name] / max(weight_sum, 1)

            # Validation
            val_str = ""
            if validation_data is not None:
                vx, vy = validation_data
                v_loss, v_metrics = self.evaluate(vx, vy, batch_size=batch_size)
                val_parts = [f"val_loss={v_loss:.4f}"] + [f"val_{k}={v_metrics[k]*100:.1f}%" if k == "accuracy" else f"val_{k}={v_metrics[k]:.4f}" for k in v_metrics]
                val_str = "  " + ", ".join(val_parts)

            if verbose:
                parts = [f"Epoch {epoch}/{epochs}", f"loss={avg_loss:.4f}"]
                for k, v in avg_metrics.items():
                    parts.append(f"{k}={v*100:.1f}%" if k == "accuracy" else f"{k}={v:.4f}")
                line = ", ".join(parts) + val_str
                print(line)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> Tuple[float, Dict[str, float]]:
        """Evaluate loss and metrics on a dataset.

        Args:
            X: Features of shape (N, D).
            y: Targets of shape (N,).
            batch_size: Batch size for evaluation.

        Returns:
            (loss, metrics): loss as float, metrics as dict (e.g., {'accuracy': 0.97}).
        """
        # Sample-weighted aggregation and exact accuracy counting
        loss_sum = 0.0
        weight_sum = 0
        metric_sums = {name: 0.0 for name, _ in self.metric_fns}
        correct_total = 0

        for bx, by in self._data_loader(X, y, batch_size, shuffle=False):
            logits = self.model(Tensor(bx, requires_grad=False))
            loss = self.loss_fn(logits, by)
            bsz = len(by)
            loss_sum += float(loss.data) * bsz
            weight_sum += bsz

            # Metrics
            for name, fn in self.metric_fns:
                if name == "accuracy":
                    # Count correct predictions for exact dataset accuracy
                    preds = np.argmax(logits.data, axis=1)
                    correct_total += int(np.sum(preds == by))
                else:
                    metric_sums[name] += float(fn(logits, by)) * bsz

        avg_loss = loss_sum / max(weight_sum, 1)
        avg_metrics: Dict[str, float] = {}
        for name, _ in self.metric_fns:
            if name == "accuracy":
                avg_metrics[name] = (correct_total / max(weight_sum, 1))
            else:
                avg_metrics[name] = metric_sums[name] / max(weight_sum, 1)
        return avg_loss, avg_metrics

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Run forward inference and return logits for X.

        Args:
            X: Features of shape (N, D).
            batch_size: Batch size for inference.

        Returns:
            ndarray of shape (N, num_classes) with logits.
        """
        outputs: List[np.ndarray] = []
        for bx, _ in self._data_loader(X, np.zeros(len(X)), batch_size, shuffle=False):
            logits = self.model(Tensor(bx, requires_grad=False))
            outputs.append(logits.data)
        return np.vstack(outputs) if outputs else np.empty((0,))


def compile(
    model: Any,
    optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
    loss: Union[str, LossFn] = "cross_entropy",
    metrics: Optional[Sequence[Union[str, MetricFn]]] = None,
) -> CompiledModel:
    """Create a CompiledModel for Keras-like training.

    Example
    -------
    >>> import forgeNN as fnn
    >>> model = fnn.Sequential([fnn.Dense(64) @ 'relu', fnn.Dense(10)])
    >>> c = fnn.compile(model, optimizer={"lr": 0.01}, loss='cross_entropy', metrics=['accuracy'])
    >>> c.fit(X, y, epochs=5)
    """
    return CompiledModel(model, optimizer=optimizer, loss=loss, metrics=metrics)
