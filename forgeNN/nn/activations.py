"""
Activation registry for forgeNN v2.

Provides a unified mapping for string and class-based activations
without importing heavy modules elsewhere. Docstring style preserved.
"""
from __future__ import annotations
from typing import Callable, Union, Type
from ..core.tensor import Tensor
"""
In v2, legacy class-based activations from forgeNN.functions were removed.
Keep optional class keys disabled by default; users can still pass strings
('relu', 'tanh', etc.) or callables. This avoids importing removed modules.
"""
RELU = LRELU = TANH = SIGMOID = SWISH = None  # type: ignore


def _relu(x: Tensor) -> Tensor:
    return x.relu()


def _sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def _tanh(x: Tensor) -> Tensor:
    return x.tanh()


def _linear(x: Tensor) -> Tensor:
    return x


def _lrelu(x: Tensor) -> Tensor:
    return x.leaky_relu()


def _swish(x: Tensor) -> Tensor:
    return x.swish()


ACTIVATION_FUNCTIONS: dict[Union[str, Type], Callable[[Tensor], Tensor]] = {
    'relu': _relu,
    'sigmoid': _sigmoid,
    'tanh': _tanh,
    'linear': _linear,
    'lrelu': _lrelu,
    'swish': _swish,
}

# Class keys when available
for cls, fn in ((RELU, _relu), (LRELU, _lrelu), (TANH, _tanh), (SIGMOID, _sigmoid), (SWISH, _swish)):
    if isinstance(cls, type):
        ACTIVATION_FUNCTIONS[cls] = fn
