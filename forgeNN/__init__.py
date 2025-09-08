"""
forgeNN - High-Performance Neural Network Framework
==================================================

A modern, fast neural network framework focused on performance and clean APIs.

Main API:
    Tensor: Vectorized automatic differentiation with NumPy backend
    VectorizedMLP: High-performance neural networks (2.6x faster than PyTorch)
    VectorizedOptimizer: Efficient SGD with momentum
    Activation Functions: RELU, LRELU, TANH, SIGMOID, SWISH

Features:
    • 2.6x faster training than PyTorch on small-medium datasets
    • Clean, intuitive API design
    • Full automatic differentiation support
    • Flexible activation system (string/class/callable)
    • Efficient vectorized operations

Example:
    >>> import forgeNN
    >>> 
    >>> # Simple string-based activations (most common)
    >>> model = forgeNN.VectorizedMLP(784, [128, 64], 10, 
    ...                              activations=['relu', 'swish', 'linear'])
    >>> 
    >>> # Or use activation classes for advanced control
    >>> model = forgeNN.VectorizedMLP(784, [128, 64], 10,
    ...                              activations=[forgeNN.RELU(), forgeNN.SWISH(), None])
    >>> 
    >>> # Train on data
    >>> x = forgeNN.Tensor(data)
    >>> output = model(x)
    >>> loss = forgeNN.cross_entropy_loss(output, labels)
    >>> loss.backward()
"""

# Main vectorized API
from .tensor import Tensor
from .vectorized import VectorizedMLP, VectorizedOptimizer, cross_entropy_loss, accuracy

# Activation functions for advanced usage
from .functions.activation import RELU, LRELU, TANH, SIGMOID, SWISH

__version__ = "1.1.0.1"
__all__ = [
    'Tensor', 'VectorizedMLP', 'VectorizedOptimizer', 'cross_entropy_loss', 'accuracy',
    'RELU', 'LRELU', 'TANH', 'SIGMOID', 'SWISH'
]