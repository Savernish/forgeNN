"""
forgeNN - High-Performance Neural Network Framework
==================================================

A modern neural network framework with both educational and production implementations.

Main API (Recommended for production):
    Tensor: Vectorized automatic differentiation
    VectorizedMLP: High-performance neural networks
    VectorizedOptimizer: Efficient training algorithms

Legacy API (Educational):
    Available in forgeNN.legacy for learning purposes

Example:
    >>> from forgeNN.tensor import Tensor
    >>> from forgeNN.vectorized import VectorizedMLP
    >>> 
    >>> # Create a neural network
    >>> model = VectorizedMLP(784, [128, 64], 10)
    >>> 
    >>> # Train on data
    >>> x = Tensor(data)
    >>> output = model(x)
"""

# Main vectorized API
from .tensor import Tensor
from .vectorized import VectorizedMLP, VectorizedOptimizer, cross_entropy_loss, accuracy

# Legacy API is available as forgeNN.legacy
from . import legacy

__version__ = "1.0.3"
__all__ = ['Tensor', 'VectorizedMLP', 'VectorizedOptimizer', 'cross_entropy_loss', 'accuracy', 'legacy']