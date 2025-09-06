"""
Legacy forgeNN Implementation
============================

This module contains the original scalar implementation of forgeNN.
These classes are kept for educational purposes and backward compatibility.

For production use, please use the vectorized implementation:
- forgeNN.tensor.Tensor instead of forgeNN.legacy.core.Value
- forgeNN.vectorized.VectorizedMLP instead of forgeNN.legacy.network.MLP

Classes:
    Value: Scalar automatic differentiation engine
    Module: Base class for neural network components  
    Neuron: Single neuron implementation
    Layer: Layer of neurons
    MLP: Multi-layer perceptron
"""

from .core import Value
from .network import Module, Neuron, Layer, MLP

__all__ = ['Value', 'Module', 'Neuron', 'Layer', 'MLP']
