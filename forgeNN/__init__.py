"""
forgeNN - Minimal, fast neural network framework (v2)
====================================================

Lean public API with stable, documented entry points. Docstrings preserved.
"""

from .core.tensor import Tensor
from .core.tensor import randint, stack
from .layers import Layer, ActivationWrapper, Sequential, Dense, Flatten, Input, Dropout, GlobalAvgPool1D, GlobalAvgPool2D, Embedding, LayerNorm
from .transformer.layers import TransformerLayer, MHA, MultiHeadAttention, TransformerBlock, PositionalEncoding, PositionalEmbedding
from .optimizers import Optimizer, SGD, Adam, AdamW
from .training import compile
from .nn.losses import cross_entropy_loss
from .nn.metrics import accuracy
from .runtime import get_default_device, set_default_device, is_cuda_available, use_device
from .onnx import export_onnx, load_onnx

# Expose subpackages as attributes for convenient access (e.g., fnn.nn.set_seed)
from . import nn as nn  # noqa: F401

__version__ = "2.0.0b0"
__all__ = [
    'Tensor', 'randint', 'stack',
    'Layer', 'ActivationWrapper', 'Sequential', 'Dense', 'Flatten', 'Input', 'Dropout',
    'TransformerLayer', 'MHA', 'MultiHeadAttention', 'TransformerBlock', 'PositionalEncoding', 'PositionalEmbedding',
    'GlobalAvgPool1D', 'GlobalAvgPool2D', 'Embedding', 'LayerNorm',
    'Optimizer', 'SGD', 'Adam', 'AdamW',
    'compile',
    'cross_entropy_loss', 'accuracy',
    # runtime
    'get_default_device', 'set_default_device', 'is_cuda_available', 'use_device',
    # onnx
    'export_onnx', 'load_onnx',
    # subpackages
    'nn',
]