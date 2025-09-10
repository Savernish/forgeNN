"""
Backends registry for computational kernels.

Today only the NumPy CPU backend is implemented. A CUDA backend will be added
in the future without changing the public API.
"""
from .cpu import numpy_backend

__all__ = ["numpy_backend"]
