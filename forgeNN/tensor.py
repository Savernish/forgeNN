"""Compatibility shim: re-export Tensor from core.

This keeps imports like 'from forgeNN.tensor import Tensor' working while
the implementation lives in 'forgeNN.core.tensor'.
"""

from .core.tensor import *  # noqa: F401,F403
