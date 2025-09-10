"""
CPU (NumPy) backend for forgeNN.

Provides a thin wrapper object so future backends can conform to the same API.
"""
import numpy as np

class NumPyBackend:
    name = "cpu-numpy"
    xp = np  # expose numpy as xp for generic code

    # Future: house kernel overrides or jit-accelerated paths

numpy_backend = NumPyBackend()

__all__ = ["NumPyBackend", "numpy_backend"]
