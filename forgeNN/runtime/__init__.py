"""
Runtime utilities for device management and future backends.

This module provides a thin, optional device API so future CUDA or other
accelerated backends can be integrated without changing user code.
"""

from .device import (
    get_default_device,
    set_default_device,
    is_cuda_available,
    use_device,
)

__all__ = [
    "get_default_device",
    "set_default_device",
    "is_cuda_available",
    "use_device",
]
