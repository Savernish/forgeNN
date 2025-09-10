"""
CUDA backend scaffold.

This is a placeholder for a future CUDA implementation (e.g., via CuPy or
custom kernels). Importing this module should remain optional.
"""
try:
    import cupy as cp  # type: ignore
    _cupy_available = True
except Exception:  # pragma: no cover - optional
    cp = None  # type: ignore
    _cupy_available = False

class CUDABackend:
    name = "cuda-cupy"
    xp = cp

    def __init__(self):
        if not _cupy_available:
            raise RuntimeError("CuPy is not installed; CUDA backend unavailable.")

cuda_backend = None
if _cupy_available:
    cuda_backend = CUDABackend()

__all__ = ["CUDABackend", "cuda_backend"]
