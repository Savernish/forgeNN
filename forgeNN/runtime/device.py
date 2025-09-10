"""
Device management scaffolding.

This is a minimal abstraction to support future CUDA (GPU) backends while
remaining a no-op for CPU/NumPy today.
"""
from contextlib import contextmanager
from typing import Literal, Optional

Device = Literal["cpu", "cuda"]

_default_device: Device = "cpu"

def is_cuda_available() -> bool:
    """Return True if a CUDA backend is available.

    This stub checks for optional dependencies lazily in the future. For now,
    always returns False.
    """
    # Future: detect via cupy, torch.cuda, or custom CUDA backend init
    return False

def get_default_device() -> Device:
    """Get the current default device ("cpu" or "cuda")."""
    return _default_device

def set_default_device(device: Device) -> None:
    """Set the global default device.

    If 'cuda' is requested but unavailable, raises a RuntimeError.
    """
    global _default_device
    if device == "cuda" and not is_cuda_available():
        raise RuntimeError("CUDA requested but not available. Install a CUDA backend.")
    _default_device = device

@contextmanager
def use_device(device: Optional[Device]):
    """Context manager to temporarily switch the default device.

    Example:
        with use_device("cpu"):
            ...
    """
    if device is None:
        yield
        return
    prev = get_default_device()
    try:
        set_default_device(device)
        yield
    finally:
        set_default_device(prev)
