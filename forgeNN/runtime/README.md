Runtime Device API
==================

Public helpers:
- get_default_device() -> "cpu"|"cuda"
- set_default_device(device)
- is_cuda_available() -> bool
- use_device(device): context manager

Tensor and layers remain NumPy/CPU today; future work can dispatch based on
the default device and a selected backend.
