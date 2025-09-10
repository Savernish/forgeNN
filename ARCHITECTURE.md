Architecture Notes (v2 scaffolding)
===================================

Scope
-----
This document describes the scaffolding added to support future CUDA and ONNX
integrations without breaking the current public API.

Runtime Device API
------------------
- forgeNN.runtime.device exposes:
  - get_default_device(), set_default_device(), is_cuda_available(), use_device()
- Default device is "cpu"; CUDA reports unavailable until implemented.
- Future: Tensor and layers consult the default device to choose an execution backend.

Backends
--------
- forgeNN.backends.cpu: NumPy backend (reference, default)
- forgeNN.backends.cuda: placeholder for CuPy or custom kernels (optional import)
- Backends expose an `xp` array module and may house optimized kernels.

ONNX
----
- forgeNN.onnx.io: export_onnx(model, path, opset=None, input_example=None), load_onnx(path)
- Both raise NotImplementedError for now and lazily require `onnx` package.
- Future: Tracing/graph capture from Sequential to ONNX graph; partial importer.

Packaging
---------
- Optional extras in pyproject.toml: `[onnx]`, `[cuda]`, and included in `[all]`.
- No hard dependency on onnx/cupy for base install.

Non-Goals (for now)
-------------------
- No device-aware Tensor implementation yet (all NumPy/CPU).
- No execution engine changes yet; this is purely structural groundwork.
