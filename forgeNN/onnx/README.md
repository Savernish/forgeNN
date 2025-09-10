ONNX Integration (Scaffold)
===========================

Public API:
- export_onnx(model, path, opset=None, input_example=None)
- load_onnx(path)

Both functions currently raise NotImplementedError and lazily check for the
optional 'onnx' package. Implementation work can evolve here without touching
the public surface.
