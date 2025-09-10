"""
ONNX IO stubs.

These functions establish the public API for ONNX export/import. They raise
clear errors until implemented and only import heavy deps lazily.
"""
from typing import Any, Optional

def _require_onnx():
    try:
        import onnx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ONNX support requires the optional 'onnx' package. Install via pip install forgeNN[onnx]"
        ) from e

def export_onnx(model: Any, path: str, opset: Optional[int] = None, input_example: Optional[Any] = None) -> None:
    """Export a forgeNN model to ONNX format.

    Args:
        model: A compiled forgeNN model (e.g., Sequential) to export.
        path: Output .onnx file path.
        opset: Optional ONNX opset version to target.
        input_example: Optional sample input for tracing.
    """
    _require_onnx()
    raise NotImplementedError("ONNX export is not implemented yet.")

def load_onnx(path: str) -> Any:
    """Load an ONNX model and return a forgeNN-compatible representation.

    Args:
        path: Path to .onnx file.
    """
    _require_onnx()
    raise NotImplementedError("ONNX load is not implemented yet.")
