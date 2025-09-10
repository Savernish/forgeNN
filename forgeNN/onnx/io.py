"""
ONNX IO stubs.

These functions establish the public API for ONNX export/import. They raise
clear errors until implemented and only import heavy deps lazily.
"""
from typing import Any, Optional, Tuple, Sequence, Union
import numpy as np
from ..core.tensor import Tensor

def _require_onnx():
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        return onnx, helper, TensorProto, numpy_helper  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ONNX support requires the optional 'onnx' package. Install via pip install forgeNN[onnx]"
        ) from e

def _shape_from_example(x: Union[Tuple[int, ...], np.ndarray]) -> Tuple[int, ...]:
    if isinstance(x, tuple):
        return tuple(int(d) for d in x)
    if hasattr(x, 'shape'):
        shp = tuple(int(d) for d in x.shape)
        if len(shp) == 0:
            raise ValueError("input_example must include a batch dimension; got scalar.")
        return shp[1:] #drop batch dim
    raise ValueError("input_example must be a shape tuple or an array-like with .shape")

def _activation_name(core: Any) -> str:
    """Best-effort activation name extraction for ActivationWrapper or tokens.

    Returns a lowercase string such as 'relu', 'sigmoid', 'tanh', 'softmax', or 'linear'.
    Returns empty string when unknown/unsupported.
    """
    # ActivationWrapper or plain token
    if core.__class__.__name__ == 'ActivationWrapper':
        # Prefer explicit attributes first
        name = getattr(core, "name", None) or getattr(core, "activation_name", None)
        if isinstance(name, str) and name:
            return name.lower()
        # Fallback to the wrapped activation object
        act = getattr(core, "activation", None)
        if isinstance(act, str):
            return act.lower()
        # Callable or class: use its __name__ if available
        cand = getattr(act, '__name__', None) or getattr(getattr(act, '__class__', None), '__name__', None)
        if isinstance(cand, str) and cand:
            return cand.lower()
        return ""
    if isinstance(core, str):
        return core.lower()
    return ""

def export_onnx(
        model: Any,
        path: str,
        opset: Optional[int] = None,
        input_example: Optional[Any] = None,
        include_softmax: bool = False,
        fold_dropout: bool = True,
        ir_version: Optional[int] = 10,
    ) -> None:
    """Export a forgeNN model to ONNX format.

    Supports: Input, Dense (Gemm), Activation (relu/sigmoid/tanh, optional softmax), Flatten.
    Future support may include Conv, Pooling, BatchNorm, Dropout (folded). @work

    Args:
        model: A compiled forgeNN model (e.g., Sequential) to export.
        path: Output .onnx file path.
        opset: Optional ONNX opset version to target.
        input_example: Optional sample input for tracing.
        include_softmax: If True, include final softmax layer if present.
        fold_dropout: If True, fold Dropout layers into preceding layers.

    Example:
        >>> import forgeNN as fnn
        >>> model = fnn.Sequential([
        ...     fnn.Input((20,)),
        ...     fnn.Dense(64) @ 'relu',
        ...     fnn.Dense(32) @ 'relu',
        ...     fnn.Dense(10) @ 'linear',  # logits
        ... ])
        >>> model.summary((20,))
        >>> fnn.onnx.export_onnx(model, "model.onnx", input_example=np.random.randn(1,20).astype(np.float32))
    """
    onnx, helper, TensorProto, numpy_helper = _require_onnx()
    if input_example is None:
        raise ValueError("export_onnx requires input_example (tuple shape or NumPy array) to infer input shape.")
    in_shape = _shape_from_example(input_example)
    if opset is None:
        opset = 13

    # Force eval during export
    was_training = getattr(model, "training", True)
    if hasattr(model, "train"):
        model.train(False)

    try:
        nodes = []
        initializers = []
        inputs = [helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["N", *list(in_shape)])]
        current = "input_0"
        current_shape = ["N", *list(in_shape)]
        name_id = {"dense": 0, "act": 0, "flatten": 0}

        def next_name(kind: str) -> str:
            i = name_id[kind]
            name_id[kind] = i + 1
            return f"{kind}_{i}"

        for lyr in getattr(model, "layers", []):
            is_wrap = (lyr.__class__.__name__ == "ActivationWrapper")
            core = getattr(lyr, "layer", lyr) if is_wrap else lyr
            cname = core.__class__.__name__

            if cname == "Input":
                continue

            if cname == "Flatten":
                n = next_name("flatten"); out = f"{n}_out"
                nodes.append(helper.make_node("Flatten", [current], [out], name=n, axis=1))
                # (N, prod(rest))
                if len(current_shape) >= 2:
                    feat = int(np.prod([int(d) for d in current_shape[1:]]))
                else:
                    feat = int(np.prod(in_shape))
                current_shape = ["N", feat]
                current = out
                continue

            if cname == "Dropout":
                if fold_dropout:
                    # identity in eval
                    continue
                n = next_name("act"); out = f"{n}_out"
                ratio = float(getattr(core, "rate", 0.5))
                nodes.append(helper.make_node("Dropout", [current], [out], name=n, ratio=ratio))
                current = out
                continue

            if cname == "Dense":
                # Dense as Gemm; decide transB from weight shape vs input features
                n = next_name("dense"); out = f"{n}_out"
                W = core.W.data.astype(np.float32, copy=False)
                b = core.b.data.astype(np.float32, copy=False)

                # Infer input features (last dim of current shape)
                in_feat = None
                if current_shape and isinstance(current_shape[-1], (int, np.integer)):
                    in_feat = int(current_shape[-1])
                else:
                    # Fallback to weight dims when symbolic
                    in_feat = int(W.shape[0])

                if W.shape[0] == in_feat:
                    transB = 0
                    out_dim = int(W.shape[1])
                elif W.shape[1] == in_feat:
                    transB = 1
                    out_dim = int(W.shape[0])
                else:
                    raise ValueError(f"Dense export: W{W.shape} not compatible with input features {in_feat}")

                Wn, bn = f"{n}_W", f"{n}_b"
                initializers.append(numpy_helper.from_array(W, Wn))
                initializers.append(numpy_helper.from_array(b, bn))
                nodes.append(helper.make_node(
                    "Gemm", inputs=[current, Wn, bn], outputs=[out],
                    name=n, alpha=1.0, beta=1.0, transB=transB
                ))
                current_shape = ["N", out_dim]
                current = out

                # Emit wrapped activation immediately
                if is_wrap:
                    act = _activation_name(lyr)
                    if act and act != "linear":
                        an = next_name("act"); aout = f"{an}_out"
                        if act == "relu":
                            nodes.append(helper.make_node("Relu", [current], [aout], name=an))
                        elif act == "sigmoid":
                            nodes.append(helper.make_node("Sigmoid", [current], [aout], name=an))
                        elif act == "tanh":
                            nodes.append(helper.make_node("Tanh", [current], [aout], name=an))
                        elif act == "softmax":
                            if include_softmax:
                                nodes.append(helper.make_node("Softmax", [current], [aout], name=an, axis=-1))
                            else:
                                aout = current
                        else:
                            raise ValueError(f"Unsupported activation for ONNX export: {act}")
                        current = aout
                continue

            # Standalone activation
            act = _activation_name(core)
            if act:
                an = next_name("act"); aout = f"{an}_out"
                if act == "relu":
                    nodes.append(helper.make_node("Relu", [current], [aout], name=an))
                elif act == "sigmoid":
                    nodes.append(helper.make_node("Sigmoid", [current], [aout], name=an))
                elif act == "tanh":
                    nodes.append(helper.make_node("Tanh", [current], [aout], name=an))
                elif act == "softmax":
                    if include_softmax:
                        nodes.append(helper.make_node("Softmax", [current], [aout], name=an, axis=-1))
                    else:
                        aout = current
                else:
                    raise ValueError(f"Unsupported activation for ONNX export: {act}")
                current = aout
                continue

            raise ValueError(f"ONNX export: unsupported layer {cname}")

        outputs = [helper.make_tensor_value_info(current, TensorProto.FLOAT, current_shape)]
        graph = helper.make_graph(nodes, "forgeNN_graph", inputs, outputs, initializer=initializers)
        model_proto = helper.make_model(
            graph,
            producer_name="forgeNN",
            opset_imports=[helper.make_opsetid("", int(opset))],
        )
        if ir_version is not None:
            try:
                model_proto.ir_version = int(ir_version)  # older ORT compat
            except Exception:
                pass
        model_proto.metadata_props.add(key="framework", value="forgeNN")
        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, path)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    # raise NotImplementedError("ONNX export is not implemented yet.") # done

def load_onnx(path: str) -> Any:
    """Load an ONNX model and return a forgeNN-compatible representation.

    Args:
        path: Path to .onnx file.
    """
    _require_onnx()
    raise NotImplementedError("ONNX load is not implemented yet.")
