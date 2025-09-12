"""
ONNX IO stubs.

These functions establish the public API for ONNX export/import. They raise
clear errors until implemented and only import heavy deps lazily.
"""
from typing import Any, Optional, Tuple, Sequence, Union
import numpy as np
from ..core.tensor import Tensor
import forgeNN as fnn  # type: ignore[import]

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
        ir_version: Optional ONNX IR version to set in the model metadata.

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

def load_onnx(path: str, strict: bool = True) -> fnn.Sequential:
    """Load an ONNX model (Stage 1 subset) into a forgeNN.Sequential.

    Supported (linear graphs only):
      - Gemm (Dense) with alpha/beta/transB
      - Activations: Relu/Sigmoid/Tanh (attached after Dense when adjacent)
      - Flatten(axis=1)
      - Optional Softmax at the end (attached after last Dense when adjacent)

    Args:
        path: Path to .onnx file
        strict: If True, raise on unsupported ops or non-linear graphs; if False, skip benign no-ops
    """
    onnx, helper, TensorProto, numpy_helper = _require_onnx()
    model = onnx.load(path)
    onnx.checker.check_model(model)
    g = model.graph

    if len(g.input) != 1 or len(g.output) != 1:
        raise ValueError("ONNX import (Stage 1) expects exactly one input and one output.")

    # Input details
    in_name = g.input[0].name
    # Extract feature dims (drop batch). If not available, leave None and rely on summary call to infer.
    in_dims = []
    try:
        tt = g.input[0].type.tensor_type
        for i, d in enumerate(tt.shape.dim):
            if i == 0:
                continue  # batch dim
            if d.dim_value > 0:
                in_dims.append(int(d.dim_value))
    except Exception:
        pass
    input_shape = tuple(in_dims) if in_dims else None

    # Initializers map
    inits = {init.name: numpy_helper.to_array(init) for init in g.initializer}

    # We'll parse nodes linearly in listed order (exporter emits a linear chain)
    nodes = list(g.node)
    layers: list[Any] = []
    dense_params: list[tuple[np.ndarray, np.ndarray]] = []  # (W, b) in (in,out) orientation

    i = 0
    N = len(nodes)
    while i < N:
        node = nodes[i]
        op = node.op_type

        if op in ("Identity", "Dropout"):
            # Skip benign no-ops; Dropout assumed eval-mode
            i += 1
            continue

        if op == "Flatten":
            # axis default is 1 in our exporter
            axis = 1
            for a in node.attribute:
                if a.name == "axis":
                    axis = int(a.i)
                    break
            if axis != 1 and strict:
                raise ValueError(f"Flatten axis={axis} not supported in Stage 1 (expected axis=1)")
            layers.append(fnn.Flatten())
            i += 1
            continue

        if op == "Gemm":
            # Extract weights and bias
            A = node.input[0]
            B_name = node.input[1]
            C_name = node.input[2] if len(node.input) > 2 else None
            if B_name not in inits or (C_name is not None and C_name not in inits):
                raise ValueError(f"ONNX import: missing initializer for Gemm weights/bias: {B_name}, {C_name}")
            B = inits[B_name]
            C = inits[C_name] if C_name is not None else None

            alpha = 1.0
            beta = 1.0
            transB = 0
            for a in node.attribute:
                if a.name == "alpha":
                    alpha = float(a.f) if a.type == 1 else float(a.i)
                elif a.name == "beta":
                    beta = float(a.f) if a.type == 1 else float(a.i)
                elif a.name == "transB":
                    transB = int(a.i)

            # Compute effective W (in,out) and b (out,)
            W_eff = (B.T if transB else B).astype(np.float32, copy=False)
            W_eff = (alpha * W_eff).astype(np.float32, copy=False)
            if C is None:
                b_eff = np.zeros((W_eff.shape[1],), dtype=np.float32)
            else:
                b_arr = C.astype(np.float32, copy=False)
                # C can be (out,) or (1,out); ensure (out,)
                b_eff = b_arr.reshape(-1).astype(np.float32, copy=False) * float(beta)

            out_dim = int(W_eff.shape[1])

            # Look ahead for immediate activation to attach
            act_name = None
            if i + 1 < N and nodes[i + 1].op_type in ("Relu", "Sigmoid", "Tanh", "Softmax"):
                nxt = nodes[i + 1]
                act_name = nxt.op_type.lower()
                # Only keep softmax if it's intended as final activation; otherwise we can attach anyway.
                i_adv = 1
            else:
                i_adv = 0

            # Create Dense (optionally wrapped)
            dense = fnn.Dense(out_dim)
            if act_name and act_name != "softmax":
                dense = dense @ act_name
            elif act_name == "softmax":
                # Attach softmax too; forgeNN supports 'softmax' token
                dense = dense @ "softmax"
            layers.append(dense)
            dense_params.append((W_eff, b_eff))

            i += 1 + i_adv
            continue

        if op in ("Relu", "Sigmoid", "Tanh", "Softmax"):
            # Standalone activation without preceding Gemm: attach as its own layer if strict is False
            if strict:
                raise ValueError(f"Unexpected standalone activation {op} without preceding Gemm in Stage 1 importer")
            layers.append(fnn.Dense(0) @ op.lower())  # placeholder; will likely fail without Dense
            i += 1
            continue

        # Unknown op
        if strict:
            raise ValueError(f"Unsupported op in Stage 1 importer: {op}")
        # else skip
        i += 1

    # Build Sequential, optionally with Input for clarity
    if input_shape:
        seq = fnn.Sequential([fnn.Input(input_shape), *layers])
        # Initialize shapes/params by running summary
        try:
            seq.summary(input_shape)
        except Exception:
            pass
    else:
        seq = fnn.Sequential(layers)

    # Assign weights to Dense layers in order
    # Find Dense cores in sequence order (unwrap ActivationWrapper)
    dense_layers: list[Any] = []
    for lyr in seq.layers:
        core = getattr(lyr, "layer", lyr)
        if core.__class__.__name__ == "Dense":
            dense_layers.append(core)

    if len(dense_layers) != len(dense_params):
        raise RuntimeError(
            f"Mismatch between parsed Dense layers ({len(dense_layers)}) and parameters ({len(dense_params)})."
        )

    for core, (W, b) in zip(dense_layers, dense_params):
        # Ensure shapes are initialized (after summary). Then assign.
        core.W.data[...] = W.astype(np.float32, copy=False)
        core.b.data[...] = b.astype(np.float32, copy=False)

    return seq
