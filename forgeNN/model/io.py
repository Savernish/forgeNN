from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np

from ..core.tensor import Tensor


def _collect_named_parameters(layer: Any, prefix: str = "") -> List[Tuple[str, Tensor]]:
    """Recursively collect named parameters from a layer or Sequential.

    Convention:
      - For Sequential, names are index-based: "layers.{i}.<name>"
      - For simple layers, common param names: W, b, gamma, beta, etc.
      - ActivationWrapper contributes params from its inner layer.
    """
    named: List[Tuple[str, Tensor]] = []

    # ActivationWrapper: unwrap to inner
    if getattr(layer, "__class__", None) and layer.__class__.__name__ == "ActivationWrapper":
        return _collect_named_parameters(layer.layer, prefix)

    # Sequential: traverse children with indices
    if getattr(layer, "__class__", None) and layer.__class__.__name__ == "Sequential":
        for i, sub in enumerate(layer.layers):
            child_prefix = f"{prefix}layers.{i}." if prefix else f"layers.{i}."
            named.extend(_collect_named_parameters(sub, child_prefix))
        return named

    # Generic Layer: inspect common attributes exposed by parameters()
    params = getattr(layer, "parameters", None)
    if callable(params):
        plist = list(layer.parameters())  # ensure stable order
        # Heuristic names: try typical attribute names, else param{i}
        attr_candidates = ["W", "b", "gamma", "beta"]
        used: set[int] = set()
        for name in attr_candidates:
            t = getattr(layer, name, None)
            if isinstance(t, Tensor) and t in plist:
                idx = plist.index(t)
                used.add(idx)
                key = f"{prefix}{name}"
                named.append((key, t))
        for i, t in enumerate(plist):
            if i in used:
                continue
            key = f"{prefix}param{i}"
            named.append((key, t))
    return named


def state_dict(model: Any) -> Dict[str, np.ndarray]:
    """Return a flat mapping of parameter names to NumPy arrays.

    Example keys for a 2-layer Sequential: layers.0.W, layers.0.b, layers.1.W, layers.1.b
    """
    sd: Dict[str, np.ndarray] = {}
    for name, t in _collect_named_parameters(model, prefix=""):
        sd[name] = np.array(t.data, copy=True)
    return sd


def load_state_dict(model: Any, state: Dict[str, np.ndarray]) -> None:
    """Load parameters from a state dict into the model in-place.

    Only keys that exist in both the model and state will be loaded. Shapes must match.
    """
    named = dict(_collect_named_parameters(model, prefix=""))
    for k, arr in state.items():
        if k not in named:
            continue
        t = named[k]
        if t.data.shape != arr.shape:
            raise ValueError(f"Shape mismatch for {k}: model {t.data.shape}, state {arr.shape}")
        t.data[...] = arr.astype(t.data.dtype, copy=False)


def save_npz(path: str, model: Any) -> None:
    """Save model parameters to a .npz archive."""
    sd = state_dict(model)
    np.savez(path, **sd)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a state dict from a .npz file without applying to a model."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


