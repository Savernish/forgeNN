"""Random utilities for reproducibility.

These helpers manage global RNGs for NumPy and Python's ``random`` only.
They intentionally avoid importing the top-level ``forgeNN`` package to
prevent circular imports and because a framework-global RNG is not required
for core functionality.
"""

import numpy as np
import random


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Args:
        seed: Random seed to use.

    This affects ``numpy.random`` and Python's ``random`` module.
    """
    np.random.seed(seed)
    random.seed(seed)


def get_rng_state() -> dict:
    """Return current RNG states for NumPy and Python's random.

    Returns a dictionary with keys ``"numpy"`` and ``"random"`` that can be
    passed back to :func:`set_rng_state`.
    """
    return {
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    """Restore RNG states previously captured by :func:`get_rng_state`."""
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])


