"""forgeNN.nn public APIs.

This subpackage exposes commonly used neural-network utilities and helpers,
including RNG helpers in ``forgeNN.nn.random``. We import the random module
eagerly so that ``fnn.nn.set_seed`` works out-of-the-box after
``import forgeNN as fnn``.
"""

# Eagerly import submodules that provide public functions we want at package level
from . import random as random  # noqa: F401

# Re-export RNG helpers at nn package level
from .random import set_seed, get_rng_state, set_rng_state  # noqa: F401

__all__ = [
	"set_seed", "get_rng_state", "set_rng_state",
	"random",
]
