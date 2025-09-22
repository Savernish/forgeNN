# forgeNN

## Table of Contents

- [Installation](#Installation)
- [Overview](#Overview)
- [Performance vs PyTorch](#Performance-vs-PyTorch)
- [Quick Start](#Quick-Start)
- [Complete Example](#Complete-Example)
- [Roadmap](#Roadmap)
- [Contributing](#Contributing)
- [Acknowledgments](#Acknowledgments)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/Savernish/forgeNN.svg?style=social&label=Stars)](https://github.com/Savernish/forgeNN)
[![NumPy](https://img.shields.io/badge/powered_by-NumPy-blue.svg)](https://numpy.org/)
[![PyPI version](https://img.shields.io/pypi/v/forgeNN.svg)](https://pypi.org/project/forgeNN/)
[![Downloads](https://img.shields.io/pypi/dm/forgeNN.svg)](https://pypi.org/project/forgeNN/)
[![License](https://img.shields.io/pypi/l/forgeNN.svg)](https://pypi.org/project/forgeNN/)

## Installation

```bash
pip install forgeNN
```

Optional extras:

```bash
# ONNX helpers (scaffold)
pip install "forgeNN[onnx]"

# CUDA backend (scaffold; requires compatible GPU/driver)
pip install "forgeNN[cuda]"
```

## Overview

**forgeNN** is a modern neural network framework with an API built around a straightforward `Sequential` model, a fast NumPy autograd `Tensor`, and a Keras-like `compile/fit` training workflow.

This project is built and maintained by a single student developer. For background and portfolio/CV, see: https://savern.me

### Key Features

- **Fast NumPy core**: Vectorized operations with fused, stable math
- **Dynamic Computation Graphs**: Automatic differentiation with gradient tracking
- **Complete Neural Networks**: From simple neurons to complex architectures
- **Production Loss Functions**: Cross-entropy, MSE with numerical stability
 - **Scaffolded Integrations**: Runtime device API for future CUDA; ONNX export/import stubs

## Performance vs PyTorch

**forgeNN is 3.52x faster than PyTorch on small models!**

| Metric | PyTorch | forgeNN | Advantage |
|--------|---------|---------|-----------|
| Training Time (MNIST) | 64.72s | 30.84s | **2.10x faster** |
| Test Accuracy | 97.30% | 97.37% | **+0.07% better** |
| Small Models (<109k params) | Baseline | **3.52x faster** | **Massive speedup** |

📊 Comparison and detailed docs are being refreshed for v2; see examples/ for runnable demos.


## Quick Start

### Keras-like Training (compile/fit)

```python
model = fnn.Sequential([
    fnn.Input((20,)),        # optional Input layer seeds summary & shapes
    fnn.Dense(64) @ 'relu',
    fnn.Dense(32) @ 'relu',
    fnn.Dense(3)  @ 'linear'
])

# Optionally inspect architecture
model.summary()              # or model.summary((20,)) if no Input layer
opt = fnn.Adam(lr=1e-3)      # or other optimizers (adamw, sgd, etc)
compiled = fnn.compile(model,
                    optimizer=opt,
                    loss='cross_entropy',
                    metrics=['accuracy'])
compiled.fit(X, y, epochs=10, batch_size=64)
loss, metrics = compiled.evaluate(X, y)

# Tip: `mse` auto-detects 1D integer class labels for (N,C) logits and one-hot encodes internally.
# model.summary() can be called any time after construction if an Input layer or input_shape is provided.
```



## Complete Example

See `examples/` for full fledged demos

## Links

- **PyPI Package**: https://pypi.org/project/forgeNN/
- **Documentation**: v2 guides coming soon; examples in `examples/`
- **Issues**: GitHub Issues for bug reports and feature requests
- **Portfolio/CV**: https://savern.me

## Roadmap (post v2.0.0)

- CUDA backend and device runtime
  - Device abstraction for `Tensor` and layers
  - Initial CUDA kernels (Conv, GEMM, elementwise) and CPU/CUDA parity tests
  - Setup and troubleshooting guide

- ONNX: export and import (full coverage for the core API)
  - Export `Sequential` graphs with Conv/Pool/Flatten/Dense/LayerNorm/Dropout/activations
  - Import linear and branched graphs where feasible; shape inference checks
  - Round‑trip parity tests and examples

- Model save and load
  - Architecture JSON + weights (NPZ) format
  - `state_dict`/`load_state_dict` compatibility helpers
  - Versioning and minimal migration guidance

- Transformer positional encodings
  - Sinusoidal `PositionalEncoding` and learnable `PositionalEmbedding`
  - Tiny encoder demo with text classification walkthrough

- Performance and stability
  - CPU optimizations for conv/pool paths, memory reuse, and fewer allocations
  - Threading guidance (MKL/OpenBLAS), deterministic runs, and profiling notes

- Documentation
  - Practical guides for `Sequential`, `compile/fit`, model I/O, ONNX, and CUDA setup
  - Design overview of autograd and execution model

## Contributing

I am not currently accepting contributions, but I'm always open to suggestions and feedback!

## Acknowledgments

- Inspired by educational automatic differentiation tutorials (micrograd)
- Built for both learning and production use
- Optimized with modern NumPy practices
- **Available on PyPI**: `pip install forgeNN`

---
