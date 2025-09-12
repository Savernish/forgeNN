# forgeNN

## Table of Contents

- [Installation](#Installation)
- [Overview](#Overview)
- [Performance vs PyTorch](#Performance-vs-PyTorch)
- [Quick Start](#Quick-Start)
- [Architecture](#Architecture)
- [Performance](#Performance)
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

**forgeNN** is a modern neural network framework with a lean v2 API focused on a clean Sequential model, fast NumPy autograd Tensor, and a Keras-like compile/fit workflow.

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

### High-Performance Training

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import forgeNN as fnn

X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=24)
X = StandardScaler().fit_transform(X).astype(np.float32)

model = fnn.Sequential([
    fnn.Input((20,)),
    fnn.Dense(64) @ 'relu',
    fnn.Dense(32) @ 'relu',
    fnn.Dense(3)
])
compiled = fnn.compile(model, optimizer={"type": "adam", "lr": 1e-3, "eps": 1e-7}, loss='cross_entropy', metrics=['accuracy'])
compiled.fit(X, y, epochs=10, batch_size=64)
loss, metrics = compiled.evaluate(X, y)
print('acc', metrics['accuracy'])
```

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

## Architecture

- **Main API**: `forgeNN.Tensor`, `forgeNN.Sequential`, `forgeNN.compile`, optimizers (`SGD`, `Adam`, `AdamW`)

## Performance

| Implementation | Speed | MNIST Accuracy |
|---------------|-------|----------------|
| Sequential (compile/fit) | 40,000+ samples/sec | 95%+ in ~1s |

**Highlights**:
- **100x+ speedup** over scalar implementations
- **Production-ready** performance with educational clarity
- **Memory efficient** vectorized operations
- **Smarter Losses**: `mse` auto one-hot & reshape logic; fused stable cross-entropy

## Complete Example

See `examples/` for full fledged demos

## Links

- **PyPI Package**: https://pypi.org/project/forgeNN/
- **Documentation**: v2 guides coming soon; examples in `examples/`
- **Issues**: GitHub Issues for bug reports and feature requests

## Roadmap
### Before 2026 (2025 Remaining Milestones – ordered)
1. ~Adam / AdamW~ 🗹 (Completed in v1.3.0) 
2. ~Dropout + LayerNorm~ 🗹 (Completed in v1.3.0)
3. Model saving & loading (state dict + `.npz`) ☐
4. Conv1D → Conv2D (naive) ☐
5. Add missing tensor ops to fully support examples ☐
5. Tiny Transformer example (encoder-only) ☐
6. ~ONNX export (Sequential/Dense/Flatten/activations)~ 🗹 (Completed in v2.0.0)
7. ~ONNX import (subset)~ 🗹 (Completed in v2.0.0)
8. Basic CUDA backend (Tensor device abstraction) ☐
9. Documentation: serialization guide, ONNX guide, Transformer walkthrough ☐
10. Parameter registry refinement ☐
11. CUDA / GPU backend prototype (Tensor device abstraction) ☐

### Q1 2026 (Early 2026 Targets)
- Formal architecture & design documents (graph execution, autograd internals)
- Expanded documentation site (narrative design + performance notes)

_Items above may be reprioritized based on user feedback; design docs explicitly deferred to early 2026._

## Contributing

I am not currently accepting contributions, but I'm always open to suggestions and feedback!

## Acknowledgments

- Inspired by educational automatic differentiation tutorials (micrograd)
- Built for both learning and production use
- Optimized with modern NumPy practices
- **Available on PyPI**: `pip install forgeNN`

---
