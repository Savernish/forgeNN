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

## Overview

**forgeNN** is a modern neural network framework that is developed by a solo developer learning about ML. Features vectorized operations for high-speed training.

### Key Features

- **Vectorized Operations**: NumPy-powered batch processing (100x+ speedup)
- **Dynamic Computation Graphs**: Automatic differentiation with gradient tracking
- **Complete Neural Networks**: From simple neurons to complex architectures
- **Production Loss Functions**: Cross-entropy, MSE with numerical stability

## Performance vs PyTorch

**forgeNN is 3.52x faster than PyTorch on small models!**

| Metric | PyTorch | forgeNN | Advantage |
|--------|---------|---------|-----------|
| Training Time (MNIST) | 64.72s | 30.84s | **2.10x faster** |
| Test Accuracy | 97.30% | 97.37% | **+0.07% better** |
| Small Models (<109k params) | Baseline | **3.52x faster** | **Massive speedup** |

📊 **[See Full Comparison Guide](guides/COMPARISON_GUIDE.md)** for detailed benchmarks, syntax differences, and when to use each framework. *Note: single-machine indicative results; not statistically rigorous multi-run averages.*


## Quick Start

### High-Performance Training

```python
import forgeNN
from sklearn.datasets import make_classification

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)

# Create vectorized model  
model = forgeNN.VectorizedMLP(20, [64, 32], 3)
optimizer = forgeNN.VectorizedOptimizer(model.parameters(), lr=0.01)

# Fast batch training
for epoch in range(10):
    # Convert to tensors
    x_batch = forgeNN.Tensor(X)
    
    # Forward pass
    logits = model(x_batch)
    loss = forgeNN.cross_entropy_loss(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    acc = forgeNN.accuracy(logits, y)
    print(f"Epoch {epoch}: Loss = {loss.data:.4f}, Acc = {acc*100:.1f}%")
```

### Keras-like Training (compile/fit)

```python
import forgeNN as fnn

model = fnn.Sequential([
    fnn.Input((20,)),        # optional Input layer seeds summary & shapes
    fnn.Dense(64) @ 'relu',
    fnn.Dense(32) @ 'relu',
    fnn.Dense(3)  @ 'linear'
])

# Optionally inspect architecture
model.summary()              # or model.summary((20,)) if no Input layer

compiled = fnn.compile(model, optimizer={"lr": 0.01, "momentum": 0.9},
                  loss='cross_entropy', metrics=['accuracy'])
compiled.fit(X, y, epochs=10, batch_size=64)
loss, metrics = compiled.evaluate(X, y)

# Tip: `mse` auto-detects 1D integer class labels for (N,C) logits and one-hot encodes internally.
# model.summary() can be called any time after construction if an Input layer or input_shape is provided.
```

## Architecture

- **Main API**: `forgeNN`, `forgeNN.Tensor`, `forgeNN.Sequential`, `forgeNN.Input`, `forgeNN.VectorizedMLP`
- **Model Introspection**: `model.summary()` (Keras-like) with symbolic shape + parameter counts
- **Examples**: Check `examples/` for MNIST and more

## Performance

| Implementation | Speed | MNIST Accuracy |
|---------------|-------|----------------|
| Vectorized | 40,000+ samples/sec | 95%+ in <1s |
| Sequential (with compile/fit) | 40,000+ samples/sec | 95%+ in <1.2s |

**Highlights**:
- **100x+ speedup** over scalar implementations
- **Production-ready** performance with educational clarity
- **Memory efficient** vectorized operations
- **Smarter Losses**: `mse` auto one-hot & reshape logic; fused stable cross-entropy

## Complete Example

See `examples/` for full fledged demos

## Links

- **PyPI Package**: https://pypi.org/project/forgeNN/
- **Documentation**: See guides in this repository
- **Guides**: SEQUENTIAL_GUIDE.md, TRAINING_GUIDE.md, COMPARISON_GUIDE.md
- **Issues**: GitHub Issues for bug reports and feature requests

## Roadmap
### Before 2026 (2025 Remaining Milestones – ordered)
1. Adam / AdamW
2. Model saving & loading (state dict + `.npz`)
3. Dropout + LayerNorm
4. Tiny Transformer example (encoder-only)
5. ONNX export (Sequential/Dense/Flatten/activations) then basic import (subset)
6. Conv1D → Conv2D (naive)
7. Documentation: serialization guide, ONNX guide, Transformer walkthrough
8. Parameter registry refinement

### Q1 2026 (Early 2026 Targets)
- CUDA / GPU backend prototype (Tensor device abstraction)
- Formal architecture & design documents (graph execution, autograd internals)
- Expanded documentation site (narrative design + performance notes)

_Items above may be reprioritized based on user feedback; GPU & design docs explicitly deferred to early 2026._

## Contributing

I am not currently accepting contributions, but I'm always open to suggestions and feedback!

## Acknowledgments

- Inspired by educational automatic differentiation tutorials (micrograd)
- Built for both learning and production use
- Optimized with modern NumPy practices
- **Available on PyPI**: `pip install forgeNN`

---
