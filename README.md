# forgeNN

## Table of Contents

- [Installation](#Installation)
- [Overview](#Overview)
- [Performance vs PyTorch](#Performance-vs-PyTorch)
- [Quick Start](#Quick-Start)
- [Architecture](#Architecture)
- [Performance](#Performance)
- [Complete Example](#Complete-Example)
- [TODO List](#TODO-List)
- [Contributing](#Contributing)
- [Acknowledgments](#Acknowledgments)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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

📊 **[See Full Comparison Guide](COMPARISON_GUIDE.md)** for detailed benchmarks, syntax differences, and when to use each framework.

![MNIST Benchmark Results](mnist_benchmark_comparison.png)

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
   fnn.Dense(64) @ 'relu',
   fnn.Dense(32) @ 'relu',
   fnn.Dense(3)  @ 'linear'
])

# Initialize lazy params if needed
_ = model(fnn.Tensor([[0.0]*20]))

compiled = fnn.compile(model, optimizer={"lr": 0.01, "momentum": 0.9},
                  loss='cross_entropy', metrics=['accuracy'])
compiled.fit(X, y, epochs=10, batch_size=64)
loss, metrics = compiled.evaluate(X, y)
```

## Architecture

- **Main API**: `forgeNN.Tensor`, `forgeNN.VectorizedMLP` (high-performance neural networks)
- **Activation Functions**: `forgeNN.RELU`, `forgeNN.SWISH`, etc. + string/callable support
- **Examples**: `example.py` - Complete MNIST classification demo

## Performance

| Implementation | Speed | MNIST Accuracy |
|---------------|-------|----------------|
| Vectorized | 38,000+ samples/sec | 93%+ in <2s |

**Highlights**:
- **100x+ speedup** over scalar implementations
- **Production-ready** performance with educational clarity
- **Memory efficient** vectorized operations

## Complete Example

See `example.py` for a full MNIST classification demo achieving professional results.

## Links

- **PyPI Package**: https://pypi.org/project/forgeNN/
- **Documentation**: See guides in this repository
- **Guides**: SEQUENTIAL_GUIDE.md, TRAINING_GUIDE.md, COMPARISON_GUIDE.md
- **Issues**: GitHub Issues for bug reports and feature requests

## TODO List

Based on comprehensive comparison with PyTorch and NumPy:

### CRITICAL MISSING FEATURES (High Priority):

1. ~TENSOR SHAPE OPERATIONS:~ - COMPLETED
   - `reshape()`      : Change tensor dimensions (tensor.reshape(2, -1)) - COMPLETED
   - `transpose()`    : Swap dimensions (tensor.transpose(0, 1)) - COMPLETED
   - `view()`         : Memory-efficient reshape (tensor.view(-1, 5)) - COMPLETED
   - `flatten()`      : Convert to 1D (tensor.flatten()) - COMPLETED
   - `squeeze()`      : Remove size-1 dims (tensor.squeeze()) - COMPLETED
   - `unsqueeze()`    : Add size-1 dims (tensor.unsqueeze(0)) - COMPLETED

2. ~MATRIX OPERATIONS:~ - COMPLETED
   - `matmul()` / `@`  : Matrix multiplication with broadcasting - COMPLETED
   - `dot()`          : Vector dot product for 1D arrays - COMPLETED

3. TENSOR COMBINATION:
   - `cat()`          : Join along existing dim (torch.cat([a, b], dim=0))
   - `stack()`        : Join along new dim (torch.stack([a, b]))

### IMPORTANT FEATURES (Medium Priority):

4. ADVANCED ACTIVATIONS:
   - `lrelu()`       : AVAILABLE as `forgeNN.functions.activation.LRELU` (needs fixing) - FIXED
   - `swish()`       : AVAILABLE as `forgeNN.functions.activation.SWISH` (needs fixing) - FIXED
   - `gelu()`         : Gaussian Error Linear Unit (missing) - ADDED
   - `elu()`          : Exponential Linear Unit (missing)

5. TENSOR UTILITIES:
   - `split()`        : Split into chunks
   - `chunk()`        : Split into equal pieces
   - `permute()`      : Rearrange dimensions
   - `contiguous()` : Make tensor memory-contiguous (tensor.contiguous()) - COMPLETED

6. INDEXING:
   - Boolean indexing: `tensor[tensor > 0]`
   - Fancy indexing: `tensor[indices]`
   - `gather()`       : Select along dimension

### NICE-TO-HAVE (Lower Priority):

7. LINEAR ALGEBRA:
   - `norm()`         : Vector/matrix norms
   - `det()`          : Matrix determinant
   - `inverse()`      : Matrix inverse

8. CONVENIENCE:
   - `clone()`        : Deep copy
   - `detach()`       : Remove from computation graph
   - `requires_grad_()`: In-place grad requirement change

9. INFRASTRUCTURE:
   - Better error messages for shape mismatches
   - Memory-efficient operations
   - API consistency improvements
   - Comprehensive documentation

### PRIORITY ORDER:
1. Shape operations (reshape, transpose, flatten)
2. Matrix multiplication (matmul, @)  
3. Tensor combination (cat, stack)
4. More activations (leaky_relu, gelu)
5. Documentation and error handling

## Contributing

I am not currently accepting contributions, but I'm always open to suggestions and feedback!

## Acknowledgments

- Inspired by educational automatic differentiation tutorials (micrograd)
- Built for both learning and production use
- Optimized with modern NumPy practices
- **Available on PyPI**: `pip install forgeNN`

---
