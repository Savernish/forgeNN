# forgeNN
*A From Scratch Neural Network Framework with Educational Purposes*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/powered_by-NumPy-blue.svg)](https://numpy.org/)

##  Overview

**forgeNN** is a modern neural network framework that is developed by a solo developer learning about ML. Features vectorized operations for high-speed training.

### Key Features

- ** Vectorized Operations**: NumPy-powered batch processing (100x+ speedup)
- ** Dynamic Computation Graphs**: Automatic differentiation with gradient tracking
- ** Complete Neural Networks**: From simple neurons to complex architectures
- ** Production Loss Functions**: Cross-entropy, MSE with numerical stability

##  Quick Start

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

### Legacy Examples

```python
# For learning automatic differentiation
from forgeNN.legacy import Value, MLP

x = Value(2.0)
y = x**2 + 3*x + 1
y.backward()
print(f"dy/dx = {x.grad}")  # 7.0

# Simple neural network
model = MLP(2, [4, 1])
output = model([Value(1.0), Value(2.0)])
```

##  Architecture

- **Main API**: `forgeNN.Tensor`, `forgeNN.VectorizedMLP` (production use)
- **Legacy API**: `forgeNN.legacy.*` (backward compatible)
- **Functions**: Complete activation and loss function library
- **Examples**: `example.py` - Complete MNIST classification demo

##  Performance

| Implementation | Speed |
|---------------|-------|
| Vectorized | 38,000+ samples/sec | 


**MNIST Results**: 93%+ accuracy in under 2 seconds!

##  Complete Example

See `example.py` for a full MNIST classification demo achieving professional results.

##  Contributing

I am not currently accepting contributions, but I'm always open to suggestions and feedback!

## 📦 PyPI Deployment

This package is ready for PyPI deployment! All deployment files are organized in the `deployment/` directory to keep the main project clean.

```bash
# Build and deploy to PyPI
cd deployment
python build_package.py
python upload_package.py
```

For detailed deployment instructions, see `deployment/DEPLOY.md`.

## 🌟 Acknowledgments

- Inspired by educational automatic differentiation tutorials
- Built for both learning and production use
- Optimized with modern NumPy practices

---
