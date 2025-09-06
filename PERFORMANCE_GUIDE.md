# Performance Optimization Guide

## Overview

forgeNN now includes two implementations:
- **Scalar Implementation** (`core.py`, `network.py`): Educational and transparent
- **Vectorized Implementation** (`tensor.py`, `vectorized.py`): High-performance production use

## Performance Comparison

| Metric | Scalar | Vectorized | Improvement |
|--------|--------|------------|-------------|
| Training Speed | ~400 samples/sec | 38,000+ samples/sec | **~100x faster** |
| Memory Usage | High (Python objects) | Low (NumPy arrays) | **~10x reduction** |
| Batch Support | No | Yes | **Enable batch training** |
| GPU Ready | No | NumPy compatible | **Ready for GPU acceleration** |

## Usage Guide

### Scalar Implementation (Educational)
```python
from forgeNN.core import Value
from forgeNN.network import MLP

# Sample-by-sample training
model = MLP(784, [128, 64, 10])
for x, y in training_data:
    x_vals = [Value(float(xi)) for xi in x]
    output = model(x_vals)
    # ... training loop
```

### Vectorized Implementation (Production)
```python
from forgeNN.tensor import Tensor
from forgeNN.vectorized import VectorizedMLP, VectorizedOptimizer

# Batch training
model = VectorizedMLP(784, [128, 64], 10)
optimizer = VectorizedOptimizer(model.parameters(), lr=0.01)

for batch_x, batch_y in data_loader:
    x_tensor = Tensor(batch_x)
    logits = model(x_tensor)
    loss = cross_entropy_loss(logits, batch_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Real-World Results

**MNIST Classification (5000 samples, 10 epochs):**
- **Vectorized**: 93.3% accuracy in 1.0 second
- **Scalar**: Would take ~100 seconds for the same task

**Production Benefits:**
- Train larger models (100K+ parameters efficiently)
- Handle larger datasets (10K+ samples practical)
- Enable real-time applications
- Reduce cloud computing costs by 100x

## When to Use Each

### Use Scalar Implementation For:
- Learning how neural networks work
- Understanding automatic differentiation
- Debugging gradient computations
- Educational projects
- Small experiments (<100 samples)

### Use Vectorized Implementation For:
- Production applications
- Large datasets (>1000 samples)
- Model experimentation
- Performance benchmarking
- Real-world machine learning projects

## Migration Path

The APIs are designed to be similar. To migrate from scalar to vectorized:

1. Replace `Value` with `Tensor` 
2. Replace `MLP` with `VectorizedMLP`
3. Add batch processing to your training loop
4. Use `VectorizedOptimizer` for parameter updates

## Future Enhancements

- GPU acceleration with CuPy
- Mixed precision training
- Distributed training support
- More optimizer algorithms (Adam, RMSprop)
- Advanced regularization techniques
