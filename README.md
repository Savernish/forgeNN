# forgeNN
*A High-Performance Automatic Differentiation Framework for Deep Learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Compatible](https://img.shields.io/badge/API-PyTorch_Compatible-orange.svg)](https://pytorch.org/)

## 🚀 Overview

**forgeNN** is a production-ready deep learning framework featuring efficient automatic differentiation, dynamic computation graphs, and optimized neural network components. Designed for simplicity and educational value.

### Key Features

- **🔥 Dynamic Computation Graphs**: Build and modify networks on-the-fly
- **⚡ Efficient Automatic Differentiation**: Reverse-mode AD with topological sorting
- **🧠 Comprehensive Neural Networks**: From neurons to multi-layer perceptrons
- **🎯 Production-Ready Loss Functions**: MSE, Cross-Entropy with numerical stability
- **🚀 Modern Activations**: ReLU, Leaky ReLU, Tanh, Sigmoid, Swish
- ** Research-Friendly**: Transparent implementations for educational use

## ⚡ Quick Start

### Linear Regression

```python
from forgeNN.core import Value
from forgeNN.network import MLP
from sklearn.datasets import make_regression

# Generate dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Create model
model = MLP(1, [1], ['linear'])

# Training loop
learning_rate = 0.01
for epoch in range(100):
    total_loss = 0
    
    for i in range(len(X)):
        # Reset gradients
        for param in model.parameters():
            param.grad = 0
            
        # Forward pass
        pred = model([Value(X[i, 0])])
        loss = pred.mse(y[i])
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        for param in model.parameters():
            param.data -= learning_rate * param.grad
        
        total_loss += loss.data
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss/len(X):.6f}")
```

### MNIST Classification Example

Check out `mnistdemo.py` for a complete neural network training on the MNIST dataset!

## 🏗️ Core Components

```python
from forgeNN.core import Value
from forgeNN.network import MLP

# Automatic differentiation
x = Value(2.0)
y = Value(3.0)  
z = x * y + x**2
z.backward()
print(x.grad)  # ∂z/∂x = 7.0

# Neural networks
model = MLP(784, [128, 64, 10])  # MNIST classifier
prediction = model(data)
```

## 🤝 Contributing

We welcome contributions! 

```bash
git clone https://github.com/Savernish/forgeNN.git
cd forgeNN
```

## 🌟 Acknowledgments

- Inspired by Andrej Karpathy's micrograd tutorial
- Built with educational clarity in mind
- Thanks to the open-source ML community

---

**Ready to forge the future of neural networks?** 🚀