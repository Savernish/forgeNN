# forgeNN Sequential: A new approach to Composable Layers

This guide introduces the Sequential API and its activation wiring with the @ operator. It shows how to build models in forgeNN and compares the syntax with PyTorch and TensorFlow/Keras.

## What Sequential Does

- Container that applies layers in order: output = L_n(...L_2(L_1(x)))
- Works with forgeNN layers: Dense, Flatten, and any custom Layer subclass
- Attaches activations to layers using the @ operator: Dense(64) @ 'relu'
- Aggregates parameters() across all child layers for optimizers

Core classes:
- Layer: base with forward(x), __call__(x) and __matmul__ for activation attachment
- ActivationWrapper: wraps a Layer to apply an activation after forward
- Sequential: ordered container of layers and wrappers
- Dense: fully-connected layer with lazy input-dimension inference
- Flatten: flattens to (batch, -1)

## Quick Start

```python
import numpy as np
import forgeNN as fnn

# Build a simple MLP: 4 -> 8 -> 2
model = fnn.Sequential([
    fnn.Dense(8) @ 'relu',   # attach ReLU via @ operator
    fnn.Dense(2) @ 'linear'
])

# Forward, loss, backward, optimize
x = fnn.Tensor(np.random.randn(5, 4).astype(np.float32))
logits = model(x)

y = np.array([0, 1, 0, 1, 0])  # integer class targets
loss = fnn.cross_entropy_loss(logits, y)

opt = fnn.VectorizedOptimizer(model.parameters(), lr=0.05, momentum=0.9)
opt.zero_grad()
loss.backward()
opt.step()
```

Tip on lazy initialization: Dense infers in_features on first forward. If you create your optimizer before the first forward, either pass in_features explicitly (Dense(out, in_features=...)) or run a single dummy forward to initialize parameters.

## Syntax Comparison: PyTorch vs TensorFlow/Keras vs forgeNN

### Build an MLP with two hidden layers and ReLU

PyTorch
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64),  nn.ReLU(),
    nn.Linear(64, 10)    # logits
)
```

TensorFlow/Keras
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation=None)   # logits
])
```

forgeNN
```python
import forgeNN as fnn

model = fnn.Sequential([
    fnn.Dense(128) @ 'relu',
    fnn.Dense(64)  @ 'relu',
    fnn.Dense(10)  @ 'linear'  # logits
])
```

Key differences:
- PyTorch: activation is a separate layer; Dense/Linear has no activation arg
- Keras: activation specified on the layer, string or callable
- forgeNN: attach activation post-layer with @; activation can be a string, class, or callable

### Activations: strings, classes, or callables

forgeNN unified activation system accepts:
- Strings: 'relu', 'sigmoid', 'tanh', 'linear', 'lrelu', 'swish'
- Classes: RELU, LRELU, TANH, SIGMOID, SWISH (instances or types)
- Callables: any function f(Tensor) -> Tensor

Examples
```python
from forgeNN.functions.activation import RELU, SWISH
import forgeNN as fnn

# Strings (recommended)
model = fnn.Sequential([
    fnn.Dense(64) @ 'relu',
    fnn.Dense(10) @ 'linear'
])

# Activation classes
model = fnn.Sequential([
    fnn.Dense(64) @ RELU(),
    fnn.Dense(10)
])

# Callables (custom parameters)
model = fnn.Sequential([
    fnn.Dense(64) @ (lambda x: x.swish(beta=1.5)),
    fnn.Dense(10)
])
```

### Flatten and mixed shapes

```python
import numpy as np
import forgeNN as fnn

model = fnn.Sequential([
    fnn.Flatten(),         # (N, H, W, C) -> (N, H*W*C)
    fnn.Dense(32) @ 'relu',
    fnn.Dense(10)
])

x = fnn.Tensor(np.random.randn(8, 2, 3).astype(np.float32))
logits = model(x)  # (8, 10)
```

## Training Loop Comparison

PyTorch
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

optimizer.zero_grad()
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

Keras
```python
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

forgeNN
```python
opt = fnn.VectorizedOptimizer(model.parameters(), lr=0.01)
logits = model(fnn.Tensor(batch_x))
loss = fnn.cross_entropy_loss(logits, batch_y)
opt.zero_grad()
loss.backward()
opt.step()
```

## Training with compile/fit

If you prefer a Keras-like workflow, use the lightweight training wrapper:

```python
import forgeNN as fnn

model = fnn.Sequential([
    fnn.Dense(128) @ 'relu',
    fnn.Dense(64)  @ 'relu',
    fnn.Dense(10)  @ 'linear'
])

# If using lazy Dense, run a dummy forward once to initialize parameters
_ = model(fnn.Tensor([[0.0]*784]))

compiled = fnn.compile(
    model,
    optimizer={"lr": 0.01, "momentum": 0.9},
    loss='cross_entropy',
    metrics=['accuracy']
)

compiled.fit(X_train, y_train, epochs=10, batch_size=64,
             validation_data=(X_test, y_test))

test_loss, test_metrics = compiled.evaluate(X_test, y_test)
pred_logits = compiled.predict(X_test[:8])
```

Mapping for familiarity:
- Keras: `model.compile(...)` → forgeNN: `fnn.compile(model, ...)`
- Keras: `model.fit(...)` → forgeNN: `compiled.fit(...)`
- Keras: `model.evaluate(...)` → forgeNN: `compiled.evaluate(...)`
- Keras: `model.predict(...)` → forgeNN: `compiled.predict(...)`

PyTorch users can skip writing the manual loop by using `compiled.fit`.


## Design Notes

- Activation attachment with @ is explicit and local: you see which activation follows a particular layer
- Lazy initialization in Dense keeps constructor minimal for readability; specify in_features to disable laziness
- Sequential.parameters() returns trainable tensors ready for VectorizedOptimizer
- Works seamlessly with Tensor’s autodiff operations implemented in forgeNN

## Migration Recipes

From PyTorch
```python
# PyTorch
nn.Sequential(nn.Linear(20, 16), nn.ReLU(), nn.Linear(16, 4))

# forgeNN
fnn.Sequential([fnn.Dense(16) @ 'relu', fnn.Dense(4)])
```

From Keras
```python
# Keras
keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(20,)),
    keras.layers.Dense(4)
])

# forgeNN
fnn.Sequential([fnn.Dense(16) @ 'relu', fnn.Dense(4)])
```

## Testing and Reliability

The repository includes unit tests that cover:
- Dense lazy initialization and forward
- ActivationWrapper with strings, classes, and callables
- Sequential parameter aggregation and zero_grad
- Optimizer interactions, including momentum buffers


## When to Use Sequential

- Small to medium MLPs and feature heads
- Clean, readable architecture definitions
- Educational settings where clarity and explicitness matter

For more context on forgeNN’s API philosophy and performance trade-offs, see `COMPARISON_GUIDE.md`.
