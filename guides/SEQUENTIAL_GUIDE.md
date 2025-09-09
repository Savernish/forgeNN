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
    fnn.Input((4,)),         # optional: seeds summary + shape inference
    fnn.Dense(8) @ 'relu',
    fnn.Dense(2)             # logits (linear)
])

x = fnn.Tensor(np.random.randn(5, 4).astype(np.float32))
logits = model(x)
y = np.array([0, 1, 0, 1, 0])
loss = fnn.cross_entropy_loss(logits, y)

# Optimizer options:
# 1. Raw loop with SGD
opt = fnn.SGD(model.parameters(), lr=0.05, momentum=0.9)
opt.zero_grad(); loss.backward(); opt.step()

# 2. Or Adam (adaptive)
# opt = fnn.Adam(model.parameters(), lr=1e-3)
# opt.zero_grad(); loss.backward(); opt.step()

# 3. Or deferred instance (no params yet) + compile
# opt = fnn.Adam(lr=1e-3)        # no params bound yet
# compiled = fnn.compile(model, optimizer=opt, loss='cross_entropy', metrics=['accuracy'])
# compiled.fit(train_X, train_y, epochs=5)
```

Tip on lazy initialization: `Dense` still infers `in_features` on the first real forward if not explicitly provided. `model.summary()` will proactively initialize a `Dense` layer only when the incoming feature dimension is resolvable (via an `Input` layer or explicit `input_shape`). With the new deferred optimizers you can now create `opt = fnn.Adam(lr=1e-3)` *before* the model has built parameters—binding happens automatically inside `compile()` or on first `optimizer` access. If you need parameters materialized early (e.g., to inspect weights), either:
1. Provide `in_features` to `Dense`, or
2. Include an `Input` layer and call `model.summary()`, or
3. Run a dummy forward once.

### Model Introspection

You can now inspect architecture details with a Keras-like summary:

```python
model.summary()            # uses Input layer if present
model.summary((4,))        # or pass input shape explicitly (excluding batch)
```

Output example:

```
============================================================
Layer (type)                  Output Shape                Param #
============================================================
Input                         (None, 4)                         0
Dense(relu)                   (None, 8)                        40
Dense(linear)                 (None, 2)                        18
============================================================
Total params: 58
Total parameter tensors: 4
============================================================
```

## Syntax Comparison: PyTorch vs TensorFlow/Keras vs forgeNN

### Build an MLP with two hidden layers and ReLU

PyTorch
```python
import torch
import torch.nn as nn

model = fnn.Sequential([
    fnn.Input((784,)),             # optional but enables immediate shape inference
    fnn.Dense(128) @ 'relu',
    fnn.Dense(64)  @ 'relu',
    fnn.Dense(10)  @ 'linear'  # logits
])

# Initialization options (choose one):
# 1. Provide Input layer (already done) -> shapes known
# 2. Or run: _ = model(fnn.Tensor([[0.0]*784]))
# 3. Or: model.summary((784,))
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
    fnn.Dense(10)              # logits (linear)
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
# Manual loop (SGD)
opt = fnn.SGD(model.parameters(), lr=0.01, momentum=0.9)
logits = model(fnn.Tensor(batch_x))
loss = fnn.cross_entropy_loss(logits, batch_y)
opt.zero_grad(); loss.backward(); opt.step()

# Or adaptive
# opt = fnn.Adam(model.parameters(), lr=1e-3)
```

## Training with compile/fit

If you prefer a Keras-like workflow, use the lightweight training wrapper:

```python
import forgeNN as fnn

model = fnn.Sequential([
    fnn.Input((784,)),            # ensures shapes known early (optional)
    fnn.Dense(128) @ 'relu',
    fnn.Dense(64)  @ 'relu',
    fnn.Dense(10)                 # logits
])

# Option A: Dict config (factory builds optimizer lazily)
compiled = fnn.compile(
    model,
    optimizer={"type": "adam", "lr": 1e-3},
    loss='cross_entropy',
    metrics=['accuracy']
)

# Option B: Deferred instance
# opt = fnn.Adam(lr=1e-3)
# compiled = fnn.compile(model, optimizer=opt, loss='cross_entropy', metrics=['accuracy'])

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
- Lazy initialization in Dense keeps constructor minimal for readability; specify in_features to disable laziness or use an Input layer/summary to pre-build
- Sequential.parameters() returns trainable tensors ready for any optimizer (SGD, Adam, AdamW)
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
- Optimizer interactions (SGD momentum, Adam state, deferred binding)


## When to Use Sequential

- Small to medium MLPs and feature heads
- Clean, readable architecture definitions
- Educational settings where clarity and explicitness matter

For more context on forgeNN’s API philosophy and performance trade-offs, see `COMPARISON_GUIDE.md`.
