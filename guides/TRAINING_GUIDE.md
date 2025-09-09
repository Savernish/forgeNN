# Training Guide: From PyTorch and Keras to forgeNN

This guide shows how to adapt common training patterns from PyTorch and Keras to forgeNN. It focuses on a minimal, familiar API: compile, fit, evaluate, and predict.

## Quick Start

```python
import forgeNN as fnn

# Define model
model = fnn.Sequential([
    fnn.Input((784,)),        # optional: seeds summary/inference
    fnn.Dense(128) @ 'relu',
    fnn.Dense(64)  @ 'relu',
    fnn.Dense(10)  @ 'linear',  # logits
])

# Initialization options (any one works):
# 1. Include an Input layer (already done)
# 2. Call model.summary() or model.summary((784,)) to force symbolic build
# 3. Run a dummy forward: _ = model(fnn.Tensor([[0.0]*784]))

# Compile with optimizer config, loss, metrics
compiled = fnn.compile(
    model,
    optimizer={"type": "adam", "lr": 1e-3},  # or {"lr":0.01, "momentum":0.9} for SGD
    loss='cross_entropy',
    metrics=['accuracy']
)

# Train with validation
compiled.fit(X_train, y_train, epochs=10, batch_size=64,
             validation_data=(X_test, y_test))

# Evaluate and predict
loss, metrics = compiled.evaluate(X_test, y_test)
logits = compiled.predict(X_test[:16])
```

## Mapping: Keras → forgeNN

| Keras | forgeNN |
|-------|---------|
| model.compile(optimizer, loss, metrics) | compiled = fnn.compile(model, optimizer=..., loss=..., metrics=...) |
| model.fit(...) | compiled.fit(...) |
| model.evaluate(...) | compiled.evaluate(...) |
| model.predict(...) | compiled.predict(...) |

Notes:
Notes:
- Optimizer can be:
    * Dict config: {"type": "sgd", "lr": 0.01, "momentum": 0.9} or {"type": "adam", "lr": 1e-3} or {"type": "adamw", "lr":1e-3, "weight_decay":0.01}
    * Instance with params: `fnn.Adam(model.parameters(), lr=1e-3)`
    * Deferred instance (no params yet): `opt = fnn.Adam(lr=1e-3)` then pass to compile
- Built-in losses: cross_entropy, mse. Built-in metrics: accuracy. Pass callables for custom ones.

## Mapping: PyTorch → forgeNN

| PyTorch | forgeNN |
|---------|---------|
| criterion(outputs, targets) | loss_fn(logits, targets) inside compiled.fit |
| optimizer.zero_grad() | handled internally |
| loss.backward() | handled internally |
| optimizer.step() | handled internally |

If you prefer explicit loops, you can still create `SGD`, `Adam`, or `AdamW` and write the loop manually; `compile/fit` is optional convenience.

## Customization

- Custom loss: pass loss=my_loss_fn where my_loss_fn(logits: Tensor, y: np.ndarray) -> Tensor.
- Custom metric: pass metrics=["accuracy", my_metric_fn] where my_metric_fn(logits: Tensor, y: np.ndarray) -> float.
- Optimizer instance: pass `fnn.SGD(...)`, `fnn.Adam(...)` or `fnn.AdamW(...)` (with or without params yet).

## Tips for Lazy Initialization

If a layer (like Dense) infers input size on the first forward, ensure parameters are created before compiling by either:

```python
model = fnn.Sequential([
    fnn.Input((input_dim,)), fnn.Dense(32) @ 'relu', fnn.Dense(10)
])  # Input layer seeds shapes

# Or: model.summary((input_dim,))  # symbolic build
# Or: _ = model(fnn.Tensor(np.zeros((1, input_dim), dtype=np.float32)))

compiled = fnn.compile(model, optimizer={"type": "adam", "lr": 1e-3})
```

## When to Use compile/fit

- You want a clean, high-level training flow similar to Keras
- You are migrating from Keras or PyTorch and want minimal boilerplate
- You prefer naming your losses/metrics and letting the loop be managed for you

When you need custom schedules, mixed losses, gradient clipping, or hooks, write an explicit loop—forgeNN remains flexible.
