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
    optimizer={"lr": 0.01, "momentum": 0.9},
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
|------|---------|
| model.compile(optimizer, loss, metrics) | compiled = fnn.compile(model, optimizer=..., loss=..., metrics=...) |
| model.fit(X, y, epochs, batch_size, validation_data) | compiled.fit(X, y, epochs, batch_size, validation_data) |
| model.evaluate(X, y) | compiled.evaluate(X, y) |
| model.predict(X) | compiled.predict(X) |

Notes:
- Optimizer in forgeNN can be an instance or a dict config. Dict config constructs VectorizedOptimizer with the given hyperparameters.
- Loss names and metric names are registered; currently cross_entropy and accuracy are built-in. You can pass callables for custom behavior.

## Mapping: PyTorch → forgeNN

| PyTorch | forgeNN |
|--------|---------|
| criterion(outputs, targets) | loss_fn(logits, targets) inside compiled.fit (built-in with cross_entropy) |
| optimizer.zero_grad() | handled inside compiled.fit |
| loss.backward() | handled inside compiled.fit |
| optimizer.step() | handled inside compiled.fit |

If you prefer explicit loops, you can still create VectorizedOptimizer and write your own training loop; compile/fit is a convenience layer.

## Customization

- Custom loss: pass loss=my_loss_fn where my_loss_fn(logits: Tensor, y: np.ndarray) -> Tensor.
- Custom metric: pass metrics=["accuracy", my_metric_fn] where my_metric_fn(logits: Tensor, y: np.ndarray) -> float.
- Optimizer instance: pass an existing VectorizedOptimizer(model.parameters(), lr=..., momentum=...).

## Tips for Lazy Initialization

If a layer (like Dense) infers input size on the first forward, ensure parameters are created before compiling by either:

```python
# Option A: Provide Input layer upfront
model = fnn.Sequential([fnn.Input((input_dim,)), fnn.Dense(32) @ 'relu', fnn.Dense(10)])

# Option B: Call summary with shape
model.summary((input_dim,))

# Option C: Dummy forward
_ = model(fnn.Tensor(np.zeros((1, input_dim), dtype=np.float32)))

compiled = fnn.compile(model, optimizer={"lr": 0.01})
```

## When to Use compile/fit

- You want a clean, high-level training flow similar to Keras
- You are migrating from Keras or PyTorch and want minimal boilerplate
- You prefer naming your losses/metrics and letting the loop be managed for you

When you need custom schedules, mixed losses, or special hooks, write an explicit loop—forgeNN remains flexible.
