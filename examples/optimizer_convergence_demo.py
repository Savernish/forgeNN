"""Optimizer Convergence Demo
================================

Compares SGD (with momentum) to Adam on a small synthetic classification task
and prints per‑epoch loss/accuracy. The original version used an aggressive
learning rate (0.05) for momentum SGD and a relatively conservative 1e-3 for
Adam, which could make Adam appear weaker. This version:

* Uses more balanced defaults (SGD lr=0.03, Adam lr=0.003) tuned so Adam usually
    achieves a lower loss earlier on this toy problem.
* Exposes CLI flags to experiment with hyperparameters.
* Prints an initial (epoch 0) evaluation so you can see relative improvement.

Run (defaults):
        python examples/optimizer_convergence_demo.py

Tune learning rates:
        python examples/optimizer_convergence_demo.py --sgd-lr 0.05 --adam-lr 0.001

If Adam looks worse:
    - Try a slightly larger Adam lr (e.g. 2e-3 to 5e-3)
    - Reduce SGD lr (aggressive momentum + high lr can dominate early)
    - Ensure batch size isn't so large that adaptivity provides little benefit

Expected (typical):
    Adam loss drops faster first few epochs; SGD may catch up or slightly surpass
    later with tuned momentum.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import forgeNN as fnn


def make_toy_classification(n=600, d=20, k=4, seed=42):
    rng = np.random.default_rng(seed)
    W_true = rng.normal(size=(d, k))
    X = rng.normal(size=(n, d)).astype(np.float32)
    logits = X @ W_true
    y = np.argmax(logits + 0.5 * rng.normal(size=logits.shape), axis=1)
    return X, y


def build_model(input_dim: int, num_classes: int):
    return fnn.Sequential([
        fnn.Dense(64, in_features=input_dim) @ 'relu',
        fnn.Dense(32) @ 'relu',
        fnn.Dense(num_classes) @ 'linear'
    ])


def train(model, optimizer, X, y, epochs=15, batch_size=64):
    n = X.shape[0]
    for epoch in range(1, epochs + 1):
        # mini-batch shuffle
        idx = np.random.permutation(n)
        X_shuf, y_shuf = X[idx], y[idx]
        total_loss = 0.0
        total_correct = 0
        count = 0
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = fnn.Tensor(X_shuf[start:end])
            yb = y_shuf[start:end]
            logits = model(xb)
            loss = fnn.cross_entropy_loss(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.data) * len(yb)
            preds = np.argmax(logits.data, axis=1)
            total_correct += int((preds == yb).sum())
            count += len(yb)
        yield epoch, total_loss / count, total_correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=24, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sgd-lr', type=float, default=0.03)
    parser.add_argument('--sgd-momentum', type=float, default=0.9)
    parser.add_argument('--adam-lr', type=float, default=0.003)
    args = parser.parse_args()

    X, y = make_toy_classification(seed=args.seed)
    input_dim = X.shape[1]
    k = int(y.max() + 1)

    # Build / init models
    sgd_model = build_model(input_dim, k)
    _ = sgd_model(fnn.Tensor(X[:2]))  # initialize params
    adam_model = build_model(input_dim, k)
    _ = adam_model(fnn.Tensor(X[:2]))

    sgd_opt = fnn.SGD(sgd_model.parameters(), lr=args.sgd_lr, momentum=args.sgd_momentum)
    adam_opt = fnn.Adam(adam_model.parameters(), lr=args.adam_lr)

    # Initial evaluation (epoch 0)
    def eval_once(model):
        logits = model(fnn.Tensor(X))
        loss = fnn.cross_entropy_loss(logits, y)
        acc = fnn.accuracy(logits, y)
        return float(loss.data), acc

    l0_sgd, a0_sgd = eval_once(sgd_model)
    l0_adam, a0_adam = eval_once(adam_model)

    print("Balanced Hyperparameters Demo")
    print(f"SGD: lr={args.sgd_lr} momentum={args.sgd_momentum} | Adam: lr={args.adam_lr}")
    print("Epoch |   SGD Loss  SGD Acc | Adam Loss Adam Acc  (epoch 0 shown below)")
    print("-----------------------------------------------------------------------")
    print(f"{0:5d} | {l0_sgd:9.4f} {a0_sgd*100:7.2f}% | {l0_adam:9.4f} {a0_adam*100:7.2f}%")

    for (e1, l_sgd, acc_sgd), (e2, l_adam, acc_adam) in zip(
        train(sgd_model, sgd_opt, X, y, epochs=args.epochs, batch_size=args.batch_size),
        train(adam_model, adam_opt, X, y, epochs=args.epochs, batch_size=args.batch_size)
    ):
        assert e1 == e2
        print(f"{e1:5d} | {l_sgd:9.4f} {acc_sgd*100:7.2f}% | {l_adam:9.4f} {acc_adam*100:7.2f}%")

    print("\nNotes:")
    print(" - Adjust --adam-lr upward (e.g. 0.004 or 0.005) if Adam underperforms early.")
    print(" - Lower --sgd-lr or momentum if SGD is dominating immediately.")
    print(" - On some random seeds, both optimizers can tie; adaptivity gains are stochastic here.")


if __name__ == "__main__":
    main()
