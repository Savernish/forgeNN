"""
PyTorch version of the MNIST fast CNN demo to compare speed/accuracy.

Model (match forgeNN fast example):
  28x28 -> Conv(8, 3x3, valid) -> ReLU -> Flatten -> Linear(10)

Usage
  python -c "import sys, os; sys.path.insert(0, os.getcwd()); import examples.mnist_cnn_fast_torch as m; m.main()"

Env knobs
  EPOCHS (default 5), BATCH (default 512), MNIST_LIMIT (default 60000)
"""

from __future__ import annotations

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:  # pragma: no cover
    raise RuntimeError("This example requires PyTorch. Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu") from e


def load_mnist(limit: int | None = None, seed: int = 42):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = (mnist["data"].astype(np.float32) / 255.0).reshape(-1, 1, 28, 28)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = mnist["target"].astype(np.int64)
    if limit is not None and limit < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=limit, random_state=seed, stratify=y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=10000, random_state=seed, stratify=y)
    return X_tr, y_tr, X_te, y_te


class FastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(8 * 26 * 26, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    limit = os.environ.get('MNIST_LIMIT', '')
    limit_int = int(limit) if (limit.strip().isdigit()) else 60000
    X_tr, y_tr, X_te, y_te = load_mnist(limit_int)

    device = torch.device('cpu')
    model = FastNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = int(os.environ.get('EPOCHS', '5'))
    batch_size = int(os.environ.get('BATCH', '512'))

    # Dataloaders
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    ds_te = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    dl_te = DataLoader(ds_te, batch_size=1024, shuffle=False, drop_last=False)

    # Warmup one small step
    model.train()
    xb, yb = next(iter(DataLoader(ds_tr, batch_size=min(64, len(ds_tr)))))
    xb = xb.to(device); yb = yb.to(device)
    opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()

    # Train
    history = {'loss': [], 'accuracy': []}
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        correct = 0; total = 0; loss_sum = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
            loss_sum += float(loss.item()) * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
        history['loss'].append(loss_sum / max(total, 1))
        history['accuracy'].append(correct / max(total, 1))
    total_s = time.time() - t0

    # Evaluate
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device); yb = yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))
    acc = correct / max(total, 1)
    print(f"Time: {total_s:.2f}s, Test acc: {acc*100:.2f}%")

    # Plot
    e = np.arange(1, len(history['loss']) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(e, history['loss'], label='loss')
    plt.plot(e, np.array(history['accuracy']) * 100, label='acc (%)')
    plt.xlabel('Epoch'); plt.title('MNIST fast CNN (PyTorch)'); plt.legend(); plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'artifacts_mnist_fast_torch.png')
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()


