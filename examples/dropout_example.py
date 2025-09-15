import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import forgeNN as fnn


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def build_model(input_dim: int, use_dropout: bool) -> fnn.Sequential:
    layers = [
        fnn.Input((input_dim,)),
        fnn.Dense(64) @ 'relu',
    ]
    if use_dropout:
        layers.append(fnn.Dropout(0.5))
    layers.extend([
        fnn.Dense(32) @ 'relu',
    ])
    if use_dropout:
        layers.append(fnn.Dropout(0.5))
    layers.append(fnn.Dense(3))  # logits
    return fnn.Sequential(layers)


def train_one(model: fnn.Sequential,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int, batch_size: int, lr: float,
              shuffles: list[np.ndarray]) -> dict:
    compiled = fnn.compile(
        model,
        optimizer={"type": "adam", "lr": lr, "eps": 1e-7, "betas": (0.9, 0.999)},
        loss="cross_entropy",
        metrics=["accuracy"],
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for e in range(epochs):
        idx = shuffles[e]
        Xs, ys = X_train[idx], y_train[idx]

        # One epoch of training using the same batch order as the other model
        compiled.fit(Xs, ys, epochs=1, batch_size=batch_size, shuffle=False,
                     validation_data=(X_val, y_val), verbose=0)

        # Epoch-end metrics (no extra forward during fit)
        tr_loss, tr_metrics = compiled.evaluate(X_train, y_train, batch_size=batch_size)
        va_loss, va_metrics = compiled.evaluate(X_val, y_val, batch_size=batch_size)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_metrics.get("accuracy", float("nan")))
        history["val_acc"].append(va_metrics.get("accuracy", float("nan")))

        print(
            f"Epoch {e+1:02d}: loss={tr_loss:.4f}, acc={history['train_acc'][-1]*100:.1f}%  "
            f"val_loss={va_loss:.4f}, val_acc={history['val_acc'][-1]*100:.1f}%"
        )

    return history


def plot_histories(hist_base: dict, hist_drop: dict, out_path: str = 'dropout_overfitting.png') -> None:
    epochs = range(1, len(hist_base['train_loss']) + 1)
    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_base['train_loss'], 'r-', label='baseline train')
    plt.plot(epochs, hist_base['val_loss'], 'r--', label='baseline val')
    plt.plot(epochs, hist_drop['train_loss'], 'b-', label='dropout train')
    plt.plot(epochs, hist_drop['val_loss'], 'b--', label='dropout val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, np.array(hist_base['train_acc']) * 100, 'r-', label='baseline train')
    plt.plot(epochs, np.array(hist_base['val_acc']) * 100, 'r--', label='baseline val')
    plt.plot(epochs, np.array(hist_drop['train_acc']) * 100, 'b-', label='dropout train')
    plt.plot(epochs, np.array(hist_drop['val_acc']) * 100, 'b--', label='dropout val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    # Data
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_classes=3,
        n_informative=6,
        flip_y=0.02,
        random_state=7,
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=7)

    input_dim = X_train.shape[1]

    # Build models with identical initial weights for fairness
    set_seed(args.seed)
    baseline = build_model(input_dim, use_dropout=False)
    # Initialize parameters deterministically
    baseline.summary((input_dim,))

    set_seed(args.seed)
    with_dropout = build_model(input_dim, use_dropout=True)
    with_dropout.summary((input_dim,))

    # Pre-compute the same shuffle per epoch and share across models
    n = len(X_train)
    rng = np.random.RandomState(args.seed)
    shuffles = [rng.permutation(n) for _ in range(args.epochs)]

    print("Training baseline (no dropout)...")
    hist_base = train_one(baseline, X_train, y_train, X_val, y_val, args.epochs, args.batch_size, args.lr, shuffles)

    print("\nTraining with dropout...")
    hist_drop = train_one(with_dropout, X_train, y_train, X_val, y_val, args.epochs, args.batch_size, args.lr, shuffles)

    plot_histories(hist_base, hist_drop)


if __name__ == '__main__':
    main()
