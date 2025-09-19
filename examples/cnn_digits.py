"""
Comprehensive image recognition example on sklearn Digits (8x8) using Conv2D/MaxPool2D.

Highlights
- Uses forgeNN Sequential + compile/fit/evaluate workflow
- NCHW layout with Conv2D and MaxPool2D (valid padding)
- Records per-epoch train/val loss and accuracy, with plots
- Shows confusion matrix and a small prediction gallery

Run
  python examples/cnn_digits.py

Requirements
  pip install -r requirements.txt  # ensures matplotlib and scikit-learn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import forgeNN as fnn


def load_data(test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    data = load_digits()
    X = data.images.astype(np.float32)  # (N, 8, 8), values 0..16
    y = data.target.astype(np.int64)    # (N,)
    # Normalize to [0,1]
    X = X / 16.0
    # NCHW: add channel dimension C=1
    X = np.expand_dims(X, axis=1)  # (N, 1, 8, 8)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    # Further split train -> train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model() -> fnn.Sequential:
    # Architecture for 8x8 inputs (valid convs):
    # 8x8 -> Conv(3x3,s1) -> 6x6 -> ReLU -> MaxPool(2x2,s2) -> 3x3 ->
    # -> Conv(3x3,s1) -> 1x1 -> Flatten -> Dense(10)
    model = fnn.Sequential([
        fnn.Input((1, 8, 8)),
        fnn.Conv2D(cin=1, cout=16, kernel_size=(3, 3), stride=(1, 1)) @ 'relu',
        fnn.MaxPool2D(kernel_size=(2, 2), stride=(2, 2)),
        fnn.Conv2D(cin=16, cout=32, kernel_size=(3, 3), stride=(1, 1)) @ 'relu',
        fnn.Flatten(),
        fnn.Dense(10)  # logits
    ])
    return model


def train_with_history(compiled, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }
    for epoch in range(1, epochs + 1):
        compiled.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=1)
        tr_loss, tr_metrics = compiled.evaluate(X_train, y_train, batch_size=batch_size)
        va_loss, va_metrics = compiled.evaluate(X_val, y_val, batch_size=batch_size)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_metrics.get('accuracy', np.nan))
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_metrics.get('accuracy', np.nan))
        print(f"Epoch {epoch}/{epochs} done  train: loss={tr_loss:.4f}, acc={tr_metrics['accuracy']*100:.2f}%  "
              f"val: loss={va_loss:.4f}, acc={va_metrics['accuracy']*100:.2f}%")
    return history


def plot_history(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train')
    plt.plot(epochs, history['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, np.array(history['train_acc']) * 100, label='train')
    plt.plot(epochs, np.array(history['val_acc']) * 100, label='val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy'); plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path)
    print(f"Saved curves to {path}")


def show_confusion_and_samples(compiled, X_test, y_test, out_dir: str, num_samples: int = 12):
    os.makedirs(out_dir, exist_ok=True)
    # Predictions
    logits = compiled.predict(X_test, batch_size=256)
    y_pred = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.tight_layout(); plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Sample predictions gallery
    idx = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    imgs = X_test[idx, 0]  # (n, 8, 8)
    preds = y_pred[idx]
    trues = y_test[idx]
    cols = 6
    rows = int(np.ceil(len(idx) / cols))
    plt.figure(figsize=(cols * 2, rows * 2))
    for k, i in enumerate(idx):
        plt.subplot(rows, cols, k + 1)
        plt.imshow(X_test[i, 0], cmap='gray', vmin=0.0, vmax=1.0)
        c = 'green' if y_pred[i] == y_test[i] else 'red'
        plt.title(f"p:{y_pred[i]} t:{y_test[i]}", color=c)
        plt.axis('off')
    gal_path = os.path.join(out_dir, 'predictions_grid.png')
    plt.tight_layout(); plt.savefig(gal_path)
    print(f"Saved predictions grid to {gal_path}")


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    # Build and inspect model
    model = build_model()

    # Optimizer and compile
    opt = fnn.Adam(lr=1e-3)
    compiled = fnn.compile(model, optimizer=opt, loss='cross_entropy', metrics=['accuracy'])

    # Train and record history
    history = train_with_history(compiled, X_train, y_train, X_val, y_val, epochs=20, batch_size=128)

    # Final evaluation
    test_loss, test_metrics = compiled.evaluate(X_test, y_test, batch_size=256)
    print(f"Test: loss={test_loss:.4f}, acc={test_metrics['accuracy']*100:.2f}%")

    out_dir = os.path.join(os.path.dirname(__file__), 'artifacts_digits')
    plot_history(history, out_dir)
    show_confusion_and_samples(compiled, X_test, y_test, out_dir)


if __name__ == '__main__':
    main()


