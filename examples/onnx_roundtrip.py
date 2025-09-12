import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
import forgeNN as fnn


def _make_dataset(rng: np.random.Generator, n: int, in_dim: int, classes: int):
    """Synthetic dataset for quick training and parity tests."""
    W_true = rng.normal(scale=1.0, size=(in_dim, classes)).astype(np.float32)
    b_true = rng.normal(scale=0.1, size=(classes,)).astype(np.float32)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    logits = X @ W_true + b_true
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


def main():
    p = argparse.ArgumentParser(description="Export → Import (ONNX) roundtrip parity check")
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--out', type=str, default='mlp.onnx')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--in-dim', type=int, default=20)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--classes', type=int, default=3)
    p.add_argument('--tol', type=float, default=5e-6, help='Max allowed absolute difference for parity assert')
    p.add_argument('--strict', action='store_true', help='Use strict mode in ONNX importer')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = _make_dataset(rng, args.samples, args.in_dim, args.classes)

    # Build and train a tiny MLP
    model = fnn.Sequential([
        fnn.Input((args.in_dim,)),
        fnn.Dense(args.hidden) @ 'relu',
        fnn.Dense(args.classes),  # logits
    ])
    model.summary((args.in_dim,))

    compiled = fnn.compile(
        model,
        optimizer={"type": "adam", "lr": args.lr, "eps": 1e-7},
        loss="cross_entropy",
        metrics=["accuracy"],
    )
    compiled.fit(X, y, epochs=args.epochs, batch_size=args.batch, validation_data=(X, y), verbose=1)

    # Baseline outputs and accuracy
    fnn_logits = model.forward(fnn.Tensor(X, requires_grad=False)).data
    preds = np.argmax(fnn_logits, axis=1)
    acc = float(np.mean(preds == y)) * 100.0
    print(f"forgeNN baseline accuracy: {acc:.2f}%")

    # Export to ONNX
    fnn.export_onnx(model, args.out, opset=args.opset, input_example=X[:1], include_softmax=False, ir_version=10)
    print(f"Exported ONNX to {args.out} (opset {args.opset})")

    # Import back from ONNX
    imported = fnn.load_onnx(args.out, strict=args.strict)
    imported.summary((args.in_dim,))

    # Compare raw logits parity
    imported_logits = imported.forward(fnn.Tensor(X, requires_grad=False)).data
    max_abs_diff = float(np.max(np.abs(imported_logits - fnn_logits)))
    print(f"Roundtrip parity max|diff| = {max_abs_diff:.6e}")
    if not np.allclose(imported_logits, fnn_logits, atol=args.tol, rtol=0.0):
        raise AssertionError(f"Parity check failed: max|diff|={max_abs_diff:.3e} > tol={args.tol}")

    # Compare accuracy parity
    preds_imported = np.argmax(imported_logits, axis=1)
    acc_imported = float(np.mean(preds_imported == y)) * 100.0
    print(f"Imported model accuracy: {acc_imported:.2f}%")

    # Optional: compile imported and evaluate to mirror training API
    compiled_imp = fnn.compile(
        imported,
        optimizer={"type": "adam", "lr": args.lr, "eps": 1e-7},
        loss="cross_entropy",
        metrics=["accuracy"],
    )
    loss_imp, metrics_imp = compiled_imp.evaluate(X, y, batch_size=args.batch)
    acc_eval = float(metrics_imp.get('accuracy', np.nan)) * 100.0
    print(f"Imported compiled evaluate -> loss={loss_imp:.4f}, accuracy={acc_eval:.2f}%")


if __name__ == "__main__":
    main()
