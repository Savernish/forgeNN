import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import numpy as np
import forgeNN as fnn

def _make_dataset(rng: np.random.Generator, n: int, in_dim: int, classes: int):
    # Synthetic linearly-separable-ish dataset for quick training
    W_true = rng.normal(scale=1.0, size=(in_dim, classes)).astype(np.float32)
    b_true = rng.normal(scale=0.1, size=(classes,)).astype(np.float32)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    logits = X @ W_true + b_true
    y = np.argmax(logits, axis=1).astype(np.int64)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--out', type=str, default='mlp.onnx')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--in-dim', type=int, default=20)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--classes', type=int, default=3)
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

    # Export to ONNX (eval mode implied in exporter)
    fnn.export_onnx(model, args.out, opset=args.opset, input_example=X[:1], include_softmax=False, ir_version=10)
    print(f"Exported ONNX to {args.out} (opset {args.opset})")

    # Try TensorFlow backend via onnx-tf first; fallback to onnxruntime if missing
    ran_tf = False
    try:
        import onnx
        from onnx_tf.backend import prepare
        onnx_model = onnx.load(args.out)
        tf_rep = prepare(onnx_model)
        tf_out_list = tf_rep.run({"input_0": X})  # returns list of np arrays
        tf_out = tf_out_list[0]
        # Parity check vs forgeNN
        fnn_out = model.forward(fnn.Tensor(X, requires_grad=False)).data
        max_diff_tf = float(np.max(np.abs(tf_out - fnn_out)))
        print(f"TensorFlow(onnx-tf) vs forgeNN max|diff| = {max_diff_tf:.6e}")
        # Simple accuracy agreement
        preds_tf = np.argmax(tf_out, axis=1)
        acc_tf = float(np.mean(preds_tf == y)) * 100.0
        print(f"TensorFlow(onnx-tf) accuracy on train set: {acc_tf:.2f}%")
        ran_tf = True
    except Exception as e:
        print(f"onnx-tf not available or failed, falling back to onnxruntime: {e}")

    if not ran_tf:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
            ort_out = sess.run(None, {"input_0": X})[0]
            fnn_out = model.forward(fnn.Tensor(X, requires_grad=False)).data
            max_diff = float(np.max(np.abs(ort_out - fnn_out)))
            print(f"ONNXRuntime vs forgeNN max|diff| = {max_diff:.6e}")
            preds_ort = np.argmax(ort_out, axis=1)
            acc_ort = float(np.mean(preds_ort == y)) * 100.0
            print(f"ONNXRuntime accuracy on train set: {acc_ort:.2f}%")
        except Exception as e:
            print(f"onnxruntime not available or failed to run parity check: {e}")

if __name__ == "__main__":
    main()