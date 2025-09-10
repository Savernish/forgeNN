"""Tiny demo of the runtime device API (scaffold).

Selects 'cuda' if available, otherwise falls back to 'cpu', and runs a short
compile/fit/evaluate cycle. CUDA is not implemented yet; this shows usage.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import forgeNN as fnn

def main():
    device = 'cuda' if fnn.is_cuda_available() else 'cpu'
    print(f"default device before: {fnn.get_default_device()}")
    with fnn.use_device(device):
        print(f"using device: {fnn.get_default_device()}")

        # Tiny toy dataset
        rng = np.random.default_rng(42)
        X = rng.normal(size=(256, 20)).astype(np.float32)
        y = rng.integers(0, 3, size=256, dtype=np.int64)

        # Simple model
        model = fnn.Sequential([
            fnn.Input((20,)),
            fnn.Dense(32) @ 'relu',
            fnn.Dense(3)
        ])
        compiled = fnn.compile(
            model,
            optimizer=fnn.Adam(lr=1e-3),
            loss='cross_entropy',
            metrics=['accuracy']
        )

        compiled.fit(X, y, epochs=1, batch_size=64, verbose=True)
        loss, metrics = compiled.evaluate(X, y)
        print(f"loss={loss:.4f}, acc={metrics.get('accuracy'):.4f}")

    print(f"default device after: {fnn.get_default_device()}")

if __name__ == "__main__":
    main()
