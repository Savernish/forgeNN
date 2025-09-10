import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import forgeNN as fnn


def build_models(input_dim: int = 20, lr: float = 1e-3):
    """Build two forgeNN models (baseline and dropout) and a Keras twin.

    Returns:
        (baseline_fnn, dropout_fnn, dropout_keras_or_None)
    """
    # Ensure identical initialization for a fair comparison
    np.random.seed(0)
    baseline = fnn.Sequential([
        fnn.Input((input_dim,)),
        fnn.Dense(64) @ 'relu',
        fnn.Dense(32) @ 'relu',
        fnn.Dense(3),
    ])
    # Force param init deterministically
    baseline.summary((input_dim,))

    # Match dropout rates with Keras version precisely
    np.random.seed(0)
    with_dropout = fnn.Sequential([
        fnn.Input((input_dim,)),
        fnn.Dense(64) @ 'relu',
        fnn.Dropout(0.5),
        fnn.Dense(32) @ 'relu',
        fnn.Dropout(0.3),
        fnn.Dense(3),
    ])
    with_dropout.summary((input_dim,))

    # Optionally build an equivalent Keras model with same architecture
    keras_model = None
    try:
        import tensorflow as tf
        tf.random.set_seed(0)
        keras_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3),  # logits
        ])
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        # Sync initial Dense weights and biases from forgeNN's dropout model
        def _dense_layers_fnn(seq_model):
            dens = []
            for lyr in seq_model.layers:
                core = lyr.layer if isinstance(lyr, fnn.ActivationWrapper) else lyr
                if isinstance(core, fnn.Dense):
                    dens.append(core)
            return dens

        def _dense_layers_keras(kmodel):
            return [l for l in kmodel.layers if getattr(l, '__class__', None).__name__ == 'Dense']

        f_d = _dense_layers_fnn(with_dropout)
        k_d = _dense_layers_keras(keras_model)
        if len(f_d) == len(k_d):
            for f, k in zip(f_d, k_d):
                W = f.W.data.astype('float32')
                b = f.b.data.astype('float32')
                k.set_weights([W, b])
    except Exception:
        keras_model = None  # TF not installed or failed to init

    return baseline, with_dropout, keras_model


# Removed helper builders; Keras model is built in build_models when available


def train_compare(X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=1e-3):
    """Train baseline vs dropout using compile/fit and collect per-epoch metrics.

    We pre-shuffle once per epoch and call fit with shuffle=False so both models
    see identical minibatches for a fair comparison.
    """
    base, drop, keras_drop = build_models(input_dim=X_train.shape[1], lr=lr)

    compiled_base = fnn.compile(
        base,
        optimizer={"type": "adam", "lr": lr, "eps": 1e-7, "betas": (0.9, 0.999)},
        loss="cross_entropy",
        metrics=["accuracy"],
    )
    compiled_drop = fnn.compile(
        drop,
        optimizer={"type": "adam", "lr": lr, "eps": 1e-7, "betas": (0.9, 0.999)},
        loss="cross_entropy",
        metrics=["accuracy"],
    )

    hist = {
        'base_train_loss': [], 'base_val_loss': [],
        'base_train_acc': [], 'base_val_acc': [],
        'drop_train_loss': [], 'drop_val_loss': [],
        'drop_train_acc': [], 'drop_val_acc': [],
    }

    # If we have a Keras twin, add TF metrics to history
    tf_enabled = keras_drop is not None
    if tf_enabled:
        hist.update({
            'tf_drop_train_loss': [], 'tf_drop_val_loss': [],
            'tf_drop_train_acc': [], 'tf_drop_val_acc': [],
        })

    n = len(X_train)
    for e in range(1, epochs + 1):
        # Pre-shuffle once for both models to ensure identical batch order
        idx = np.random.permutation(n)
        Xs, ys = X_train[idx], y_train[idx]

        # One epoch of training each (no internal shuffle)
        compiled_base.fit(Xs, ys, epochs=1, batch_size=batch_size, shuffle=False,
                          validation_data=(X_val, y_val), verbose=0)
        compiled_drop.fit(Xs, ys, epochs=1, batch_size=batch_size, shuffle=False,
                          validation_data=(X_val, y_val), verbose=0)

        # Train Keras counterpart if enabled
        if tf_enabled:
            keras_drop.fit(
                Xs, ys, epochs=1, batch_size=batch_size, shuffle=False,
                verbose=0, validation_data=(X_val, y_val)
            )

        # Evaluate on full train and val sets to log history
        bl, bm = compiled_base.evaluate(X_train, y_train, batch_size=batch_size)
        blv, bmv = compiled_base.evaluate(X_val, y_val, batch_size=batch_size)
        dl, dm = compiled_drop.evaluate(X_train, y_train, batch_size=batch_size)
        dlv, dmv = compiled_drop.evaluate(X_val, y_val, batch_size=batch_size)

        hist['base_train_loss'].append(bl);  hist['base_val_loss'].append(blv)
        hist['base_train_acc'].append(bm.get('accuracy', float('nan')))
        hist['base_val_acc'].append(bmv.get('accuracy', float('nan')))
        hist['drop_train_loss'].append(dl);  hist['drop_val_loss'].append(dlv)
        hist['drop_train_acc'].append(dm.get('accuracy', float('nan')))
        hist['drop_val_acc'].append(dmv.get('accuracy', float('nan')))

        msg = (
            f"Epoch {e:02d}: base loss={bl:.4f}/{blv:.4f} "
            f"acc={hist['base_train_acc'][-1]*100:.1f}%/{hist['base_val_acc'][-1]*100:.1f}% | "
            f"drop loss={dl:.4f}/{dlv:.4f} "
            f"acc={hist['drop_train_acc'][-1]*100:.1f}%/{hist['drop_val_acc'][-1]*100:.1f}%"
        )
        print(msg)

        if tf_enabled:
            # Evaluate Keras model
            k_dl, k_da = keras_drop.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
            k_dlv, k_dav = keras_drop.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)

            hist['tf_drop_train_loss'].append(k_dl);  hist['tf_drop_val_loss'].append(k_dlv)
            hist['tf_drop_train_acc'].append(k_da);   hist['tf_drop_val_acc'].append(k_dav)

            print(
                f"         TF: drop loss={k_dl:.4f}/{k_dlv:.4f} "
                f"acc={k_da*100:.1f}%/{k_dav*100:.1f}%"
            )

    return base, drop, hist


def plot_history(hist, out_path='dropout_overfitting.png'):
    epochs = range(1, len(hist['base_train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['base_train_loss'], 'r-', label='baseline train')
    plt.plot(epochs, hist['base_val_loss'], 'r--', label='baseline val')
    plt.plot(epochs, hist['drop_train_loss'], 'b-', label='dropout train')
    plt.plot(epochs, hist['drop_val_loss'], 'b--', label='dropout val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss vs Epochs'); plt.legend();
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, np.array(hist['base_train_acc']) * 100, 'r-', label='baseline train')
    plt.plot(epochs, np.array(hist['base_val_acc']) * 100, 'r--', label='baseline val')
    plt.plot(epochs, np.array(hist['drop_train_acc']) * 100, 'b-', label='dropout train')
    plt.plot(epochs, np.array(hist['drop_val_acc']) * 100, 'b--', label='dropout val')
    # Optional TF curves
    if 'tf_drop_train_loss' in hist:
        plt.plot(epochs, np.array(hist['tf_drop_train_acc']) * 100, 'y-', label='TF dropout train')
        plt.plot(epochs, np.array(hist['tf_drop_val_acc']) * 100, 'y--', label='TF dropout val')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy vs Epochs'); plt.legend();
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
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

    # Train both models and collect history
    base, drop, hist = train_compare(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # Plot overfitting trends: baseline should overfit more (train up, val plateaus/dips);
    # dropout should narrow the train-val gap
    plot_history(hist)


if __name__ == '__main__':
    main()
