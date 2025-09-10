"""
forgeNN MNIST Classification Example
===================================

Complete example demonstrating high-performance neural network training
with forgeNN v2 Sequential + compile/fit API.

Features:
- MNIST handwritten digit classification
- Vectorized NumPy core for fast training
- Batch processing with progress tracking
- Professional metrics and evaluation

Performance: 93%+ accuracy in under 2 seconds!
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import forgeNN as fnn
from forgeNN.core.tensor import Tensor

def load_mnist_vectorized(n_samples=5000):
    """Load and preprocess MNIST for training."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use more samples since v2 is fast
    print(f"Using {n_samples} samples for vectorized training...")
    X, y = X[:n_samples], y[:n_samples]
    
    # Normalize features
    X = X.astype(np.float32) / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test

def create_data_loader(X, y, batch_size=32, shuffle=True):
    """Create simple data loader for batch training."""
    n_samples = len(X)
    if shuffle:
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        yield X[i:end_idx], y[i:end_idx]

def train_epoch(compiled, X_train, y_train, batch_size=64):
    """Train for one epoch using compile/fit style helper (single-epoch run)."""
    compiled.fit(X_train, y_train, epochs=1, batch_size=batch_size, shuffle=True, verbose=0)
    # After fit, run evaluate to get epoch metrics without extra state tracking
    return compiled.evaluate(X_train, y_train, batch_size=batch_size)

def evaluate(compiled, X_test, y_test, batch_size=64):
    """Evaluate compiled model on test set."""
    return compiled.evaluate(X_test, y_test, batch_size=batch_size)

def main():
    print("="*60)
    print("MNIST CLASSIFICATION WITH forgeNN (v2)")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_vectorized(n_samples=5000)
    
    # Create model
    print("\nCreating Sequential neural network (v2)...")
    model = fnn.Sequential([
        fnn.Input((784,)),
        fnn.Dense(128) @ 'relu',
        fnn.Dense(64) @ 'relu',
        fnn.Dense(10)
    ])
    compiled = fnn.compile(
        model,
        optimizer={"type": "adam", "lr": 0.01, "eps": 1e-7},
        loss='cross_entropy',
        metrics=['accuracy']
    )
    print(f"Model parameters: {len(model.parameters())}")
    total_params = sum(p.data.size for p in model.parameters())
    print(f"Total parameter count: {total_params:,}")
    
    # Training configuration
    epochs = 10
    batch_size = 64
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: Adam lr=0.01 eps=1e-7")
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
    # Train and evaluate
    train_loss, train_metrics = train_epoch(compiled, X_train, y_train, batch_size)
    test_loss, test_metrics = evaluate(compiled, X_test, y_test, batch_size)
        
        # Print metrics
    print(f"  Train: Loss = {train_loss:.4f}, Acc = {train_metrics['accuracy']*100:.1f}%")
    print(f"  Test:  Loss = {test_loss:.4f}, Acc = {test_metrics['accuracy']*100:.1f}%")
    
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Final evaluation
    final_test_loss, final_test_metrics = evaluate(compiled, X_test, y_test, batch_size)
    
    print(f"Final Test Accuracy: {final_test_metrics['accuracy']*100:.2f}%")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Training Time: {training_time:.1f} seconds")
    print(f"Samples per Second: {len(X_train) * epochs / training_time:.0f}")
    
    # Performance comparison
    samples_per_epoch = len(X_train)
    time_per_sample = training_time / (samples_per_epoch * epochs)
    print(f"Time per Sample: {time_per_sample*1000:.2f} ms")
    
    # Show some predictions
    print("\nSample Predictions:")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for i, idx in enumerate(sample_indices):
        x_sample = Tensor(X_test[idx:idx+1], requires_grad=False)
        logits = model(x_sample)
        probs = logits.softmax(axis=1)
        predicted = np.argmax(probs.data[0])
        confidence = np.max(probs.data[0])
        
        print(f"  Sample {i+1}: True = {y_test[idx]}, "
              f"Predicted = {predicted}, Confidence = {confidence:.3f}")
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    print(f"✅ v2 implementation achieved:")
    print(f"   • {final_test_metrics['accuracy']*100:.1f}% accuracy on MNIST")
    print(f"   • {samples_per_epoch * epochs / training_time:.0f} samples/second")
    print(f"   • {time_per_sample*1000:.2f} ms per sample")
    

if __name__ == "__main__":
    main()
