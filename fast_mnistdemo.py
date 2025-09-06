"""
Fast MNIST Demo - Optimized Version
===================================

Demonstrates several optimization techniques to speed up forgeNN training.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

from forgeNN.core import Value
from forgeNN.network import MLP

# Optimization 1: Pre-allocate Value objects to reduce object creation
class ValuePool:
    def __init__(self, size=100000):
        self.values = [Value(0.0) for _ in range(size)]
        self.index = 0
    
    def get(self, data):
        val = self.values[self.index % len(self.values)]
        val.data = data
        val.grad = 0.0
        val._prev = set()
        self.index += 1
        return val

# Global value pool
value_pool = ValuePool()

# Optimization 2: Batch processing utilities
def create_value_batch(data_batch):
    """Convert numpy array to list of Values efficiently"""
    return [value_pool.get(float(x)) for x in data_batch.flatten()]

# Optimization 3: Faster softmax using numpy
def fast_softmax(logits):
    """Numerically stable softmax"""
    exp_vals = np.exp([l.data for l in logits] - np.max([l.data for l in logits]))
    sum_exp = np.sum(exp_vals)
    return exp_vals / sum_exp

# Optimization 4: Simplified cross-entropy
def fast_cross_entropy(logits, target_class):
    """Fast cross-entropy computation"""
    probs = fast_softmax(logits)
    return Value(-np.log(probs[target_class] + 1e-8))

# Optimization 5: Mini-batch training
def train_mini_batch(model, X_batch, y_batch, learning_rate):
    """Train on a mini-batch of samples"""
    batch_size = len(X_batch)
    total_loss = 0
    correct = 0
    
    # Accumulate gradients across batch
    accumulated_grads = [0.0] * len(model.parameters())
    
    for i in range(batch_size):
        # Reset gradients for this sample
        for param in model.parameters():
            param.grad = 0
        
        # Forward pass
        inputs = create_value_batch(X_batch[i])
        logits = model(inputs)
        
        # Loss and accuracy
        loss = fast_cross_entropy(logits, y_batch[i])
        total_loss += loss.data
        
        probs = fast_softmax(logits)
        if np.argmax(probs) == y_batch[i]:
            correct += 1
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        for j, param in enumerate(model.parameters()):
            accumulated_grads[j] += param.grad
    
    # Update parameters with averaged gradients
    for j, param in enumerate(model.parameters()):
        avg_grad = accumulated_grads[j] / batch_size
        param.data -= learning_rate * avg_grad
    
    return total_loss / batch_size, correct / batch_size

def load_small_mnist():
    """Load a very small MNIST subset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use tiny subset for speed testing
    X, y = X[:100], y[:100]
    X = X / 255.0
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_small_mnist()
    
    # Create smaller network
    print("Creating optimized neural network...")
    model = MLP(784, [32, 10], ['relu', 'linear'])
    print(f"Parameters: {len(model.parameters())}")
    
    # Training with mini-batches
    batch_size = 10
    learning_rate = 0.1
    epochs = 3
    
    print(f"\nTraining with batch size {batch_size}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle and create mini-batches
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            batch_loss, batch_acc = train_mini_batch(
                model, X_batch, y_batch, learning_rate
            )
            
            epoch_loss += batch_loss
            epoch_acc += batch_acc
            num_batches += 1
            
            if (i // batch_size + 1) % 2 == 0:
                print(f"  Batch {i//batch_size + 1}: "
                      f"Loss = {batch_loss:.4f}, Acc = {batch_acc*100:.1f}%")
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"  Epoch avg: Loss = {avg_loss:.4f}, Acc = {avg_acc*100:.1f}%")
    
    training_time = time.time() - start_time
    print(f"\nOptimized training completed in {training_time:.1f} seconds")
    
    # Quick test
    print("\nTesting...")
    correct = 0
    for i in range(len(X_test)):
        inputs = create_value_batch(X_test[i])
        logits = model(inputs)
        probs = fast_softmax(logits)
        if np.argmax(probs) == y_test[i]:
            correct += 1
    
    test_acc = correct / len(X_test)
    print(f"Final test accuracy: {test_acc*100:.1f}%")

if __name__ == "__main__":
    main()
