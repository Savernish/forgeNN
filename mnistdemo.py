"""
MNIST Handwritten Digit Classification with forgeNN
===================================================

A complete example of training a neural network on the MNIST dataset
using the forgeNN framework. Demonstrates multi-class classification
with a feedforward neural network.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

from forgeNN.core import Value
from forgeNN.network import MLP

def load_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use a very small subset for testing
    print("Using subset of 200 samples for testing...")
    X, y = X[:200], y[:200]
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    
    return X_train, X_test, y_train, y_test

def softmax(logits):
    """Compute softmax probabilities"""
    exp_logits = [np.exp(logit.data) for logit in logits]
    sum_exp = sum(exp_logits)
    return [exp_val / sum_exp for exp_val in exp_logits]

def cross_entropy_loss(logits, target_class):
    """Compute cross-entropy loss for multi-class classification"""
    probs = softmax(logits)
    return Value(-np.log(probs[target_class] + 1e-8))

def predict(model, X):
    """Make predictions on a batch of data"""
    predictions = []
    print(f"    Making predictions on {len(X)} samples...")
    
    for i, x in enumerate(X):
        if (i + 1) % 200 == 0:
            print(f"      Predicting sample {i+1}/{len(X)}")
            
        # Convert to Value objects
        inputs = [Value(float(pixel)) for pixel in x]
        
        # Forward pass
        logits = model(inputs)
        
        # Get predicted class
        probs = softmax(logits)
        pred_class = np.argmax(probs)
        predictions.append(pred_class)
    
    return np.array(predictions)

def train_epoch(model, X_train, y_train, learning_rate):
    """Train for one epoch"""
    total_loss = 0
    correct = 0
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    
    print(f"  Starting training on {len(X_train)} samples...")
    
    for i, idx in enumerate(indices):
        x, target = X_train[idx], y_train[idx]
        
        # Progress indicator for every 25 samples
        if (i + 1) % 25 == 0:
            print(f"    Sample {i+1}/{len(X_train)} "
                  f"(Loss: {total_loss/(i+1):.4f}, Acc: {correct/(i+1)*100:.1f}%)")
        
        # Reset gradients
        for param in model.parameters():
            param.grad = 0
        
        # Convert input to Value objects
        inputs = [Value(float(pixel)) for pixel in x]
        
        # Forward pass
        logits = model(inputs)
        
        # Compute loss
        loss = cross_entropy_loss(logits, target)
        total_loss += loss.data
        
        # Check accuracy
        probs = softmax(logits)
        predicted = np.argmax(probs)
        if predicted == target:
            correct += 1
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        for param in model.parameters():
            param.data -= learning_rate * param.grad
    
    avg_loss = total_loss / len(X_train)
    accuracy = correct / len(X_train)
    
    return avg_loss, accuracy

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Create neural network
    print("\nCreating neural network...")
    model = MLP(
        nin=784,                    # MNIST images are 28x28 = 784 pixels
        nouts=[16, 10],            # Very small network: 16 hidden, 10 output
        activations=['relu', 'linear']
    )
    
    print(f"Model parameters: {len(model.parameters())}")
    
    # Training parameters
    learning_rate = 0.05  # Higher learning rate
    epochs = 1            # Just one epoch for testing
    
    print(f"\nTraining for {epochs} epochs with learning rate {learning_rate}")
    print("=" * 60)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, X_train, y_train, learning_rate)
        
        # Test
        print("  Evaluating on test set...")
        test_predictions = predict(model, X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"  Test Acc: {test_accuracy*100:.1f}%")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    final_predictions = predict(model, X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    
    print(f"Final Test Accuracy: {final_accuracy*100:.2f}%")
    print(f"Total Parameters: {len(model.parameters()):,}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, final_predictions))
    
    # Show some example predictions
    print("\nExample Predictions:")
    for i in range(5):
        inputs = [Value(float(pixel)) for pixel in X_test[i]]
        logits = model(inputs)
        probs = softmax(logits)
        predicted = np.argmax(probs)
        confidence = max(probs)
        
        print(f"  True: {y_test[i]}, Predicted: {predicted}, "
              f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()
