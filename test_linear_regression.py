from forgeNN.core import Value
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

def test_basic_linear_regression():
    """Test if our Value class can handle basic linear regression"""
    
    # Generate a simple linear dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Use a small subset for testing
    X_sample = X[:5].flatten()  # Take first 5 samples
    y_sample = y[:5]
    
    print("Sample data:")
    for i in range(len(X_sample)):
        print(f"X[{i}]: {X_sample[i]:.3f}, y[{i}]: {y_sample[i]:.3f}")
    
    # Initialize parameters (weight and bias)
    w = Value(0.5)  # weight
    b = Value(0.0)  # bias
    
    print(f"\nInitial parameters: w = {w.data:.3f}, b = {b.data:.3f}")
    
    # Forward pass for one data point
    def linear_model(x_val, weight, bias):
        """Simple linear model: y = w*x + b"""
        x = Value(x_val)
        return weight * x + bias
    
    # Test forward pass
    print("\nTesting forward pass:")
    for i in range(3):  # Test first 3 samples
        pred = linear_model(X_sample[i], w, b)
        target = y_sample[i]
        loss = pred.mse(target)
        print(f"Sample {i}: pred = {pred.data:.3f}, target = {target:.3f}, loss = {loss.data:.3f}")
    
    # Test backward pass
    print("\nTesting backward pass:")
    pred = linear_model(X_sample[0], w, b)
    loss = pred.mse(y_sample[0])
    
    print(f"Before backward: w.grad = {w.grad:.6f}, b.grad = {b.grad:.6f}")
    loss.backward()
    print(f"After backward: w.grad = {w.grad:.6f}, b.grad = {b.grad:.6f}")
    
    return True

def test_manual_gradient_descent():
    """Test manual gradient descent with multiple iterations"""
    
    # Simple dataset
    X_data = [1.0, 2.0, 3.0, 4.0]
    y_data = [2.0, 4.0, 6.0, 8.0]  # y = 2*x (perfect linear relationship)
    
    # Parameters
    w = Value(0.1)
    b = Value(0.1)
    learning_rate = 0.01
    
    print("Manual Gradient Descent Test")
    print("True relationship: y = 2*x")
    print(f"Initial: w = {w.data:.3f}, b = {b.data:.3f}")
    
    # Training loop
    for epoch in range(10):
        total_loss = 0
        
        # Reset gradients
        w.grad = 0
        b.grad = 0
        
        # Forward and backward for each sample
        for i in range(len(X_data)):
            # Forward pass
            x = Value(X_data[i])
            pred = w * x + b
            loss = pred.mse(y_data[i])
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.data
        
        avg_loss = total_loss / len(X_data)
        
        # Update parameters
        w.data -= learning_rate * w.grad
        b.data -= learning_rate * b.grad
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}, w = {w.data:.3f}, b = {b.data:.3f}")
    
    print(f"Final: w = {w.data:.3f}, b = {b.data:.3f}")
    print("Expected: w ≈ 2.0, b ≈ 0.0")

if __name__ == "__main__":
    print("="*50)
    print("Testing Linear Regression Implementation")
    print("="*50)
    
    test_basic_linear_regression()
    
    print("\n" + "="*50)
    test_manual_gradient_descent()
