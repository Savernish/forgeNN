from forgeNN.core import Value
import numpy as np
from sklearn.datasets import make_regression

def test_proper_gradient_descent():
    """Test proper gradient descent with gradient averaging"""
    
    # Simple dataset
    X_data = [1.0, 2.0, 3.0, 4.0]
    y_data = [2.0, 4.0, 6.0, 8.0]  # y = 2*x
    
    # Parameters
    w = Value(0.1)
    b = Value(0.1)
    learning_rate = 0.1
    
    print("Proper Gradient Descent Test")
    print("True relationship: y = 2*x")
    print(f"Initial: w = {w.data:.3f}, b = {b.data:.3f}")
    
    # Training loop
    for epoch in range(50):
        total_loss = 0
        grad_w_sum = 0
        grad_b_sum = 0
        
        # Forward and backward for each sample
        for i in range(len(X_data)):
            # Reset gradients for this sample
            w.grad = 0
            b.grad = 0
            
            # Forward pass
            x = Value(X_data[i])
            pred = w * x + b
            loss = pred.mse(y_data[i])
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            grad_w_sum += w.grad
            grad_b_sum += b.grad
            total_loss += loss.data
        
        # Average gradients
        avg_grad_w = grad_w_sum / len(X_data)
        avg_grad_b = grad_b_sum / len(X_data)
        avg_loss = total_loss / len(X_data)
        
        # Update parameters with averaged gradients
        w.data -= learning_rate * avg_grad_w
        b.data -= learning_rate * avg_grad_b
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {avg_loss:.6f}, w = {w.data:.3f}, b = {b.data:.3f}")
    
    print(f"Final: w = {w.data:.3f}, b = {b.data:.3f}")
    print("Expected: w ≈ 2.0, b ≈ 0.0")
    
    # Test predictions
    print("\nTest predictions:")
    for x_val in [1.0, 2.0, 3.0, 4.0, 5.0]:
        x = Value(x_val)
        pred = w * x + b
        expected = 2.0 * x_val
        print(f"x = {x_val}, pred = {pred.data:.3f}, expected = {expected:.3f}")

if __name__ == "__main__":
    test_proper_gradient_descent()
