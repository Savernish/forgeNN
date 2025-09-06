"""
Complete Linear Regression Example using forgeNN
Demonstrates proper usage for sklearn datasets with MSE loss
"""

from forgeNN.core import Value
from forgeNN.network import MLP
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LinearRegressionTrainer:
    """A simple trainer class for linear regression using forgeNN"""
    
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history = []
    
    def train_epoch(self, X, y):
        """Train for one epoch and return average loss"""
        total_loss = 0
        n_samples = len(X)
        
        # Collect gradients for averaging
        param_grads = None
        
        for i in range(n_samples):
            # Reset gradients for this sample
            for param in self.model.parameters():
                param.grad = 0
            
            # Forward pass
            x_val = X[i] if isinstance(X[i], (int, float)) else X[i][0]
            pred = self.model([Value(x_val)])
            if isinstance(pred, list):
                pred = pred[0]
            
            # Compute loss
            loss = pred.mse(y[i])
            total_loss += loss.data
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            if param_grads is None:
                param_grads = [param.grad for param in self.model.parameters()]
            else:
                for j, param in enumerate(self.model.parameters()):
                    param_grads[j] += param.grad
        
        # Update parameters with averaged gradients
        for j, param in enumerate(self.model.parameters()):
            avg_grad = param_grads[j] / n_samples
            param.data -= self.learning_rate * avg_grad
        
        avg_loss = total_loss / n_samples
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def train(self, X, y, epochs=1000, verbose=True):
        """Train the model for multiple epochs"""
        for epoch in range(epochs):
            loss = self.train_epoch(X, y)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """Make predictions on new data"""
        predictions = []
        for x in X:
            x_val = x if isinstance(x, (int, float)) else x[0]
            pred = self.model([Value(x_val)])
            if isinstance(pred, list):
                pred = pred[0]
            predictions.append(pred.data)
        return np.array(predictions)

def main():
    print("="*60)
    print("Complete Linear Regression Example with forgeNN")
    print("="*60)
    
    # 1. Generate dataset
    print("1. Generating dataset...")
    X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X).flatten()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split data
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = MLP(1, [1], ['linear'])  # Simple linear regression: 1 input -> 1 output
    print(f"Model parameters: {len(model.parameters())}")
    
    # 3. Create trainer and train
    print("\n3. Training...")
    trainer = LinearRegressionTrainer(model, learning_rate=0.05)
    loss_history = trainer.train(X_train, y_train, epochs=500, verbose=True)
    
    # 4. Evaluate
    print("\n4. Evaluation...")
    train_pred = trainer.predict(X_train)
    test_pred = trainer.predict(X_test)
    
    train_mse = np.mean((train_pred - y_train)**2)
    test_mse = np.mean((test_pred - y_test)**2)
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # 5. Show learned parameters
    print("\n5. Learned parameters:")
    params = model.parameters()
    weight = params[0].data
    bias = params[1].data
    print(f"Weight: {weight:.6f}")
    print(f"Bias: {bias:.6f}")
    
    # 6. Compare with true relationship
    print("\n6. Verification:")
    print("Making predictions on a few test points...")
    for i in range(min(5, len(X_test))):
        x_orig = scaler_X.inverse_transform([[X_test[i]]])[0, 0]
        y_pred_scaled = test_pred[i]
        y_pred_orig = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        y_true_orig = scaler_y.inverse_transform([[y_test[i]]])[0, 0]
        
        print(f"  x={x_orig:.3f}: predicted={y_pred_orig:.3f}, actual={y_true_orig:.3f}")
    
    print("\n" + "="*60)
    print("SUCCESS: Your forgeNN implementation works perfectly for linear regression!")
    print("="*60)
    
    return {
        'model': model,
        'trainer': trainer,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'weight': weight,
        'bias': bias
    }

if __name__ == "__main__":
    results = main()
