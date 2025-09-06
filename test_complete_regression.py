from forgeNN.core import Value
from forgeNN.network import MLP
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def test_linear_regression_with_sklearn():
    """Test linear regression using sklearn dataset and compare with sklearn's LinearRegression"""
    
    # Generate dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Normalize the data for better training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_norm = scaler_X.fit_transform(X).flatten()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Split into train/test
    split = 80
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y_norm[:split], y_norm[split:]
    
    print("Linear Regression with sklearn dataset")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 1. Train with sklearn for comparison
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train.reshape(-1, 1), y_train)
    sklearn_pred = sklearn_model.predict(X_test.reshape(-1, 1))
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    print(f"\nSklearn LinearRegression:")
    print(f"  Weight: {sklearn_model.coef_[0]:.6f}")
    print(f"  Bias: {sklearn_model.intercept_:.6f}")
    print(f"  Test MSE: {sklearn_mse:.6f}")
    
    # 2. Train with our implementation
    print(f"\nOur Implementation:")
    
    # Create a simple linear model (1 input, 1 output, no hidden layers)
    model = MLP(1, [1], ['linear'])  # Linear regression
    
    learning_rate = 0.01
    epochs = 1000
    
    print(f"Initial parameters: {len(model.parameters())} total")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        
        # Collect gradients for all parameters
        param_grads = []
        
        for param in model.parameters():
            param.grad = 0  # Reset gradient
        
        # Forward and backward for each training sample
        for i in range(len(X_train)):
            # Reset gradients for this sample
            for param in model.parameters():
                param.grad = 0
            
            # Forward pass
            pred = model([Value(X_train[i])])
            if isinstance(pred, list):
                pred = pred[0]
            
            # Compute loss
            loss = pred.mse(y_train[i])
            total_loss += loss.data
            
            # Backward pass
            loss.backward()
            
            # Store gradients for averaging
            if i == 0:  # Initialize gradient storage
                param_grads = [param.grad for param in model.parameters()]
            else:  # Accumulate gradients
                for j, param in enumerate(model.parameters()):
                    param_grads[j] += param.grad
        
        # Average gradients and update parameters
        avg_loss = total_loss / len(X_train)
        for j, param in enumerate(model.parameters()):
            avg_grad = param_grads[j] / len(X_train)
            param.data -= learning_rate * avg_grad
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")
    
    # Get final parameters
    params = model.parameters()
    weight = params[0].data
    bias = params[1].data
    
    print(f"  Final Weight: {weight:.6f}")
    print(f"  Final Bias: {bias:.6f}")
    
    # Test our model
    test_predictions = []
    for x_val in X_test:
        pred = model([Value(x_val)])
        if isinstance(pred, list):
            pred = pred[0]
        test_predictions.append(pred.data)
    
    our_mse = np.mean([(pred - true)**2 for pred, true in zip(test_predictions, y_test)])
    print(f"  Test MSE: {our_mse:.6f}")
    
    print(f"\nComparison:")
    print(f"  MSE Difference: {abs(our_mse - sklearn_mse):.6f}")
    print(f"  Weight Difference: {abs(weight - sklearn_model.coef_[0]):.6f}")
    print(f"  Bias Difference: {abs(bias - sklearn_model.intercept_):.6f}")

def test_nonlinear_regression():
    """Test nonlinear regression with hidden layers"""
    
    print("\n" + "="*60)
    print("Testing Nonlinear Regression (MLP with hidden layer)")
    print("="*60)
    
    # Generate nonlinear dataset
    np.random.seed(42)
    X = np.linspace(-2, 2, 100)
    y = X**2 + 0.5*X + 0.1*np.random.randn(100)  # Quadratic relationship
    
    # Normalize
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    
    # Split
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create MLP: 1 input -> 5 hidden (ReLU) -> 1 output (linear)
    model = MLP(1, [5, 1], ['relu', 'linear'])
    
    learning_rate = 0.05
    epochs = 500
    
    print(f"Model: 1 -> 5 (ReLU) -> 1 (linear)")
    print(f"Parameters: {len(model.parameters())}")
    
    # Training loop (simplified for demonstration)
    for epoch in range(epochs):
        total_loss = 0
        
        # Mini-batch of size 1 (stochastic gradient descent)
        for i in range(len(X_train)):
            # Reset gradients
            for param in model.parameters():
                param.grad = 0
            
            # Forward pass
            pred = model([Value(X_train[i])])
            if isinstance(pred, list):
                pred = pred[0]
            
            # Loss
            loss = pred.mse(y_train[i])
            total_loss += loss.data
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        
        if epoch % 100 == 0:
            avg_loss = total_loss / len(X_train)
            print(f"  Epoch {epoch}: loss = {avg_loss:.6f}")
    
    # Test
    test_loss = 0
    for i in range(len(X_test)):
        pred = model([Value(X_test[i])])
        if isinstance(pred, list):
            pred = pred[0]
        test_loss += (pred.data - y_test[i])**2
    
    test_mse = test_loss / len(X_test)
    print(f"  Test MSE: {test_mse:.6f}")

if __name__ == "__main__":
    test_linear_regression_with_sklearn()
    test_nonlinear_regression()
