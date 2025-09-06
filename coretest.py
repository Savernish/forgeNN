from forgeNN.core import Value
from forgeNN.network import Neuron, Layer, MLP

def test_all_activations():
    print("=== Testing All Activation Functions ===\n")
    
    # Test values
    test_val = 1.0
    x = Value(test_val)
    
    print(f"Input: {test_val}\n")
    
    # Test all activation functions
    activations = [
        ("ReLU", x.relu()),
        ("Leaky ReLU", x.lrelu(alpha=0.1)),
        ("Tanh", x.tanh()),
        ("Sigmoid", x.sigmoid()),
        ("Swish", x.swish())
    ]
    
    for name, result in activations:
        print(f"{name:12}: {result.data:.6f}")
    
    print("\n" + "="*40)
    print("Testing with negative input: -2.0\n")
    
    # Test with negative value
    x_neg = Value(-2.0)
    activations_neg = [
        ("ReLU", x_neg.relu()),
        ("Leaky ReLU", x_neg.lrelu(alpha=0.1)),
        ("Tanh", x_neg.tanh()),
        ("Sigmoid", x_neg.sigmoid()),
        ("Swish", x_neg.swish())
    ]
    
    for name, result in activations_neg:
        print(f"{name:12}: {result.data:.6f}")
    
    print("\n" + "="*40)
    print("Testing gradient computation\n")
    
    # Test gradients
    x_grad = Value(0.5)
    y = x_grad.sigmoid()
    z = y * Value(2.0)
    z.backward()
    
    print(f"x = {x_grad.data}, sigmoid(x) = {y.data:.6f}")
    print(f"z = sigmoid(x) * 2 = {z.data:.6f}")
    print(f"x.grad = {x_grad.grad:.6f}")

def demonstrate_chaining():
    print("\n" + "="*50)
    print("=== Demonstrating Activation Function Chaining ===\n")
    
    # Create a simple computation chain with multiple activations
    x = Value(0.5)
    print(f"Starting with x = {x.data}")
    
    # Chain different activations
    h1 = x.tanh()
    print(f"h1 = tanh(x) = {h1.data:.6f}")
    
    h2 = h1 * Value(2.0)
    print(f"h2 = h1 * 2 = {h2.data:.6f}")
    
    h3 = h2.sigmoid()
    print(f"h3 = sigmoid(h2) = {h3.data:.6f}")
    
    # Compute gradients
    h3.backward()
    print(f"\nAfter backward pass:")
    print(f"x.grad = {x.grad:.6f}")


if __name__ == "__main__":
    test_all_activations()
    demonstrate_chaining()
    a = Value(2.0)
    b = Value(3.0)
    c = a * (b + a)
    d = c ** 2
    print(d)
    print(d._prev) # Output: (Value(6.0), Value(2.0))
