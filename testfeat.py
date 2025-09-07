from forgeNN.tensor import Tensor
from forgeNN.vectorized import VectorizedMLP, VectorizedOptimizer, cross_entropy_loss, accuracy
import numpy as np

def main():
    # Implements and tests new features in Tensor class.(version 1.0.3)
    # 1.0.3 - Added reshape functionality
    randomval = 2
    x = Tensor(np.random.randn(2, 3))
    print("Original shape:", x.shape)
    print("Original size :", x.size)
    print("Original data:\n", x.data)
    x_reshaped = x.reshape(-1, randomval)
    print("Reshaped:", x_reshaped.shape)
    print("Reshaped data:\n", x_reshaped.data)
    print("Reshaped size :", x_reshaped.size)
    # Check backward pass
    x_reshaped.sum().backward()
    print("Gradient after backward:\n", x.grad)
    print("Gradient shape:", x.grad.shape)
    print("Gradient matches original shape:", x.grad.shape == x.shape)
    print("="*60)

    # Test with larger tensor
    large_x = Tensor(np.random.randn(100, 50))
    reshaped = large_x.reshape(1, -1)
    print(f"Large reshape: {large_x.shape} → {reshaped.shape}")


    # Test edge cases
    try:
        x.reshape(-1, -1)  # Should raise error
    except ValueError as e:
        print("Multiple -1s error:", e)

    # Test empty tensor
    empty = Tensor(np.array([]))
    empty_reshaped = empty.reshape(0, 5)
    print("Empty tensor shape:", empty_reshaped.shape)

if __name__ == "__main__":
    main()