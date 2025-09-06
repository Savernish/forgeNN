"""
Performance Benchmark: Scalar vs Vectorized forgeNN
=================================================

Direct comparison between scalar and vectorized implementations.
"""

import time
import numpy as np
from sklearn.datasets import make_classification

# Test data generation
def generate_test_data(n_samples=1000, n_features=20, n_classes=3):
    """Generate synthetic classification data for benchmarking."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features//2,
        n_redundant=0,
        random_state=42
    )
    X = X.astype(np.float32)
    return X, y

def benchmark_scalar():
    """Benchmark the original scalar implementation."""
    from forgeNN.core import Value
    from forgeNN.network import MLP
    
    print("Testing Scalar Implementation...")
    X, y = generate_test_data(n_samples=500, n_features=20, n_classes=3)  # Smaller dataset
    
    model = MLP(20, [10, 5, 3])
    
    start_time = time.time()
    
    # Train for a few steps
    for i in range(min(50, len(X))):  # Limited iterations for fair comparison
        x = [Value(float(val)) for val in X[i]]
        target = y[i]
        
        # Forward pass
        logits = model(x)
        
        # Simple loss (just MSE with one-hot target)
        target_one_hot = [1.0 if j == target else 0.0 for j in range(3)]
        loss = sum((logits[j] - Value(target_one_hot[j]))**2 for j in range(3))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Simple update
        for p in model.parameters():
            p.data -= 0.01 * p.grad
    
    scalar_time = time.time() - start_time
    samples_processed = min(50, len(X))
    
    return scalar_time, samples_processed

def benchmark_vectorized():
    """Benchmark the vectorized implementation."""
    from forgeNN.tensor import Tensor
    from forgeNN.vectorized import VectorizedMLP, VectorizedOptimizer, cross_entropy_loss
    
    print("Testing Vectorized Implementation...")
    X, y = generate_test_data(n_samples=1000, n_features=20, n_classes=3)
    
    model = VectorizedMLP(20, [10, 5], 3, ['relu', 'relu', 'linear'])
    optimizer = VectorizedOptimizer(model.parameters(), lr=0.01)
    
    start_time = time.time()
    
    # Process in batches
    batch_size = 32
    samples_processed = 0
    
    for i in range(0, min(1000, len(X)), batch_size):
        end_idx = min(i + batch_size, len(X))
        batch_x = X[i:end_idx]
        batch_y = y[i:end_idx]
        
        x_tensor = Tensor(batch_x)
        logits = model(x_tensor)
        loss = cross_entropy_loss(logits, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        samples_processed += len(batch_x)
        
        if samples_processed >= 1000:  # Process more samples since it's faster
            break
    
    vectorized_time = time.time() - start_time
    
    return vectorized_time, samples_processed

def main():
    print("="*60)
    print("forgeNN PERFORMANCE BENCHMARK")
    print("="*60)
    
    print("\\nRunning benchmarks on synthetic classification data...")
    
    # Benchmark scalar version
    try:
        scalar_time, scalar_samples = benchmark_scalar()
        scalar_rate = scalar_samples / scalar_time
        print(f"\\n📊 Scalar Implementation:")
        print(f"   Time: {scalar_time:.2f} seconds")
        print(f"   Samples: {scalar_samples}")
        print(f"   Rate: {scalar_rate:.1f} samples/second")
    except Exception as e:
        print(f"\\n❌ Scalar benchmark failed: {e}")
        scalar_time, scalar_samples, scalar_rate = None, None, None
    
    # Benchmark vectorized version
    try:
        vectorized_time, vectorized_samples = benchmark_vectorized()
        vectorized_rate = vectorized_samples / vectorized_time
        print(f"\\n🚀 Vectorized Implementation:")
        print(f"   Time: {vectorized_time:.2f} seconds")
        print(f"   Samples: {vectorized_samples}")
        print(f"   Rate: {vectorized_rate:.1f} samples/second")
    except Exception as e:
        print(f"\\n❌ Vectorized benchmark failed: {e}")
        vectorized_time, vectorized_samples, vectorized_rate = None, None, None
    
    # Compare performance
    if scalar_rate and vectorized_rate:
        speedup = vectorized_rate / scalar_rate
        print(f"\\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"\\n🏆 Speedup: {speedup:.1f}x faster")
        print(f"\\n💡 Key Improvements:")
        print(f"   • Vectorized operations with NumPy")
        print(f"   • Batch processing instead of sample-by-sample")
        print(f"   • Optimized matrix multiplications")
        print(f"   • Reduced Python overhead")
        
        print(f"\\n📈 Practical Impact:")
        if speedup > 50:
            print(f"   • Training that took 1 hour now takes {60/speedup:.1f} minutes")
        elif speedup > 10:
            print(f"   • Training that took 10 minutes now takes {10*60/speedup:.0f} seconds")
        else:
            print(f"   • {speedup:.1f}x reduction in training time")
        
        print(f"   • Can handle {vectorized_samples}+ samples vs {scalar_samples} samples")
        print(f"   • Enables larger models and datasets")
    
    print(f"\\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("\\n✅ Use vectorized implementation for:")
    print("   • Production training")
    print("   • Large datasets (>1000 samples)")
    print("   • Performance-critical applications")
    print("   • Experimentation with larger models")
    
    print("\\n📚 Use scalar implementation for:")
    print("   • Learning automatic differentiation")
    print("   • Understanding backpropagation")
    print("   • Educational purposes")
    print("   • Debugging gradient computations")

if __name__ == "__main__":
    main()
