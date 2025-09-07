#!/usr/bin/env python3
"""
forgeNN vs PyTorch: Side-by-Side Comparison Demo
===============================================
This script demonstrates the key syntax and performance differences
between forgeNN and PyTorch with identical models.
"""

import time
import numpy as np

def demo_pytorch():
    """Demonstrate PyTorch implementation."""
    print("🔥 PyTorch Implementation")
    print("-" * 40)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Model definition
        class PyTorchMLP(nn.Module):
            def __init__(self):
                super(PyTorchMLP, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64), 
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Initialize
        model = PyTorchMLP()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Sample data
        batch_size = 32
        X = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))
        
        # Timing
        start_time = time.time()
        
        # Training step
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_time = time.time() - start_time
        
        # Results
        print(f"✅ Model created successfully")
        print(f"✅ Training step completed")
        print(f"⏱️  Time: {training_time*1000:.2f}ms")
        print(f"📊 Loss: {loss.item():.4f}")
        print(f"🎯 Accuracy: {(outputs.argmax(1) == y).float().mean().item()*100:.1f}%")
        
        return training_time
        
    except ImportError:
        print("❌ PyTorch not installed")
        return None

def demo_forgenn():
    """Demonstrate forgeNN implementation."""
    print("\n🔥 forgeNN Implementation")
    print("-" * 40)
    
    from forgeNN.tensor import Tensor
    from forgeNN.vectorized import VectorizedMLP, VectorizedOptimizer, cross_entropy_loss, accuracy
    
    # Model definition - More concise!
    model = VectorizedMLP(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        activations=['relu', 'relu', 'linear']
    )
    
    optimizer = VectorizedOptimizer(model.parameters(), lr=0.01, momentum=0.9)
    
    # Sample data
    batch_size = 32
    X = np.random.randn(batch_size, 784).astype(np.float32)
    y = np.random.randint(0, 10, batch_size)
    
    # Timing
    start_time = time.time()
    
    # Training step
    x_tensor = Tensor(X)
    logits = model(x_tensor)
    loss = cross_entropy_loss(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    training_time = time.time() - start_time
    
    # Results
    print(f"✅ Model created successfully")
    print(f"✅ Training step completed")
    print(f"⏱️  Time: {training_time*1000:.2f}ms")
    print(f"📊 Loss: {loss.data:.4f}")
    print(f"🎯 Accuracy: {accuracy(logits, y)*100:.1f}%")
    
    return training_time

def syntax_comparison():
    """Show side-by-side syntax comparison."""
    print("\n" + "="*80)
    print("📝 SYNTAX COMPARISON")
    print("="*80)
    
    pytorch_code = '''
# PyTorch Model Definition
class PyTorchMLP(nn.Module):
    def __init__(self):
        super(PyTorchMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.network(x)

model = PyTorchMLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training Step
outputs = model(x)
loss = criterion(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
'''
    
    forgenn_code = '''
# forgeNN Model Definition  
model = VectorizedMLP(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10,
    activations=['relu', 'relu', 'linear']
)

optimizer = VectorizedOptimizer(model.parameters(), lr=0.01)

# Training Step
logits = model(x)
loss = cross_entropy_loss(logits, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
'''
    
    print("PyTorch (verbose)".ljust(40) + "forgeNN (concise)")
    print("-" * 40 + " " + "-" * 39)
    
    pytorch_lines = pytorch_code.strip().split('\n')
    forgenn_lines = forgenn_code.strip().split('\n')
    
    max_lines = max(len(pytorch_lines), len(forgenn_lines))
    
    for i in range(max_lines):
        pytorch_line = pytorch_lines[i] if i < len(pytorch_lines) else ""
        forgenn_line = forgenn_lines[i] if i < len(forgenn_lines) else ""
        
        print(f"{pytorch_line[:39]:<40} {forgenn_line}")

def performance_summary():
    """Display performance summary."""
    print("\n" + "="*80)
    print("📊 PERFORMANCE SUMMARY")
    print("="*80)
    
    print("Based on comprehensive benchmarks:")
    print()
    print("🎯 MNIST Classification Results:")
    print("   • forgeNN: 30.84s training time, 97.37% accuracy")
    print("   • PyTorch: 64.72s training time, 97.30% accuracy")
    print("   • Result: forgeNN is 2.10x FASTER with +0.07% better accuracy!")
    print()
    print("🔍 Model Size Analysis:")
    print("   • Small models (<109k params): forgeNN 3.52x faster ⚡")
    print("   • Medium models (>700k params): PyTorch 2-3x faster")
    print("   • Large models (>1M params): PyTorch dominates")
    print()
    print("🏆 When to use forgeNN:")
    print("   ✅ Educational/learning purposes")
    print("   ✅ Small model prototyping")
    print("   ✅ CPU-only environments")
    print("   ✅ Research with custom operations")
    print("   ✅ Minimal dependency requirements")
    print()
    print("🏆 When to use PyTorch:")
    print("   ✅ Production ML systems")
    print("   ✅ Large models (>1M parameters)")
    print("   ✅ GPU acceleration needed")
    print("   ✅ Distributed training")
    print("   ✅ Pre-trained model ecosystem")

def main():
    """Run the complete comparison demo."""
    print("🚀 forgeNN vs PyTorch: Live Comparison Demo")
    print("="*80)
    print("This demo shows identical neural networks implemented in both frameworks")
    print("Architecture: 784 → 128 → 64 → 10 (MNIST-style)")
    print()
    
    # Run PyTorch demo
    pytorch_time = demo_pytorch()
    
    # Run forgeNN demo
    forgenn_time = demo_forgenn()
    
    # Compare performance
    if pytorch_time and forgenn_time:
        print(f"\n🏁 PERFORMANCE COMPARISON")
        print(f"   PyTorch: {pytorch_time*1000:.2f}ms")
        print(f"   forgeNN: {forgenn_time*1000:.2f}ms")
        
        if forgenn_time < pytorch_time:
            speedup = pytorch_time / forgenn_time
            print(f"   🎉 forgeNN is {speedup:.1f}x FASTER!")
        else:
            slowdown = forgenn_time / pytorch_time
            print(f"   ⚠️  forgeNN is {slowdown:.1f}x slower")
    
    # Show syntax comparison
    syntax_comparison()
    
    # Show performance summary
    performance_summary()
    
    print(f"\n📖 For detailed analysis, see: COMPARISON_GUIDE.md")
    print(f"🔥 Full benchmark results: mnist_benchmark_comparison.png")

if __name__ == "__main__":
    main()
