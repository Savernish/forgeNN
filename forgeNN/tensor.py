"""
forgeNN Vectorized Operations Module
====================================

High-performance vectorized implementations using NumPy for batch processing.
This module provides significant speedups for training neural networks while
maintaining the same API as the scalar Value class.

Classes:
    Tensor: Vectorized version of Value for batch operations
    VectorizedMLP: Batch-optimized neural network implementation
"""

import numpy as np
from typing import Union, List, Tuple, Optional

class Tensor:
    """
    Vectorized automatic differentiation engine supporting batch operations.
    
    This class extends the Value concept to handle batches of data efficiently
    using NumPy operations. It maintains the same API as Value but operates
    on arrays instead of scalars for dramatic performance improvements.
    
    Key Features:
    - Batch operations using NumPy
    - Memory-efficient gradient computation
    - Broadcasting support for different tensor shapes
    - Automatic differentiation with vectorized backward passes
    - Drop-in replacement for Value in many use cases
    
    Args:
        data (np.ndarray): The tensor data (any shape)
        requires_grad (bool): Whether to compute gradients. Defaults to True
        _children (tuple): Parent tensors in computation graph
        _op (str): Operation that created this tensor
        
    Attributes:
        data (np.ndarray): The forward pass tensor values
        grad (np.ndarray): The computed gradients (same shape as data)
        requires_grad (bool): Whether gradients are computed
        shape (tuple): Shape of the tensor
        
    Example:
        >>> import numpy as np
        >>> # Batch of 32 samples with 784 features each
        >>> x = Tensor(np.random.randn(32, 784))
        >>> W = Tensor(np.random.randn(784, 128))
        >>> y = x @ W  # Matrix multiplication
        >>> y.backward()  # Compute gradients for entire batch
    """
    
    def __init__(self, data: Union[np.ndarray, float, int], requires_grad: bool = True, 
                 _children: tuple = (), _op: str = ''):
        """Initialize a new Tensor with vectorized operations support."""
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float32)
        elif not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        """Vectorized addition with broadcasting support."""
        other = self._ensure_tensor(other)
        out_data = self.data + other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting by summing over added dimensions
                grad = out.grad
                # Sum out added dims and squeeze broadcasted dimensions
                for i in range(len(out.grad.shape) - len(self.data.shape)):
                    grad = grad.sum(axis=0)
                for i, (dim_out, dim_self) in enumerate(zip(out.grad.shape, self.data.shape)):
                    if dim_self == 1 and dim_out > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = out.grad
                for i in range(len(out.grad.shape) - len(other.data.shape)):
                    grad = grad.sum(axis=0)
                for i, (dim_out, dim_other) in enumerate(zip(out.grad.shape, other.data.shape)):
                    if dim_other == 1 and dim_out > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Vectorized element-wise multiplication."""
        other = self._ensure_tensor(other)
        out_data = self.data * other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                # Handle broadcasting
                for i in range(len(out.grad.shape) - len(self.data.shape)):
                    grad = grad.sum(axis=0)
                for i, (dim_out, dim_self) in enumerate(zip(out.grad.shape, self.data.shape)):
                    if dim_self == 1 and dim_out > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
                
            if other.requires_grad:
                grad = self.data * out.grad
                for i in range(len(out.grad.shape) - len(other.data.shape)):
                    grad = grad.sum(axis=0)
                for i, (dim_out, dim_other) in enumerate(zip(out.grad.shape, other.data.shape)):
                    if dim_other == 1 and dim_out > 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Vectorized matrix multiplication."""
        other = self._ensure_tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        """Vectorized ReLU activation."""
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """Vectorized sigmoid activation."""
        out_data = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))  # Numerical stability
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='sigmoid')
        
        def _backward():
            if self.requires_grad:
                sigmoid_grad = out_data * (1 - out_data)
                self.grad += sigmoid_grad * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        """Vectorized tanh activation."""
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='tanh')
        
        def _backward():
            if self.requires_grad:
                tanh_grad = 1 - out_data**2
                self.grad += tanh_grad * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis."""
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    # Expand gradient back to original shape
                    if not keepdims:
                        grad = np.expand_dims(grad, axis)
                    grad = np.broadcast_to(grad, self.data.shape)
                else:
                    grad = np.broadcast_to(grad, self.data.shape)
                self.grad += grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """Mean of tensor along specified axis."""
        out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='mean')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis)
                    grad = np.broadcast_to(grad, self.data.shape)
                    grad = grad / self.data.shape[axis]
                else:
                    grad = np.broadcast_to(grad, self.data.shape)
                    grad = grad / self.data.size
                self.grad += grad
        
        out._backward = _backward
        return out
    
    def mse_loss(self, target):
        """Vectorized Mean Squared Error loss."""
        target = self._ensure_tensor(target)
        diff = self - target
        loss = (diff * diff).mean()
        return loss
    
    def cross_entropy_loss(self, targets):
        """Vectorized cross-entropy loss for classification."""
        # Apply log-softmax for numerical stability
        shifted_logits = self - self.max(axis=1, keepdims=True)
        log_probs = shifted_logits - shifted_logits.exp().sum(axis=1, keepdims=True).log()
        
        # Select log probabilities for correct classes
        batch_size = self.data.shape[0]
        selected_log_probs = log_probs.data[np.arange(batch_size), targets]
        loss = -np.mean(selected_log_probs)
        
        return Tensor(loss, requires_grad=self.requires_grad)
    
    def softmax(self, axis=-1):
        """Vectorized softmax function."""
        shifted = self - self.max(axis=axis, keepdims=True)
        exp_vals = shifted.exp()
        return exp_vals / exp_vals.sum(axis=axis, keepdims=True)
    
    def exp(self):
        """Element-wise exponential."""
        out_data = np.exp(np.clip(self.data, -500, 500))  # Numerical stability
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='exp')
        
        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """Element-wise natural logarithm."""
        out_data = np.log(np.clip(self.data, 1e-8, None))  # Numerical stability
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='log')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / np.clip(self.data, 1e-8, None)) * out.grad
        
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        """Maximum along specified axis."""
        out_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='max')
        
        def _backward():
            if self.requires_grad:
                if axis is None:
                    mask = (self.data == np.max(self.data)).astype(np.float32)
                    self.grad += mask * out.grad / np.sum(mask)
                else:
                    expanded_max = np.expand_dims(out_data, axis) if not keepdims else out_data
                    mask = (self.data == expanded_max).astype(np.float32)
                    expanded_grad = np.expand_dims(out.grad, axis) if not keepdims else out.grad
                    self.grad += mask * expanded_grad / (np.sum(mask, axis=axis, keepdims=True) + 1e-8)
        
        out._backward = _backward
        return out
    
    def backward(self):
        """Perform backpropagation using topological sorting."""
        topo = []
        visited = set()
        
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._children:
                    build_topo(child)
                topo.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad.fill(0)
            self.grad += np.ones_like(self.data)
        
        # Backpropagate
        for tensor in reversed(topo):
            tensor._backward()
    
    def zero_grad(self):
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad.fill(0)
    
    def _ensure_tensor(self, other):
        """Convert scalar or array to Tensor if needed."""
        if not isinstance(other, Tensor):
            return Tensor(other, requires_grad=False)
        return other
    
    # Operator overloads for convenience
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * Tensor(-1.0, requires_grad=False)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __truediv__(self, other):
        """Element-wise division."""
        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        
        # Division: self / other = self * (1/other)
        # We implement this as self * other^(-1)
        reciprocal = other.__pow__(-1)
        return self * reciprocal
    
    def __rtruediv__(self, other):
        """Right division: other / self"""
        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        return other / self
    
    def __pow__(self, exponent):
        """Element-wise power operation."""
        out_data = np.power(self.data, exponent)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op=f'pow{exponent}')
        
        def _backward():
            if self.requires_grad:
                self.grad += exponent * np.power(self.data, exponent - 1) * out.grad
        
        out._backward = _backward
        return out
