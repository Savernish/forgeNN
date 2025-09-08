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
        size (int): Total number of elements in the tensor
        
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
        self.size = self.data.size
    
    def __repr__(self):
        """Return a concise representation including shape and grad flag."""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __add__(self, other):
        """Vectorized addition with NumPy-style broadcasting.

        Args:
            other (Tensor | array-like | float | int): Value to add.

        Returns:
            Tensor: Result with broadcasted shape.

        Examples:
            >>> a = Tensor([[1., 1., 1.], [1., 1., 1.]])
            >>> b = Tensor([1.0, 2.0, 3.0])  # Broadcast across rows
            >>> (a + b).shape
            (2, 3)
        """
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
        """Element-wise multiplication with broadcasting.

        Args:
            other (Tensor | array-like | float | int): Multiplier.

        Returns:
            Tensor: Element-wise product.

        Example:
         >>> x = Tensor([[1., 2.], [3., 4.]])
            >>> (x * 2.0).data
         array([[2., 4.],
             [6., 8.]])
        """
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
        """Matrix multiplication (batch-aware).

        Shapes follow NumPy's @ operator: (N, D) @ (D, M) -> (N, M).

        Args:
            other (Tensor | array-like): Right-hand matrix.

        Returns:
            Tensor: Product tensor.

        Example:
            >>> X = Tensor([[1., 2.], [3., 4.]])
            >>> W = Tensor([[5., 6.], [7., 8.]])
            >>> (X @ W).shape
            (2, 2)
        """
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
        """ReLU activation: max(0, x).

        Example:
            >>> x = Tensor([-1.0, 0.0, 2.0])
            >>> x.relu().data
            array([0., 0., 2.])
        """
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        
        out._backward = _backward
        return out
    
    def gelu(self):
        """GELU activation (tanh approximation).

        Example:
            >>> x = Tensor([-3., -1.5, 0., 1.5, 3.])
            >>> x.gelu().shape
            (5,)
        """
        out_data = 0.5 * self.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data**3)))
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='gelu')

        def _backward():
            if self.requires_grad:
                x = self.data
                tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
                tanh_val = np.tanh(tanh_arg)
                left = 0.5 * tanh_val + 0.5
                right = 0.5 * x * (1 - tanh_val**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
                gelu_grad = left + right
                self.grad += gelu_grad * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid activation: 1 / (1 + exp(-x)).

        Example:
            >>> x = Tensor([-2.0, 0.0, 2.0])
            >>> x.sigmoid().data.round(3)
            array([0.119, 0.5  , 0.881])
        """
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
        """Hyperbolic tangent activation.

        Example:
            >>> x = Tensor([-1.0, 0.0, 1.0])
            >>> x.tanh().data.round(3)
            array([-0.762,  0.   ,  0.762])
        """
        out_data = np.tanh(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='tanh')
        
        def _backward():
            if self.requires_grad:
                tanh_grad = 1 - out_data**2
                self.grad += tanh_grad * out.grad
        
        out._backward = _backward
        return out
    
    def leaky_relu(self, alpha=0.01):
        """Vectorized Leaky ReLU activation.  
        
        Args:
            alpha (float): Slope for negative inputs. Default is 0.01.
        
        Examples:
            x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])    
            y = x.leaky_relu(alpha=0.1)  # Leaky ReLU activation    
            print(y.data) # Output: [-0.2 -0.1  0.   1.   2. ]
        """
        out_data = np.where(self.data > 0, self.data, alpha * self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op=f'leaky_relu({alpha})')
        
        def _backward():
            if self.requires_grad:
                grad_mask = np.where(self.data > 0, 1.0, alpha)
                self.grad += grad_mask * out.grad
        
        out._backward = _backward
        return out
    
    def swish(self, beta=1.0):
        """Vectorized Swish activation: x * sigmoid(beta * x).
        
        Args:
            beta (float): The beta parameter controlling the "steepness" of the sigmoid.
                          Default is 1.0.
        Examples:
            x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  
            y = x.swish(beta=1.0)  # Swish activation  
            print(y.data) # Output: [-0.23840582 -0.26894143  0.          0.7310586   1.7615942 ]
        """
        sigmoid_input = beta * self.data
        sigmoid_data = 1 / (1 + np.exp(-np.clip(sigmoid_input, -500, 500)))
        out_data = self.data * sigmoid_data
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op=f'swish({beta})')
        
        def _backward():
            if self.requires_grad:
                # Swish derivative: sigmoid(βx) + x * β * sigmoid(βx) * (1 - sigmoid(βx))
                swish_grad = sigmoid_data + self.data * beta * sigmoid_data * (1 - sigmoid_data)
                self.grad += swish_grad * out.grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        """Sum elements along an axis.

        Args:
            axis (int | tuple[int] | None): Axis or axes to sum over. If None, sum all elements.
            keepdims (bool): Keep reduced dimensions with length 1.

        Returns:
            Tensor: Reduced tensor.

        Examples:
            >>> x = Tensor([[1., 2.], [3., 4.]])
            >>> x.sum().data
            10.0
            >>> x.sum(axis=0).data
            array([4., 6.])
        """
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
        """Mean along an axis.

        Args:
            axis (int | tuple[int] | None): Axis or axes to average over. If None, global mean.
            keepdims (bool): Keep reduced dimensions with length 1.

        Returns:
            Tensor: Reduced tensor.

        Example:
            >>> x = Tensor([[1., 2.], [3., 5.]])
            >>> x.mean(axis=1).data
            array([1.5, 4. ])
        """
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
    
    def dot(self, other) -> 'Tensor':
        """Vectorized dot product for 1D tensors. It only supports 1D tensors intentionally just like pytorch.

        Args:
            other: The other tensor to perform the dot product with. Must be 1D.

        Returns:
            New tensor with dot product result.
            
        Examples:
            x = Tensor(np.array([1.0, 2.0, 3.0]))  
            y = Tensor(np.array([4.0, 5.0, 6.0]))  
            z = x.dot(y)  # z.data = 32.0
        """
        #Ensure both tensors are 1D
        if len(self.shape) != 1 or len(other.shape) != 1:
            raise ValueError("Dot product is only supported for 1D tensors.")
        if self.shape[0] != other.shape[0]:
            raise ValueError("Tensors must have the same length for dot product.")
        out_data = np.dot(self.data, other.data)
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='dot')
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out


    def reshape(self, *new_shape) -> 'Tensor':   
        """Reshape tensor to new shape.

        Supports -1 to infer a single dimension. Accepts a tuple or multiple ints.

        Args:
            *new_shape: Target shape as a tuple or variadic integers.

        Returns:
            Tensor: Reshaped tensor.

        Examples:
            >>> x = Tensor([0, 1, 2, 3, 4, 5])
            >>> x.reshape(2, 3).shape
            (2, 3)
            >>> x.reshape(-1, 2, 3).shape
            (1, 2, 3)
        """
        # Support both reshape((2,3)) and reshape(2,3)
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)
        
        # Edge case: Multiple -1s in shape (should raise error)
        if new_shape.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred (-1)")
        
        # Edge case: Empty tensor case
        if self.size == 0:
            return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)
        
        if -1 in new_shape:
            # Check for zero dimensions first
            non_neg_dims = [d for d in new_shape if d != -1]
            if 0 in non_neg_dims:
                # If any dimension is 0, the inferred dimension must be 0 (for empty tensor) or error
                if self.size == 0:
                    inferred_dim = 0
                else:
                    raise ValueError(f"Cannot reshape tensor of size {self.size} to shape with zero dimension")
            else:
                inferred_dim = int(self.size // np.prod(non_neg_dims))
            new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)
        if np.prod(new_shape) != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to shape {new_shape}")
        out_data = self.data.reshape(new_shape)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape')
        def _backward():
            if self.requires_grad:
                # Reshape gradient back to original shape
                reshaped_grad = out.grad.reshape(self.data.shape)
                self.grad += reshaped_grad
        out._backward = _backward
        return out

    def flatten(self) -> 'Tensor':
        """Flatten to 1D.

        Example:
            >>> x = Tensor([[1, 2], [3, 4]])
            >>> x.flatten().shape
            (4,)
        """
        return self.reshape(-1)

    def view(self, *new_shape) -> 'Tensor':
        """View tensor with a new shape without copying memory.

        Requires the tensor to be contiguous. Supports -1 to infer one dimension.

        Args:
            *new_shape: Target shape as a tuple or variadic integers.

        Returns:
            Tensor: A view sharing storage with the original.

        Notes:
            Use ``contiguous()`` or ``reshape()`` if the tensor is not contiguous.

        Example:
            >>> x = Tensor([[0, 1, 2], [3, 4, 5]])
            >>> y = x.view(3, 2)
            >>> y.data.base is not None  # shares memory
            True
        """
        # Support both view((2,3)) and view(2,3)
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)
        
        # Edge case: Multiple -1s in shape (should raise error)
        if new_shape.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred (-1)")
        
        # Edge case: Empty tensor case
        if self.size == 0:
            return Tensor(self.data.view().reshape(new_shape), requires_grad=self.requires_grad)
        
        # Check if tensor is contiguous in memory
        if not self.data.flags['C_CONTIGUOUS']:
            raise RuntimeError("view() requires contiguous tensor. Use reshape() instead or call contiguous() first.")
        
        if -1 in new_shape:
            # Check for zero dimensions first
            non_neg_dims = [d for d in new_shape if d != -1]
            if 0 in non_neg_dims:
                # If any dimension is 0, the inferred dimension must be 0 (for empty tensor) or error
                if self.size == 0:
                    inferred_dim = 0
                else:
                    raise ValueError(f"Cannot view tensor of size {self.size} to shape with zero dimension")
            else:
                inferred_dim = int(self.size // np.prod(non_neg_dims))
            new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)
        if np.prod(new_shape) != self.size:
            raise ValueError(f"Cannot view tensor of size {self.size} to shape {new_shape}")
        
        # Create view using shared memory
        out_data = self.data.view()
        out_data = out_data.reshape(new_shape)
        
        # Create output tensor manually to preserve memory sharing
        out = Tensor.__new__(Tensor)  # Create without calling __init__
        out.data = out_data  # Don't copy, use the view directly
        out.grad = np.zeros_like(out.data) if self.requires_grad else None
        out.requires_grad = self.requires_grad
        out.shape = out.data.shape
        out.size = out.data.size
        out._children = set((self,))
        out._op = 'view'
        out._backward = lambda: None
        
        def _backward():
            if self.requires_grad:
                # Reshape gradient back to original shape (same as reshape)
                reshaped_grad = out.grad.reshape(self.data.shape)
                self.grad += reshaped_grad
        
        out._backward = _backward
        return out

    def contiguous(self) -> 'Tensor':
        """Return a contiguous tensor with the same data.
        If the tensor is already contiguous, returns self.
        Otherwise, returns a copy with contiguous memory layout.  
        
        Returns:  
            New contiguous tensor or self if already contiguous.  
        Examples:  
            x = Tensor(np.random.randn(2, 3, 4))
            y = x.contiguous()  # y is a contiguous copy of x
        """
        if self.data.flags['C_CONTIGUOUS']:
            return self  # Already contiguous, no need to copy
        
        # Create contiguous copy
        contiguous_data = np.ascontiguousarray(self.data)
        out = Tensor(contiguous_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='contiguous')
        
        def _backward():
            if self.requires_grad:
                # Simply copy gradients back (same shape)
                self.grad += out.grad
        
        out._backward = _backward
        return out
        
    def transpose(self, *axes) -> 'Tensor':
        """Transpose tensor dimensions.

        Args:
            *axes: Either individual dim indices, a tuple of indices, or nothing
                (reverse all dims).

        Returns:
            Tensor: Transposed tensor.

        Examples:
            tensor.transpose()          # Reverse all dimensions
            tensor.transpose(0, 1)      # Swap dims 0 and 1
            tensor.transpose(2, 0, 1)   # Permute dims
        """
        if len(axes) == 0:
            # Default transpose: reverse all dimensions
            axes = tuple(reversed(range(len(self.shape))))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            # Handle tuple input: transpose((0, 1))
            axes = tuple(axes[0])
        else:
            # Handle individual arguments: transpose(0, 1)
            axes = tuple(axes)
            
        out_data = self.data.transpose(axes)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                # Inverse transpose: if we transposed with axes, 
                # we need to undo it with the inverse permutation
                inverse_axes = tuple(np.argsort(axes))
                self.grad += out.grad.transpose(inverse_axes)

        out._backward = _backward
        return out

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1.
    
        Args:
            dim: If specified, only squeeze this dimension if it has size 1.
                If None, squeeze all dimensions with size 1.
                
        Returns:
            New tensor with size-1 dimensions removed.
            
        Examples:
            x = Tensor(np.ones((3, 1, 4, 1)))
            x.squeeze()      # Shape: (3, 4)
            x.squeeze(1)     # Shape: (3, 4, 1) 
            x.squeeze(3)     # Shape: (3, 1, 4)
        """
        if dim is None:
            # Remove ALL dimensions with size 1
            # Use np.squeeze() without axis parameter
            out_data = np.squeeze(self.data)
            squeezed_dims = [i for i, s in enumerate(self.shape) if s == 1]
        else:
            if dim < 0:
                dim += len(self.shape)
            if dim < 0 or dim >= len(self.shape):
                raise IndexError(f"Dimension {dim} out of range for tensor with {len(self.shape)} dimensions")
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} with size {self.shape[dim]}. Only size-1 dimensions can be squeezed.")
            # Remove the specific dimension
            out_data = np.squeeze(self.data, axis=dim)  
            squeezed_dims = [dim]  # Remember which dim was squeezed for gradient
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='squeeze')
        
        def _backward():
            if self.requires_grad:
                if dim is None:
                    grad = out.grad
                    # Expand dimensions back in correct order - must use ascending order
                    # so that indices don't shift as we add dimensions
                    for d in sorted(squeezed_dims):
                        grad = np.expand_dims(grad, axis=d)
                    self.grad += grad
                else:
                    self.grad += np.expand_dims(out.grad, axis=dim)
        
        out._backward = _backward
        return out
    
    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add dimension of size 1 at specified position.
    
        Args:
            dim: Position where to insert the new dimension.
                Can be negative (counted from the end).
                
        Returns:
            New tensor with size-1 dimension added.
            
        Examples:
            x = Tensor(np.ones((3, 4)))
            x.unsqueeze(0)   # Shape: (1, 3, 4)
            x.unsqueeze(1)   # Shape: (3, 1, 4)
            x.unsqueeze(-1)  # Shape: (3, 4, 1)
        """
        if dim < 0:
            dim = len(self.shape) + 1 + dim
        if dim < 0 or dim > len(self.shape):
            raise IndexError(f"Dimension {dim} out of range for inserting into tensor with {len(self.shape)} dimensions")
        out_data = np.expand_dims(self.data, axis=dim)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='unsqueeze')
        def _backward():
            if self.requires_grad:
                # Remove the dimension we added
                self.grad += np.squeeze(out.grad, axis=dim)
        out._backward = _backward
        return out
        
    def mse_loss(self, target):
        """Mean Squared Error loss.

        Args:
            target (Tensor | array-like): Ground truth values (same shape).

        Returns:
            Tensor: Scalar loss.

        Example:
            >>> import numpy as np
            >>> pred = Tensor(np.array([0.2, 0.7]))
            >>> y = Tensor(np.array([0.0, 1.0]))
            >>> pred.mse_loss(y).data
            0.02499999850988388
        """
        target = self._ensure_tensor(target)
        diff = self - target
        loss = (diff * diff).mean()
        return loss
    
    def cross_entropy_loss(self, targets):
        """Cross-entropy loss for integer class targets.

        Args:
            targets (ndarray | list[int]): Class indices for each row.

        Returns:
            Tensor: Scalar loss (no full graph; see forgeNN.vectorized.cross_entropy_loss for full autodiff).

        Example:
            >>> import numpy as np
            >>> logits = Tensor(np.random.randn(4, 10))
            >>> y = np.array([1, 2, 3, 4])
            >>> loss = logits.cross_entropy_loss(y)
            >>> isinstance(loss, Tensor)
            True
        """
        # Apply log-softmax for numerical stability
        shifted_logits = self - self.max(axis=1, keepdims=True)
        log_probs = shifted_logits - shifted_logits.exp().sum(axis=1, keepdims=True).log()
        
        # Select log probabilities for correct classes
        batch_size = self.data.shape[0]
        selected_log_probs = log_probs.data[np.arange(batch_size), targets]
        loss = -np.mean(selected_log_probs)
        
        return Tensor(loss, requires_grad=self.requires_grad)
    
    def softmax(self, axis=-1):
        """Softmax over a given axis.

        Args:
            axis (int): Axis to normalize over.

        Returns:
            Tensor: Probabilities that sum to 1 along the axis.

        Example:
            >>> x = Tensor([[1., 2.], [0., 0.]])
            >>> probs = x.softmax(axis=1)
            >>> probs.data.sum(axis=1)
            array([1., 1.])
        """
        shifted = self - self.max(axis=axis, keepdims=True)
        exp_vals = shifted.exp()
        return exp_vals / exp_vals.sum(axis=axis, keepdims=True)
    
    def exp(self):
        """Element-wise exponential.

        Example:
            >>> Tensor([0., 1.]).exp().data
            array([1.       , 2.7182817])
        """
        out_data = np.exp(np.clip(self.data, -500, 500))  # Numerical stability
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='exp')
        
        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """Element-wise natural logarithm.

        Example:
            >>> Tensor([1., 2.7182817]).log().data.round(4)
            array([0., 1.])
        """
        out_data = np.log(np.clip(self.data, 1e-8, None))  # Numerical stability
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op='log')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / np.clip(self.data, 1e-8, None)) * out.grad
        
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        """Maximum along an axis.

        Args:
            axis (int | None): Axis to reduce. If None, returns global max as scalar tensor.
            keepdims (bool): Keep reduced dimensions with length 1.

        Returns:
            Tensor: Reduced tensor containing maximum values.

        Example:
            >>> x = Tensor([[1., 5.], [3., 2.]])
            >>> x.max(axis=0).data
            array([3., 5.])
        """
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
        """Run backpropagation from this tensor.

        This treats the tensor as a scalar objective by seeding d(output)/d(output) = 1.

        Example:
            >>> import numpy as np
            >>> x = Tensor(np.array([[1., 2.]]))
            >>> W = Tensor(np.array([[3.], [4.]]))
            >>> y = (x @ W).sum()
            >>> y.backward()
            >>> W.grad.shape
            (2, 1)
        """
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
        """Reset gradients to zero in-place."""
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
        """Element-wise division.

        Example:
            >>> (Tensor([2., 4.]) / 2).data
            array([1., 2.])
        """
        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        
        # Division: self / other = self * (1/other)
        # We implement this as self * other^(-1)
        reciprocal = other.__pow__(-1)
        return self * reciprocal
    
    def __rtruediv__(self, other):
        """Right division: compute other / self."""
        if isinstance(other, (int, float)):
            other = Tensor(other, requires_grad=False)
        return other / self
    
    def __pow__(self, exponent):
        """Element-wise power operation.

        Args:
            exponent (float | int): Exponent to raise each element to.

        Returns:
            Tensor: x ** exponent element-wise.

        Example:
            >>> (Tensor([2., 3.]) ** 2).data
            array([4., 9.])
        """
        out_data = np.power(self.data, exponent)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                    _children=(self,), _op=f'pow{exponent}')
        
        def _backward():
            if self.requires_grad:
                self.grad += exponent * np.power(self.data, exponent - 1) * out.grad
        
        out._backward = _backward
        return out
