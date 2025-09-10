"""
Core Tensor implementation.

This is the primary vectorized autodiff engine for forgeNN.
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
    
	# Internal: sum gradient to a target shape (handles broadcasting efficiently)
	@staticmethod
	def _sum_to_shape(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
		# Reduce leading extra dimensions
		while grad.ndim > len(shape):
			grad = grad.sum(axis=0)
		# Reduce broadcasted dims
		for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
			if sdim == 1 and gdim > 1:
				grad = grad.sum(axis=i, keepdims=True)
		return grad
    
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
				self.grad += Tensor._sum_to_shape(out.grad, self.data.shape)
			if other.requires_grad:
				other.grad += Tensor._sum_to_shape(out.grad, other.data.shape)
        
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
				self.grad += Tensor._sum_to_shape(grad, self.data.shape)
			if other.requires_grad:
				grad = self.data * out.grad
				other.grad += Tensor._sum_to_shape(grad, other.data.shape)
        
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
		"""ReLU activation: max(0, x)."""
		out_data = np.maximum(0, self.data)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='relu')
        
		def _backward():
			if self.requires_grad:
				self.grad += (self.data > 0).astype(np.float32) * out.grad
        
		out._backward = _backward
		return out
    
	def gelu(self):
		"""GELU activation (tanh approximation)."""
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
		"""Sigmoid activation: 1 / (1 + exp(-x))."""
		out_data = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='sigmoid')
        
		def _backward():
			if self.requires_grad:
				sigmoid_grad = out_data * (1 - out_data)
				self.grad += sigmoid_grad * out.grad
        
		out._backward = _backward
		return out
    
	def tanh(self):
		"""Hyperbolic tangent activation."""
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
		"""Vectorized Leaky ReLU activation."""
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
		"""Vectorized Swish activation: x * sigmoid(beta * x)."""
		sigmoid_input = beta * self.data
		sigmoid_data = 1 / (1 + np.exp(-np.clip(sigmoid_input, -500, 500)))
		out_data = self.data * sigmoid_data
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op=f'swish({beta})')
        
		def _backward():
			if self.requires_grad:
				swish_grad = sigmoid_data + self.data * beta * sigmoid_data * (1 - sigmoid_data)
				self.grad += swish_grad * out.grad
        
		out._backward = _backward
		return out
    
	def sum(self, axis=None, keepdims=False):
		"""Sum elements along an axis."""
		out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='sum')
        
		def _backward():
			if self.requires_grad:
				grad = out.grad
				if axis is not None:
					if not keepdims:
						grad = np.expand_dims(grad, axis)
					grad = np.broadcast_to(grad, self.data.shape)
				else:
					grad = np.broadcast_to(grad, self.data.shape)
				self.grad += grad
        
		out._backward = _backward
		return out
    
	def mean(self, axis=None, keepdims=False):
		"""Mean along an axis."""
		out_data = np.mean(self.data, axis=axis, keepdims=keepdims)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='mean')
        
		def _backward():
			if self.requires_grad:
				grad = out.grad
				if axis is not None:
					axes = axis if isinstance(axis, tuple) else (axis,)
					if not keepdims:
						for ax in sorted(axes):
							grad = np.expand_dims(grad, axis=ax)
					grad = np.broadcast_to(grad, self.data.shape)
					denom = 1
					for ax in axes:
						denom *= self.data.shape[ax]
					grad = grad / float(denom)
				else:
					grad = np.broadcast_to(grad, self.data.shape) / float(self.data.size)
				self.grad += grad
        
		out._backward = _backward
		return out
    
	def dot(self, other) -> 'Tensor':
		"""Vectorized dot product for 1D tensors."""
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
		"""Reshape tensor to new shape."""
		if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
			new_shape = tuple(new_shape[0])
		else:
			new_shape = tuple(new_shape)
        
		if new_shape.count(-1) > 1:
			raise ValueError("Only one dimension can be inferred (-1)")
        
		if self.size == 0:
			return Tensor(self.data.reshape(new_shape), requires_grad=self.requires_grad)
        
		if -1 in new_shape:
			non_neg_dims = [d for d in new_shape if d != -1]
			if 0 in non_neg_dims:
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
				reshaped_grad = out.grad.reshape(self.data.shape)
				self.grad += reshaped_grad
		out._backward = _backward
		return out

	def flatten(self) -> 'Tensor':
		"""Flatten to 1D."""
		return self.reshape(-1)

	def view(self, *new_shape) -> 'Tensor':
		"""View tensor with a new shape without copying memory."""
		if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
			new_shape = tuple(new_shape[0])
		else:
			new_shape = tuple(new_shape)
        
		if new_shape.count(-1) > 1:
			raise ValueError("Only one dimension can be inferred (-1)")
        
		if self.size == 0:
			return Tensor(self.data.view().reshape(new_shape), requires_grad=self.requires_grad)
        
		if not self.data.flags['C_CONTIGUOUS']:
			raise RuntimeError("view() requires contiguous tensor. Use reshape() instead or call contiguous() first.")
        
		if -1 in new_shape:
			non_neg_dims = [d for d in new_shape if d != -1]
			if 0 in non_neg_dims:
				if self.size == 0:
					inferred_dim = 0
				else:
					raise ValueError(f"Cannot view tensor of size {self.size} to shape with zero dimension")
			else:
				inferred_dim = int(self.size // np.prod(non_neg_dims))
			new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)
		if np.prod(new_shape) != self.size:
			raise ValueError(f"Cannot view tensor of size {self.size} to shape {new_shape}")
        
		out_data = self.data.view()
		out_data = out_data.reshape(new_shape)
        
		out = Tensor.__new__(Tensor)
		out.data = out_data
		out.grad = np.zeros_like(out.data) if self.requires_grad else None
		out.requires_grad = self.requires_grad
		out.shape = out.data.shape
		out.size = out.data.size
		out._children = set((self,))
		out._op = 'view'
		out._backward = lambda: None
        
		def _backward():
			if self.requires_grad:
				reshaped_grad = out.grad.reshape(self.data.shape)
				self.grad += reshaped_grad
        
		out._backward = _backward
		return out

	def contiguous(self) -> 'Tensor':
		"""Return a contiguous tensor with the same data."""
		if self.data.flags['C_CONTIGUOUS']:
			return self
		contiguous_data = np.ascontiguousarray(self.data)
		out = Tensor(contiguous_data, requires_grad=self.requires_grad,
					_children=(self,), _op='contiguous')
        
		def _backward():
			if self.requires_grad:
				self.grad += out.grad
        
		out._backward = _backward
		return out
        
	def transpose(self, *axes) -> 'Tensor':
		"""Transpose tensor dimensions."""
		if len(axes) == 0:
			axes = tuple(reversed(range(len(self.shape))))
		elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
			axes = tuple(axes[0])
		else:
			axes = tuple(axes)
            
		out_data = self.data.transpose(axes)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='transpose')

		def _backward():
			if self.requires_grad:
				inverse_axes = tuple(np.argsort(axes))
				self.grad += out.grad.transpose(inverse_axes)

		out._backward = _backward
		return out

	def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
		"""Remove dimensions of size 1."""
		if dim is None:
			out_data = np.squeeze(self.data)
			squeezed_dims = [i for i, s in enumerate(self.shape) if s == 1]
		else:
			if dim < 0:
				dim += len(self.shape)
			if dim < 0 or dim >= len(self.shape):
				raise IndexError(f"Dimension {dim} out of range for tensor with {len(self.shape)} dimensions")
			if self.shape[dim] != 1:
				raise ValueError(f"Cannot squeeze dimension {dim} with size {self.shape[dim]}.")
			out_data = np.squeeze(self.data, axis=dim)  
			squeezed_dims = [dim]
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='squeeze')
        
		def _backward():
			if self.requires_grad:
				if dim is None:
					grad = out.grad
					for d in sorted(squeezed_dims):
						grad = np.expand_dims(grad, axis=d)
					self.grad += grad
				else:
					self.grad += np.expand_dims(out.grad, axis=dim)
        
		out._backward = _backward
		return out
    
	def unsqueeze(self, dim: int) -> 'Tensor':
		"""Add dimension of size 1 at specified position."""
		if dim < 0:
			dim = len(self.shape) + 1 + dim
		if dim < 0 or dim > len(self.shape):
			raise IndexError(f"Dimension {dim} out of range for inserting into tensor with {len(self.shape)} dimensions")
		out_data = np.expand_dims(self.data, axis=dim)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='unsqueeze')
		def _backward():
			if self.requires_grad:
				self.grad += np.squeeze(out.grad, axis=dim)
		out._backward = _backward
		return out
        
	def mse_loss(self, target):
		"""Mean Squared Error loss."""
		target = self._ensure_tensor(target)
		diff = self - target
		loss = (diff * diff).mean()
		return loss
    
	def cross_entropy_loss(self, targets):
		"""Cross-entropy loss for integer class targets."""
		shifted_logits = self - self.max(axis=1, keepdims=True)
		log_probs = shifted_logits - shifted_logits.exp().sum(axis=1, keepdims=True).log()
		batch_size = self.data.shape[0]
		selected_log_probs = log_probs.data[np.arange(batch_size), targets]
		loss = -np.mean(selected_log_probs)
		return Tensor(loss, requires_grad=self.requires_grad)
    
	def softmax(self, axis=-1):
		"""Softmax over a given axis."""
		shifted = self - self.max(axis=axis, keepdims=True)
		exp_vals = shifted.exp()
		return exp_vals / exp_vals.sum(axis=axis, keepdims=True)
    
	def exp(self):
		"""Element-wise exponential."""
		out_data = np.exp(np.clip(self.data, -500, 500))
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='exp')
        
		def _backward():
			if self.requires_grad:
				self.grad += out_data * out.grad
        
		out._backward = _backward
		return out
    
	def log(self):
		"""Element-wise natural logarithm."""
		out_data = np.log(np.clip(self.data, 1e-8, None))
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='log')
        
		def _backward():
			if self.requires_grad:
				self.grad += (1.0 / np.clip(self.data, 1e-8, None)) * out.grad
        
		out._backward = _backward
		return out
    
	def max(self, axis=None, keepdims=False):
		"""Maximum along an axis."""
		out_data = np.max(self.data, axis=axis, keepdims=keepdims)
		out = Tensor(out_data, requires_grad=self.requires_grad,
					_children=(self,), _op='max')
        
		def _backward():
			if self.requires_grad:
				eps = 1e-8
				if axis is None:
					m = np.max(self.data)
					mask = (self.data == m).astype(np.float32)
					denom = np.sum(mask) + eps
					self.grad += mask * out.grad / denom
				else:
					expanded_max = np.expand_dims(out_data, axis) if not keepdims else out_data
					mask = (self.data == expanded_max).astype(np.float32)
					expanded_grad = np.expand_dims(out.grad, axis) if not keepdims else out.grad
					denom = np.sum(mask, axis=axis, keepdims=True) + eps
					self.grad += mask * expanded_grad / denom
        
		out._backward = _backward
		return out
    
	def backward(self):
		"""Run backpropagation from this tensor."""
		topo = []
		visited = set()
        
		def build_topo(tensor):
			if tensor not in visited:
				visited.add(tensor)
				for child in tensor._children:
					build_topo(child)
				topo.append(tensor)
        
		build_topo(self)
        
		if self.grad is None:
			self.grad = np.ones_like(self.data)
		else:
			self.grad.fill(0)
			self.grad += np.ones_like(self.data)
        
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
		other = self._ensure_tensor(other)
		out_data = self.data / other.data
		out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
					 _children=(self, other), _op='/')

		def _backward():
			if self.requires_grad:
				self.grad += Tensor._sum_to_shape(out.grad / other.data, self.data.shape)
			if other.requires_grad:
				other.grad += Tensor._sum_to_shape(-out.grad * (self.data / (other.data ** 2)), other.data.shape)

		out._backward = _backward
		return out
    
	def __rtruediv__(self, other):
		"""Right division: compute other / self."""
		other = self._ensure_tensor(other)
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

