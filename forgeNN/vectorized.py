"""
forgeNN Vectorized Neural Network Module
========================================

High-performance neural network implementations using vectorized operations.
These classes provide significant speedups over the scalar implementations
while maintaining API compatibility.

Classes:
    VectorizedLayer: Efficient layer implementation using matrix operations
    VectorizedMLP: Fast multi-layer perceptron for batch training
    VectorizedOptimizer: SGD optimizer with momentum support
"""

import numpy as np
from .tensor import Tensor
from .optimizers import SGD as VectorizedOptimizer  # Backward compatibility alias
from typing import List, Optional, Union, Callable
from .functions.activation import RELU, LRELU, TANH, SIGMOID, SWISH

# Activation function mapping for unified activation system
ACTIVATION_FUNCTIONS = {
    # String-based activations
    'relu': lambda x: x.relu(),
    'sigmoid': lambda x: x.sigmoid(), 
    'tanh': lambda x: x.tanh(),
    'linear': lambda x: x,
    'lrelu': lambda x: x.leaky_relu(),
    'swish': lambda x: x.swish(),
    
    # Class-based activations (new integration)
    RELU: lambda x: x.relu(),
    LRELU: lambda x: x.leaky_relu(),
    TANH: lambda x: x.tanh(),
    SIGMOID: lambda x: x.sigmoid(),
    SWISH: lambda x: x.swish(),
    
    # Direct callable support
    'function': lambda x, fn: fn(x)
}

class VectorizedLayer:
    """
    Vectorized implementation of a fully-connected neural network layer.
    
    This layer implementation uses matrix operations to process entire batches
    of data simultaneously, providing significant performance improvements
    over the sample-by-sample approach.
    
    Mathematical Operation:
        output = activation(input @ weights + bias)
        
    Where:
        - input: (batch_size, input_features)
        - weights: (input_features, output_features)  
        - bias: (output_features,)
        - output: (batch_size, output_features)
    
    Args:
        input_size (int): Number of input features
        output_size (int): Number of output neurons
        activation (str or class or callable): Activation function. Supports:
            - Strings: 'relu', 'sigmoid', 'tanh', 'linear', 'lrelu', 'swish'
            - Classes: RELU, LRELU, TANH, SIGMOID, SWISH from forgeNN.functions.activation
            - Callable: Any function that takes a Tensor and returns a Tensor
        
    Attributes:
        weights (Tensor): Weight matrix (input_size, output_size)
        bias (Tensor): Bias vector (output_size,)
        activation: Activation function or string identifier
        
    Example:
        >>> # String-based activation names
        >>> layer1 = VectorizedLayer(784, 128, 'relu')
        >>> 
        >>> # Activation class instances
        >>> layer2 = VectorizedLayer(128, 64, LRELU)
        >>> 
        >>> # Custom function
        >>> layer3 = VectorizedLayer(64, 10, lambda x: x.sigmoid())
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: Union[str, type, Callable] = 'linear'):
        """Initialize layer with Xavier/Glorot weight initialization."""
        # Xavier initialization for better training dynamics
        fan_in = input_size
        fan_out = output_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        self.weights = Tensor(
            np.random.uniform(-limit, limit, (input_size, output_size)),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(output_size),
            requires_grad=True
        )
        self.activation = activation
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass through the layer.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).

        Returns:
            Tensor: Output of shape (batch_size, output_size).

        Example:
            >>> layer = VectorizedLayer(4, 3, 'relu')
            >>> out = layer(Tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]))
            >>> out.shape
            (2, 3)
        """
        # Linear transformation: x @ W + b
        output = x @ self.weights + self.bias
        
        # Apply activation function using unified system
        return self._apply_activation(output)
    
    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply activation function in a unified way.

        Supports strings (e.g., 'relu'), activation classes (RELU, etc.),
        instances with .forward(x), or callables.
        """
        if callable(self.activation) and not isinstance(self.activation, type):
            # Direct callable (lambda, function) or activation class instance
            if hasattr(self.activation, 'forward'):
                # Activation class instance (e.g., RELU(), SWISH())
                return self.activation.forward(x)
            else:
                # Regular callable (lambda, function)
                return self.activation(x)
        elif self.activation in ACTIVATION_FUNCTIONS:
            # String or class-based activation
            return ACTIVATION_FUNCTIONS[self.activation](x)
        elif type(self.activation) in ACTIVATION_FUNCTIONS:
            # Instance of activation class - get the class and apply
            return ACTIVATION_FUNCTIONS[type(self.activation)](x)
        elif hasattr(x, str(self.activation)):
            # Method name on tensor (e.g., 'relu', 'sigmoid')
            return getattr(x, str(self.activation))()
        else:
            raise ValueError(f"Unknown activation: {self.activation}. "
                           f"Supported: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        return [self.weights, self.bias]

class VectorizedMLP:
    """
    Vectorized Multi-Layer Perceptron for efficient batch training.
    
    This implementation processes entire batches of data simultaneously,
    providing dramatic speedups over sample-by-sample training while
    maintaining the same mathematical operations.
    
    Key Performance Features:
    - Matrix operations instead of loops
    - Efficient memory usage with in-place operations
    - Vectorized activation functions
    - Batch gradient computation
    
    Args:
        input_size (int): Input feature dimensionality
        hidden_sizes (List[int]): List of hidden layer sizes
        output_size (int): Output dimensionality
        activations (List[str or class or callable], optional): Activation per layer. 
            
            **Recommended: Use string names for simplicity and consistency**
            - Strings: 'relu', 'sigmoid', 'tanh', 'linear', 'leaky_relu', 'swish'
            - Classes: RELU(), LRELU(), TANH(), SIGMOID(), SWISH() (advanced control)
            - Callables: Custom functions (maximum flexibility)
            - Mixed: Can combine different types, but strings are preferred
        
    Example:
        >>> # String-based activation names (recommended)
        >>> model = VectorizedMLP(784, [128, 64], 10, ['relu', 'swish', 'linear'])
        >>> 
        >>> # Activation class instances (advanced control)
        >>> model = VectorizedMLP(784, [128, 64], 10, [RELU(), SWISH(), None])
        >>> 
        >>> # Custom functions (maximum flexibility)
        >>> model = VectorizedMLP(784, [128, 64], 10, ['relu', lambda x: x.swish(beta=2.0), 'sigmoid'])
        >>> 
        >>> # Batch forward pass
        >>> batch_x = Tensor(np.random.randn(32, 784))  # 32 samples
        >>> logits = model(batch_x)  # Shape: (32, 10)
        >>> 
        >>> # Compute loss and gradients
        >>> loss = logits.cross_entropy_loss(batch_targets)
        >>> loss.backward()
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 activations: Optional[List[Union[str, type, Callable]]] = None):
        """Initialize vectorized MLP with specified architecture."""
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Default activations: ReLU for hidden, linear for output
        if activations is None:
            activations = ['relu'] * len(hidden_sizes) + ['linear']
        
        assert len(activations) == len(hidden_sizes) + 1, \
            f"Need {len(hidden_sizes) + 1} activations, got {len(activations)}"
        
        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = VectorizedLayer(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i]
            )
            self.layers.append(layer)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass through the entire network.

        Args:
            x (Tensor): Input of shape (batch_size, input_size).

        Returns:
            Tensor: Output logits of shape (batch_size, output_size).

        Example:
            >>> model = VectorizedMLP(8, [16], 4, ['relu', 'linear'])
            >>> model(Tensor([[1]*8, [2]*8])).shape
            (2, 4)
        """
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters from all layers.

        Returns:
            list[Tensor]: Weights and biases for each layer.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all parameters.

        Example:
            >>> model = VectorizedMLP(4, [3], 2, ['relu', 'linear'])
            >>> x = Tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
            >>> y = model(x).sum(); y.backward()
            >>> model.zero_grad()  # clears in-place
        """
        for param in self.parameters():
            param.zero_grad()

class VectorizedOptimizer:
    """
    Notes:
        **Deprecated in v1.2.2:** Use `forgeNN.optimizers.SGD` for consistency.
        This class is kept for backward compatibility only and may be removed in future releases.
    
    Simple SGD optimizer for vectorized training.
    
    Implements stochastic gradient descent with optional momentum
    for efficient parameter updates on vectorized models.
    
    Args:
        parameters (List[Tensor]): Model parameters to optimize
        lr (float): Learning rate. Defaults to 0.01
        momentum (float): Momentum factor. Defaults to 0.0
        
    Example:
        >>> model = VectorizedMLP(784, [128], 10)
        >>> optimizer = VectorizedOptimizer(model.parameters(), lr=0.01)
        >>> 
        >>> # Training step
        >>> loss = compute_loss(model, batch_x, batch_y)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        """Initialize optimizer with parameters and hyperparameters."""
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        
        # Initialize momentum buffers
        if momentum > 0:
            self.momentum_buffers = [np.zeros_like(p.data) for p in parameters]
        else:
            self.momentum_buffers = None
    
    def step(self):
        """Perform one optimization step.

        Applies SGD update with optional momentum.
        """
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            if self.momentum > 0:
                # Apply momentum
                self.momentum_buffers[i] = (
                    self.momentum * self.momentum_buffers[i] + param.grad
                )
                update = self.momentum_buffers[i]
            else:
                update = param.grad
            
            # Update parameters
            param.data -= self.lr * update
    
    def zero_grad(self):
        """Reset gradients for all parameters.

        Example:
            >>> import numpy as np
            >>> from forgeNN.tensor import Tensor
            >>> model = VectorizedMLP(3, [5], 2)
            >>> opt = VectorizedOptimizer(model.parameters(), lr=0.1)
            >>> Tensor(np.random.randn(4, 3))  # simulate usage
            array([...])
        """
        for param in self.parameters:
            param.zero_grad()

def mse(logits: Tensor, targets: Union[np.ndarray, Tensor]) -> Tensor:
    """
    Compute Mean Squared Error (MSE) loss.

    This returns the mean over all elements, i.e., mean((logits - targets)^2).
    Gradients are handled by the existing Tensor operations, ensuring
    correct scaling over batch and feature dimensions and supporting broadcasting.
    
        Args:
                logits (Tensor): Predictions of shape (N, D, ...)
                targets (ndarray | Tensor):
                        - If same shape as logits (or broadcastable), used directly.
                        - If 1D integer class indices and logits has shape (N, C) with C>1,
                            targets are automatically one-hot encoded to (N, C).
                        - If 1D floating values and logits has shape (N, 1) they are reshaped to (N,1).
        
    Returns:
        Tensor: Scalar loss value connected to logits for backprop.

    Example:
        >>> preds = Tensor([[0.5, 0.2], [0.1, 0.4]])
        >>> y = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> loss = mse(preds, y)
        >>> loss.backward()
        >>> preds.grad.shape
        (2, 2)
    """
    t = targets if isinstance(targets, Tensor) else Tensor(np.asarray(targets), requires_grad=False)

    if isinstance(t, Tensor) and t.data.ndim == 1 and logits.data.ndim >= 2:
        batch = logits.data.shape[0]
        # Case A: binary/regression logits with singleton non-batch dims (N,1,...)
        if all(d == 1 for d in logits.data.shape[1:]):
            t = Tensor(t.data.reshape((batch,) + logits.data.shape[1:]), requires_grad=False)
        # Case B: multi-class classification logits (N,C) and integer class labels
        elif len(logits.data.shape) == 2:
            C = logits.data.shape[1]
            # Heuristic: if targets look like class indices (integers within [0, C)) -> one-hot
            labels = t.data.astype(int)
            if labels.min() >= 0 and labels.max() < C:
                one_hot = np.zeros((batch, C), dtype=logits.data.dtype)
                one_hot[np.arange(batch), labels] = 1.0
                t = Tensor(one_hot, requires_grad=False)
    diff = logits - t
    return (diff * diff).mean()

def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """
    Compute cross-entropy loss for classification with numerical stability.
    
    Args:
        logits (Tensor): Raw model outputs (batch_size, num_classes)
        targets (np.ndarray): Class indices (batch_size,)
        
    Returns:
        Tensor: Scalar loss value connected to logits for backprop.

    Example:
        >>> logits = Tensor([[1., 0.5], [0.2, 0.8]])
        >>> y = np.array([0, 1])
        >>> loss = cross_entropy_loss(logits, y)
        >>> loss.backward()
        >>> logits.grad.shape
        (2, 2)
    """
    # Ensure targets are a contiguous int array
    targets = np.asarray(targets, dtype=np.int64)
    data = logits.data
    # Cast internally to float32 for speed (keep original reference if already)
    if data.dtype != np.float32:
        data32 = data.astype(np.float32, copy=False)
        # If we had to cast we still need autograd linkage, so operate through Tensor ops
        # (casting outside graph would detach). Only cast the underlying array if safe.
        logits.data = data32  # type: ignore

    batch_size = data.shape[0]

    # Numerical stability: subtract per-row max
    max_per_row = logits.max(axis=1, keepdims=True)  # Tensor
    shifted = logits - max_per_row  # Tensor

    # exp & sum
    exp_shifted = shifted.exp()  # Tensor
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)  # Tensor
    # probabilities = exp / sum
    probs = exp_shifted / sum_exp  # Tensor (softmax output)

    # We need log probs for selected targets: log(softmax) = shifted - log(sum_exp)
    log_sum_exp = sum_exp.log()
    log_probs = shifted - log_sum_exp  # Tensor

    batch_idx = np.arange(batch_size)
    selected = log_probs.data[batch_idx, targets]
    loss_value = -float(np.mean(selected))

    # Create scalar loss tensor referencing logits for gradient flow
    loss = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op='cross_entropy')

    if logits.requires_grad:
        # Capture needed arrays once
        probs_data = probs.data  # shape (B,C)
        targets_local = targets
        bsz = batch_size

        def _backward():
            # Gradient of cross-entropy w.r.t logits: (probs - one_hot)/B
            grad = probs_data.copy()
            grad[batch_idx, targets_local] -= 1.0
            grad /= bsz
            logits.grad += grad * loss.grad

        loss._backward = _backward

    return loss

def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits (Tensor): Model predictions (batch_size, num_classes)
        targets (np.ndarray): True class indices (batch_size,)
        
    Returns:
        float: Accuracy as fraction between 0 and 1

    Example:
        >>> logits = Tensor([[1.0, 3.0], [2.0, 0.1]])
        >>> targets = np.array([1, 0])
        >>> round(accuracy(logits, targets), 2)
        1.0
    """
    predictions = np.argmax(logits.data, axis=1)
    return np.mean(predictions == targets)

def accuracy_counts(logits: Tensor, targets: np.ndarray) -> tuple[int, int]:
    """Return (correct, total) for exact aggregation over dataset.

    This helper is not used by default APIs but can be used where
    tighter aggregation is needed.
    """
    predictions = np.argmax(logits.data, axis=1)
    correct = int(np.sum(predictions == targets))
    total = int(len(targets))
    return correct, total
