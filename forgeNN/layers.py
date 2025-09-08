"""
Layer building blocks: base Layer, ActivationWrapper, and Sequential container.

This module provides a simple, clean API for stacking layers using a
Sequential model and for attaching activations via the @ operator.

Usage example:
    >>> import forgeNN as nn
    >>> model = nn.Sequential([
    ...     nn.Dense(128) @ 'relu',
    ...     nn.Flatten(),
    ...     nn.Dense(10) @ 'softmax',
    ... ])

Notes:
    - Activations can be strings ('relu', 'tanh', 'sigmoid', 'swish', 'linear'),
      activation classes (RELU, TANH, etc.), or callables taking a Tensor.
    - Parameters are collected from all layers to work with VectorizedOptimizer.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Union

from .tensor import Tensor
from .vectorized import ACTIVATION_FUNCTIONS  # Reuse unified activation mapping


ActivationType = Union[str, type, Callable[[Tensor], Tensor]]


class Layer:
    """Base class for layers.

    Subclasses should implement forward(x) and, optionally, backward(dout).

    Examples:
        >>> class Identity(Layer):
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return x
        ...
        >>> layer = Identity()
        >>> out = layer(Tensor([[1., 2.]]))
        >>> out.shape
        (1, 2)
    """

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the layer to input tensor ``x``.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after the layer's forward computation.
        """
        return self.forward(x)

    # Allow attaching activation: Layer @ "relu" -> ActivationWrapper(layer, "relu")
    def __matmul__(self, activation: ActivationType) -> "ActivationWrapper":
        """Return an activation-wrapped version of this layer.

        Example:
            >>> dense = Dense(8)
            >>> wrapped = dense @ 'relu'
            >>> isinstance(wrapped, ActivationWrapper)
            True
        """
        return ActivationWrapper(self, activation)

    # Default: non-parametric
    def parameters(self) -> List[Tensor]:
        """Return trainable parameters (override in subclasses).

        Returns:
            A list of Tensors to be optimized.
        """
        return []

    def num_parameter_tensors(self) -> int:
        """Return the number of parameter tensors.

        Example:
            >>> # Typically 2 per Dense layer (W, b)
            >>> # so a 3-layer MLP often yields 6 tensors total.
            >>> # Use ``num_parameters()`` for total scalar count instead.
        """
        return len(self.parameters())

    def num_parameters(self) -> int:
        """Return the total number of learnable scalars across all parameters.

        Notes:
            For lazily initialized layers (e.g., Dense without ``in_features``),
            this may be 0 until the first forward pass initializes weights.
        """
        return sum(p.data.size for p in self.parameters())

    # Optional in advanced layers
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface only
        """Forward pass of the layer. Must be implemented by subclasses."""
        raise NotImplementedError


class ActivationWrapper(Layer):
    """Wrap a layer and apply an activation after its forward pass.

    Supports string, activation class, or callable activations.

    Example:
        >>> layer = Dense(4) @ 'relu'
        >>> out = layer(Tensor([[1., 2., 3., 4.]]))
        >>> out.shape
        (1, 4)
    """

    def __init__(self, layer: Layer, activation: ActivationType):
        self.layer = layer
        self.activation = activation

    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply the configured activation to tensor ``x``.

        Supports:
            - String activations registered in the shared mapping
            - Activation classes or instances (e.g., RELU)
            - Arbitrary callables: ``fn(Tensor) -> Tensor``
        """
        # Direct callable (not a class): fn(Tensor) -> Tensor
        if callable(self.activation) and not isinstance(self.activation, type):
            if hasattr(self.activation, 'forward'):
                return self.activation.forward(x)  # activation class instance
            return self.activation(x)

        # String or known activation types via shared mapping
        if self.activation in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[self.activation](x)

        # Instance of activation class
        if type(self.activation) in ACTIVATION_FUNCTIONS:
            return ACTIVATION_FUNCTIONS[type(self.activation)](x)

        # Fallback: method on Tensor
        if hasattr(x, str(self.activation)):
            return getattr(x, str(self.activation))()

        raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: layer(x) followed by activation."""
        return self._apply_activation(self.layer(x))

    def parameters(self) -> List[Tensor]:
        return self.layer.parameters()


class Sequential(Layer):
    """Container that applies layers in sequence.

    Args:
        layers (Sequence[Layer]): Layers to apply in order. Can include
            ActivationWrapper instances created via the @ operator.

    Examples:
        >>> model = Sequential([
        ...     Dense(8) @ 'relu',
        ...     Flatten(),
        ...     Dense(10) @ 'softmax',
        ... ])
        >>> x = Tensor([[1., 2., 3., 4., 5., 6., 7., 8.]])
        >>> model(x).shape
        (1, 10)
    """

    def __init__(self, layers: Sequence[Layer]):
        self.layers: List[Layer] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layers in order to input ``x``.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after all layers.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self) -> List[Tensor]:
        """Collect trainable parameters from all sub-layers."""
        params: List[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero in-place."""
        for p in self.parameters():
            p.zero_grad()


class Dense(Layer):
    """Fully-connected (linear) layer with optional lazy initialization.

    Args:
        out_features (int): Number of output features.
        in_features (Optional[int]): If provided, initialize immediately; otherwise
            infer from the first input at runtime.

    Examples:
        >>> dense = Dense(4)  # lazy input dim
        >>> y = dense(Tensor([[1., 2., 3.]]))  # in_features inferred as 3
        >>> y.shape
        (1, 4)
    """

    def __init__(self, out_features: int, in_features: Optional[int] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.W: Optional[Tensor] = None
        self.b: Optional[Tensor] = None

        if in_features is not None:
            self._init_params(in_features)

    def _init_params(self, in_features: int) -> None:
        """Initialize weights with Xavier/Glorot uniform and zero bias."""
        import numpy as np
        fan_in, fan_out = in_features, self.out_features
        limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
        self.W = Tensor(
            np.random.uniform(-limit, limit, (in_features, self.out_features)),
            requires_grad=True,
        )
        self.b = Tensor([0.0] * self.out_features, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ W + b`` with lazy parameter initialization.

        If ``in_features`` was omitted, the first call infers it from
        ``x.shape[-1]`` and initializes parameters accordingly.
        """
        if self.W is None or self.b is None:
            # Infer on first use; assume shape (N, D)
            self._init_params(x.shape[-1])
        return x @ self.W + self.b

    def parameters(self) -> List[Tensor]:
        """Return the weight and bias tensors (if initialized)."""
        return [p for p in (self.W, self.b) if p is not None]


class Flatten(Layer):
    """Flatten all dimensions except the batch dimension.

    Examples:
        >>> x = Tensor([[1., 2.], [3., 4.]])
        >>> Flatten()(x).shape
        (2, 2)
    """

    def forward(self, x: Tensor) -> Tensor:
        """Flatten all non-batch dimensions.

        If input is already 2D or less, returns ``x`` unchanged.
        """
        if len(x.shape) <= 2:
            return x
        batch = x.shape[0]
        return x.view(batch, -1)

    def parameters(self) -> List[Tensor]:
        """Flatten has no trainable parameters."""
        return []


# Optional placeholders for future convolutional/pooling layers.
# These are provided for API completeness and can be implemented later.

class Conv2D(Layer):  # pragma: no cover - placeholder
    """2D convolution layer (placeholder).

    Args:
        filters (int): Number of output channels.
        kernel_size (int): Kernel size (assumes square kernels).
        input_shape (Optional[tuple]): Expected input shape (H, W, C) for first layer.

    Notes:
        This placeholder exposes parameters() and the Layer API but does not
        implement the convolution math yet.
    """

    def __init__(self, filters: int, kernel_size: int, input_shape: Optional[tuple] = None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        # Parameters would be initialized lazily when implemented

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Conv2D forward is not implemented yet.")

    def parameters(self) -> List[Tensor]:
        return []


class MaxPool2D(Layer):  # pragma: no cover - placeholder
    """2D max pooling layer (placeholder)."""

    def __init__(self, pool_size: int):
        self.pool_size = pool_size

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("MaxPool2D forward is not implemented yet.")

    def parameters(self) -> List[Tensor]:
        return []
