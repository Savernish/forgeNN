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

from typing import Callable, Iterable, List, Optional, Sequence, Union, Tuple

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
        if not self.layers:
            raise ValueError("Sequential requires at least one layer")
        # Building is now *symbolic / optional*. Real params for Dense can still be lazily
        # initialized at first forward. We keep a flag if we have attempted a symbolic build.
        self._built_symbolic = False

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

    def _build(self) -> None:
        """Symbolically build the network by initializing layer parameters.

        Uses the shape information from the Input layer to initialize Dense layers
        that were defined without explicit in_features. No lazy initialization will
        occur later; Dense layers must be ready after construction.
        """
        input_layer = self.layers[0]
        assert isinstance(input_layer, Input)
        current_shape = (None,) + tuple(input_layer.shape)
        for layer in self.layers[1:]:
            # Unwrap activation wrapper for shape / init handling
            target = layer.layer if isinstance(layer, ActivationWrapper) else layer
            if isinstance(target, Flatten):
                if len(current_shape) > 2:
                    prod = 1
                    for d in current_shape[1:]:
                        prod *= d
                    current_shape = (None, prod)
                # else unchanged
            elif isinstance(target, Dense):
                if target.W is None:
                    # Determine in_features from current_shape
                    if len(current_shape) == 2 and current_shape[1] is not None:
                        in_feats = current_shape[1]
                    else:
                        # Flatten implicit multi-d input
                        prod = 1
                        for d in current_shape[1:]:
                            prod *= d
                        in_feats = prod
                    target._init_params(in_feats)
                current_shape = (None, target.out_features)
            # Other layer types (conv/pool) not implemented: keep shape conservative
        self._built = True

        def summary(self, input_shape: Optional[Sequence[int]] = None) -> None:
                """Print a Keras-like model summary.

                Behavior:
                        - If an Input layer is present (anywhere, typically first), its declared
                            shape seeds symbolic shape propagation.
                        - If ``input_shape`` argument is provided it overrides and seeds
                            propagation (batch dim implicit as None).
                        - Dense layers with known input feature size (either via existing
                            weights or explicit ``in_features`` or inferred current shape) will
                            show concrete output shapes and parameter counts.
                        - Dense layers without resolvable input size remain *unbuilt* until
                            first real forward; they display output shape='?' and Param#=0.
                        - Flatten collapses known trailing dimensions when they are all
                            concrete; otherwise outputs '(None, ?)' until resolvable.

                Args:
                        input_shape: Optional tuple excluding batch dimension to seed shape
                                inference when no Input layer exists.
                """
                # Column widths
                col_layer = 30
                col_shape = 28
                header_line = "=" * (col_layer + col_shape + 11)
                print(header_line)
                print(f"{'Layer (type)':<{col_layer}}{'Output Shape':<{col_shape}}Param #")
                print(header_line)

                # Seed shape: from explicit arg or first Input layer
                current_shape: Optional[Tuple[Optional[int], ...]] = None
                if input_shape is not None:
                        current_shape = (None,) + tuple(input_shape)
                else:
                        for lyr in self.layers:
                                if isinstance(lyr, Input):
                                        current_shape = (None,) + tuple(lyr.shape)
                                        break

                total_params = 0
                total_tensors = 0

                for layer in self.layers:
                        display_name = self._format_layer_name(layer)

                        # Unwrap for logic but keep wrapper display
                        core = layer.layer if isinstance(layer, ActivationWrapper) else layer
                        out_shape_str = '?'

                        # Symbolic inference if we have current shape
                        if current_shape is not None:
                                inferred = self._infer_next_shape_symbolic(core, current_shape)
                                if inferred is not None:
                                        current_shape = inferred
                                        out_shape_str = str(current_shape)
                        else:
                                # If layer itself provides a static output (Input)
                                if isinstance(core, Input):
                                        current_shape = (None,) + tuple(core.shape)
                                        out_shape_str = str(current_shape)

                        # Parameter computation (may conditionally initialize Dense if resolvable)
                        if isinstance(core, Dense) and core.W is None and current_shape is not None:
                                # Try build only if last dim known
                                if len(current_shape) == 2 and current_shape[1] is not None:
                                        core._init_params(current_shape[1])
                                        # Refresh out shape
                                        current_shape = (None, core.out_features)
                                        out_shape_str = str(current_shape)

                        p = layer.num_parameters()
                        t = layer.num_parameter_tensors()
                        total_params += p
                        total_tensors += t

                        print(f"{display_name:<{col_layer}}{out_shape_str:<{col_shape}}{p:>7}")

                print(header_line)
                print(f"Total params: {total_params}")
                print(f"Total parameter tensors: {total_tensors}")
                print(header_line)

    def _format_layer_name(self, layer: Layer) -> str:
        """Return a concise layer display name including activation if wrapped."""
        if isinstance(layer, ActivationWrapper):
            # Show underlying and activation
            act = layer.activation
            if callable(act) and not isinstance(act, type):
                act_name = getattr(act, '__name__', act.__class__.__name__)
            elif isinstance(act, type):
                act_name = act.__name__
            else:
                act_name = str(act)
            base_name = layer.layer.__class__.__name__
            return f"{base_name}({act_name})"
        return layer.__class__.__name__

    def _safe_layer_output_shape(self, layer: Layer) -> str:  # kept for backward compat
        if isinstance(layer, Input):
            return str((None,)+tuple(layer.shape))
        if isinstance(layer, Dense) and getattr(layer, 'W', None) is not None:
            return str((None, layer.W.data.shape[1]))
        if isinstance(layer, Flatten):
            return '(None, ? )'
        if isinstance(layer, ActivationWrapper):
            return self._safe_layer_output_shape(layer.layer)
        return '?'

    def _infer_next_shape_symbolic(self, core: Layer, in_shape: tuple) -> Optional[tuple]:
        if isinstance(core, Input):
            return (None,)+tuple(core.shape)
        if isinstance(core, Flatten):
            if len(in_shape) <= 2:
                return in_shape
            prod = 1
            for d in in_shape[1:]:
                if d is None:
                    return (None, None)
                prod *= d
            return (None, prod)
        if isinstance(core, Dense):
            # Already built
            if core.W is not None:
                return (None, core.W.data.shape[1])
            # If explicit in_features known
            if core.in_features is not None:
                core._init_params(core.in_features)
                return (None, core.out_features)
            # Infer from incoming shape if 2D
            if len(in_shape) == 2 and in_shape[1] is not None:
                core._init_params(in_shape[1])
                return (None, core.out_features)
            return None
        # For activation wrappers handled outside, for others unknown
        return in_shape if isinstance(core, ActivationWrapper) else None

    # Added (or re-added) summary method: ensure it's defined at class scope
    def summary(self, input_shape: Optional[Sequence[int]] = None) -> None:
        """Print a Keras-like model summary.

        Args:
            input_shape: Optional shape (excluding batch) to seed inference if no Input layer.
        """
        col_layer = 30
        col_shape = 28
        header_line = "=" * (col_layer + col_shape + 11)
        print(header_line)
        print(f"{'Layer (type)':<{col_layer}}{'Output Shape':<{col_shape}}Param #")
        print(header_line)

        current_shape: Optional[Tuple[Optional[int], ...]] = None
        if input_shape is not None:
            current_shape = (None,) + tuple(input_shape)
        else:
            for lyr in self.layers:
                core = lyr.layer if isinstance(lyr, ActivationWrapper) else lyr
                if isinstance(core, Input):
                    current_shape = (None,) + tuple(core.shape)
                    break

        total_params = 0
        total_tensors = 0
        for layer in self.layers:
            name = self._format_layer_name(layer)
            core = layer.layer if isinstance(layer, ActivationWrapper) else layer
            shape_str = '?'

            if current_shape is not None:
                inferred = self._infer_next_shape_symbolic(core, current_shape)
                if inferred is not None:
                    current_shape = inferred
                    shape_str = str(current_shape)
            elif isinstance(core, Input):
                current_shape = (None,) + tuple(core.shape)
                shape_str = str(current_shape)

            if isinstance(core, Dense) and core.W is None and current_shape is not None:
                if len(current_shape) == 2 and current_shape[1] is not None:
                    core._init_params(current_shape[1])
                    current_shape = (None, core.out_features)
                    shape_str = str(current_shape)

            p = layer.num_parameters()
            t = layer.num_parameter_tensors()
            total_params += p
            total_tensors += t
            print(f"{name:<{col_layer}}{shape_str:<{col_shape}}{p:>7}")

        print(header_line)
        print(f"Total params: {total_params}")
        print(f"Total parameter tensors: {total_tensors}")
        print(header_line)


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
        # Do not rely on lazy init at forward; will be initialized during Sequential build
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
        """Compute ``x @ W + b`` with lazy initialization if needed."""
        if self.W is None or self.b is None:
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
        """Apply the max pooling operation. """
        raise NotImplementedError("MaxPool2D forward is not implemented yet.")

    def parameters(self) -> List[Tensor]:
        return []


class Input(Layer):
    """Input placeholder layer defining the expected input shape (excluding batch).

    Example:
        >>> model = Sequential([
        ...     Input((784,)),
        ...     Dense(128) @ 'relu',
        ...     Dense(10)
        ... ])
    """

    def __init__(self, shape: Tuple[int, ...]):
        if not isinstance(shape, (tuple, list)) or not shape:
            raise ValueError("Input shape must be a non-empty tuple/list of dimensions")
        self.shape = tuple(int(d) for d in shape)

    def forward(self, x: Tensor) -> Tensor:
        # Optionally validate trailing shape lengths if feasible
        if len(x.shape) - 1 != len(self.shape):
            # Allow mismatch silently (user may feed different rank); skip strictness
            return x
        return x

    def parameters(self) -> List[Tensor]:  # no params
        return []
