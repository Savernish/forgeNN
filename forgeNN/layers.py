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
    - Activations can be strings ('relu', 'tanh', 'sigmoid', 'swish', 'linear', etc.),
      activation classes (RELU, TANH, etc.), or callables taking a Tensor.
    - Parameters are collected from all layers for optimization.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Union, Tuple

from .core.tensor import Tensor, stack
from .nn.activations import ACTIVATION_FUNCTIONS  # v2 unified activation mapping
import numpy as np


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

    # Train/eval toggles (no-op by default unless a layer uses self.training)
    def train(self, flag: bool = True) -> "Layer":
        self.training = bool(flag)
        return self

    def eval(self) -> "Layer":
        return self.train(False)

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
        # Pre-resolve to a callable to reduce per-forward overhead
        act = activation
        fn = None
        if callable(act) and not isinstance(act, type):
            if hasattr(act, 'forward'):
                fn = lambda t, a=act: a.forward(t)
            else:
                fn = act
        elif act in ACTIVATION_FUNCTIONS:
            fn = ACTIVATION_FUNCTIONS[act]
        elif isinstance(act, type) and act in ACTIVATION_FUNCTIONS:
            fn = ACTIVATION_FUNCTIONS[act]
        else:
            name = str(act)
            fn = lambda t, n=name: getattr(t, n)()
        self._fn: Callable[[Tensor], Tensor] = fn

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(self.layer(x))

    def parameters(self) -> List[Tensor]:
        return self.layer.parameters()

    def train(self, flag: bool = True) -> "ActivationWrapper":
        self.training = bool(flag)
        if hasattr(self.layer, 'train'):
            self.layer.train(flag)
        return self


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

    def train(self, flag: bool = True) -> "Sequential":
        self.training = bool(flag)
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(flag)
        return self

    def eval(self) -> "Sequential":
        return self.train(False)

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

    # Removed unused _build() method from v1/v1.5 era to reduce noise and confusion.

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
            return (None,) + tuple(core.shape)
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
        # Default: assume layers we don't explicitly handle are shape-preserving
        # (covers Dropout and similar pass-through layers)
        return in_shape

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

            if isinstance(core, GlobalAvgPool2D):
                if current_shape is not None and len(current_shape) == 4:
                    C = current_shape[1]
                    current_shape = (None, C, 1, 1) if getattr(core, "keepdims", False) else (None, C)
                    shape_str = str(current_shape)

            if isinstance(core, GlobalAvgPool1D):
                if current_shape is not None and len(current_shape) == 3:
                    C = current_shape[1]
                    current_shape = (None, C, 1) if getattr(core, "keepdims", False) else (None, C)
                    shape_str = str(current_shape)

            if isinstance(core, Embedding):
                if current_shape is not None:
                    current_shape = current_shape + (core.embedding_dim,)
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
        fan_in, fan_out = in_features, self.out_features
        limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
        self.W = Tensor(
            np.random.uniform(-limit, limit, (in_features, self.out_features)).astype(np.float32),
            requires_grad=True,
        )
        self.b = Tensor(np.zeros(self.out_features, dtype=np.float32), requires_grad=True)

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

class Dropout(Layer):
    """Inverted Dropout (shape-preserving, activation-noise during training only).

    Args:
        rate (float): Probability to drop an activation in [0, 1).
        seed (int | None): Optional seed for reproducibility.

    Behavior:
        - Training: y = x * mask, where mask ~ Bernoulli(1-rate)/(1-rate)
        - Eval: pass-through (no randomness, no scaling)
    """

    def __init__(self, rate: float = 0.5, seed: Optional[int] = None):
        if not (0.0 <= rate < 1.0):
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = float(rate)
        self.training = True
        self._seed = seed
        # Eagerly create RNG if seed provided; otherwise use global np.random
        self._rng = None
        if seed is not None:
            import numpy as np  # local import
            self._rng = np.random.default_rng(seed)

    def _random(self, shape):
        import numpy as np
        if self._rng is not None:
            return self._rng.random(shape)
        return np.random.random(shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.rate == 0.0 or not getattr(self, 'training', True):
            return x
        p_keep = 1.0 - self.rate
        # Pre-scale mask (0 or 1/p_keep) to avoid extra Tensor division op
        m = (self._random(x.shape) < p_keep).astype(x.data.dtype)
        m /= p_keep
        return x * m

    def parameters(self) -> List[Tensor]:
        return []
    
class Embedding(Layer):
    """Embedding layer mapping discrete indices to dense vectors.

    Args:
        num_embeddings (int): Size of the vocabulary (number of unique indices).
        embedding_dim (int): Dimensionality of each embedding vector.
        padding_idx (int | None): If set, the embedding at this index is initialized
            to zeros and its gradient is kept zeroed.

    Notes:
        - Input x should contain integer indices (Tensor or array-like).
        - Output shape is x.shape + (embedding_dim,).
        - Gradients accumulate correctly even when indices repeat in a batch.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive integers")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx if padding_idx is not None else None

        # Initialize weights (normal with small std, similar to transformer defaults)
        W = np.random.normal(loc=0.0, scale=0.02, size=(self.num_embeddings, self.embedding_dim)).astype(np.float32)
        if self.padding_idx is not None:
            if not (0 <= self.padding_idx < self.num_embeddings):
                raise ValueError(f"padding_idx {self.padding_idx} out of range [0, {self.num_embeddings})")
            W[self.padding_idx].fill(0.0)

        self.W = Tensor(W, requires_grad=True)
        
    def forward(self, x: Union[Tensor, list, tuple]) -> Tensor:
        # Get integer indices as NumPy array
        if isinstance(x, Tensor):
            idx = x.data.astype(np.int64, copy=False)
        else:
            idx = np.asarray(x, dtype=np.int64)

        if idx.size > 0:
            if (idx < 0).any() or (idx >= self.num_embeddings).any():
                raise IndexError("Embedding indices are out of range")

        # Advanced indexing: shape -> idx.shape + (embedding_dim,)
        out_data = self.W.data[idx]

        # Build output tensor that depends on self.W; gradient flows into W only
        out = Tensor(out_data, requires_grad=self.W.requires_grad, _children=(self.W,), _op="embedding")

        def _backward():
            if not self.W.requires_grad:
                return
            grad = out.grad  # shape idx.shape + (embedding_dim,)
            flat_idx = idx.reshape(-1)
            flat_grad = grad.reshape(-1, self.embedding_dim)
            np.add.at(self.W.grad, flat_idx, flat_grad)
            if self.padding_idx is not None:
                self.W.grad[self.padding_idx, :].fill(0.0)

        out._backward = _backward
        return out

    def parameters(self) -> List[Tensor]:
        return [self.W]

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

class GlobalAvgPool2D(Layer):
    """Global average pooling over spatial dims (H, W) for NCHW input."""
    def __init__(self, keepdims: bool = False):
        self.keepdims = bool(keepdims)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError("GlobalAvgPool2D expects input of shape (N, C, H, W)")
        return x.mean(axis=(2, 3), keepdims=self.keepdims)

    def parameters(self) -> List[Tensor]:
        return []
    
class GlobalAvgPool1D(Layer):
    """Global average pooling over temporal dim (L) for NCL input."""
    def __init__(self, keepdims: bool = False):
        self.keepdims = bool(keepdims)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 3:
            raise ValueError("GlobalAvgPool1D expects input of shape (N, C, L)")
        return x.mean(axis=2, keepdims=self.keepdims)

    def parameters(self) -> List[Tensor]:
        return []

class LayerNorm(Layer):
    """Layer normalization over the last dimension.
    Args:
        normalized_shape (int | None): Size of the last dimension to normalize.
            If None, will be lazily initialized on first forward pass.
        eps (float): Small constant for numerical stability.
    Examples:
        >>> ln = LayerNorm(4)
        >>> x = Tensor([[1., 2., 3., 4.], [ 5., 6., 7., 8.]])
        >>> y = ln(x)   # shape (2, 4)
        >>> y.shape
        (2, 4)
    """

    def __init__(self, normalized_shape: Optional[int] = None, eps: float = 1e-5):
        self.eps = float(eps)
        self._dim = int(normalized_shape) if normalized_shape is not None else None
        self.gamma: Optional[Tensor] = None
        self.beta: Optional[Tensor] = None
        if self._dim is not None:
            self._init_params(self._dim)

    def _init_params(self, d: int) -> None:
        self.gamma = Tensor(np.ones(d, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(d, dtype=np.float32), requires_grad=True)
        self._dim = d

    def forward(self, x: Tensor) -> Tensor:
        # Lazy init based on trailing dimension
        d = x.shape[-1]
        if self._dim is None:
            self._init_params(d)
        elif self._dim != d:
            raise ValueError(f"LayerNorm expected last dim {self._dim}, got {d}")

        # Compute mean/var along last dim using Tensor ops (autodiff-friendly)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) * (x - mu)).mean(axis=-1, keepdims=True)
        norm = (x - mu) / (var + self.eps) ** 0.5  # sqrt via pow

        # Affine transform (broadcast gamma/beta)
        y = norm * self.gamma + self.beta
        return y

    def parameters(self) -> List[Tensor]:
        return [p for p in (self.gamma, self.beta) if p is not None]

class Conv2D(Layer):
    """Placeholder for 2D convolution (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Conv2D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("Conv2D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []

class MaxPool2D(Layer):
    """Placeholder for 2D max pooling (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MaxPool2D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("MaxPool2D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []
class AvgPool2D(Layer):
    """Placeholder for 2D average pooling (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AvgPool2D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("AvgPool2D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []

class Conv1D(Layer):
    """Naive 1D convolution over NCL inputs using sliding windows (no padding/dilation).

    Args:
        cin: Input channels.
        cout: Output channels (filters).
        kernel_size: Size of the 1D kernel (>0).
        stride: Stride along length (>=1).
        padding: Currently unsupported (must be 0).
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int = 1, padding: int = 0):
        if kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if padding != 0:
            raise NotImplementedError("Conv1D padding is not implemented yet")

        self.cin = int(cin)
        self.cout = int(cout)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        # Xavier/Glorot uniform init similar to Dense
        fan_in = self.cin * self.kernel_size
        fan_out = self.cout * self.kernel_size
        limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
        W = np.random.uniform(-limit, limit, (self.cout, self.cin, self.kernel_size)).astype(np.float32)
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(np.zeros(self.cout, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, L)
        if len(x.shape) != 3:
            raise ValueError("Conv1D expects input of shape (N, C, L)")
        N, C_in, L = x.shape
        if C_in != self.cin:
            raise ValueError(f"Conv1D: expected {self.cin} input channels, got {C_in}")

        K = self.kernel_size
        s = self.stride
        L_out = (L - K) // s + 1
        if L_out <= 0 or (L - K) < 0:
            raise ValueError(f"Invalid shapes for Conv1D: input length {L}, kernel {K}, stride {s}")

        # Unfold along length into windows: list of (N, C_in, L_out)
        slices: List[Tensor] = []
        for u in range(K):
            sl = x[:, :, u : u + L_out * s : s]
            slices.append(sl)
        # Stack into window dimension -> (N, C_in, L_out, K)
        windows = stack(slices, axis=3)

        # Reshape for batched matmul: (N, L_out, C_in*K)
        X_col = windows.transpose(0, 2, 1, 3).reshape(N, L_out, C_in * K)

        # Weights: (Cout, Cin, K) -> (Cin*K, Cout)
        W_col = self.W.reshape(self.cout, C_in * K).transpose(1, 0)

        # MatMul -> (N, L_out, Cout)
        Y = X_col @ W_col
        # Add bias
        Y = Y + self.b
        # Return (N, Cout, L_out)
        return Y.transpose(0, 2, 1)

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]

class MaxPool1D(Layer):
    """1D max pooling over NCL inputs (no padding/dilation).

    Args:
        kernel_size: Size of the pooling window (>0).
        stride: Stride along length (defaults to kernel_size).
    """

    def __init__(self, kernel_size: int, stride: Optional[int] = None):
        if int(kernel_size) <= 0:
            raise ValueError("kernel_size must be a positive integer")
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        if self.stride <= 0:
            raise ValueError("stride must be >= 1")

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, L)
        if len(x.shape) != 3:
            raise ValueError("MaxPool1D expects input of shape (N, C, L)")
        N, C, L = x.shape
        K = self.kernel_size
        s = self.stride
        L_out = (L - K) // s + 1
        if L_out <= 0 or (L - K) < 0:
            raise ValueError(f"Invalid shapes for MaxPool1D: input length {L}, kernel {K}, stride {s}")

        # Unfold along length into windows: (N, C, L_out, K)
        slices: List[Tensor] = []
        for u in range(K):
            sl = x[:, :, u : u + L_out * s : s]
            slices.append(sl)
        windows = stack(slices, axis=3)
        # Max over window dimension -> (N, C, L_out)
        return windows.max(axis=3, keepdims=False)

    def parameters(self) -> List[Tensor]:
        return []
class AvgPool1D(Layer):
    """Placeholder for 1D average pooling (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AvgPool1D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("AvgPool1D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []

class BatchNorm1D(Layer):
    """Placeholder for 1D batch normalization (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BatchNorm1D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("BatchNorm1D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []

class BatchNorm2D(Layer):
    """Placeholder for 2D batch normalization (not implemented yet)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BatchNorm2D is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError("BatchNorm2D is not implemented yet")

    def parameters(self) -> List[Tensor]:
        return []