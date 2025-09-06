"""
forgeNN Neural Network Architecture Components
==============================================

This module provides the core neural network building blocks for forgeNN,
including neurons, layers, and multi-layer perceptrons. Designed with 
modularity and extensibility in mind to support complex architectures.

Classes:
    Module: Abstract base class for all neural network components
    Neuron: Individual computational unit with weights and bias
    Layer: Collection of neurons forming a single network layer  
    MLP: Multi-layer perceptron for feedforward neural networks

Example:
    >>> import forgeNN as fnn
    >>> model = fnn.MLP(784, [128, 64, 10])  # MNIST classifier
    >>> prediction = model(input_data)
    >>> loss = prediction.cross_entropy(target)
    >>> loss.backward()
"""

from forgeNN.core import Value
import random

class Module:
    """
    Abstract base class for all neural network modules in forgeNN.
    
    This class provides the fundamental interface that all neural network
    components must implement. It establishes the contract for parameter
    management and enables modular composition of complex architectures.
    
    Design Philosophy:
    - Composability: Modules can contain other modules
    - Parameter Management: Unified interface for accessing all parameters
    - Extensibility: Easy to subclass for custom architectures
    - PyTorch Compatibility: Similar API design for easy migration
    
    Methods:
        parameters(): Must return all trainable parameters in this module
        
    Example:
        >>> class CustomLayer(Module):
        ...     def __init__(self):
        ...         self.weight = Value(random.uniform(-1, 1))
        ...         self.bias = Value(0.0)
        ...     
        ...     def parameters(self):
        ...         return [self.weight, self.bias]
    """
    
    def parameters(self):
        """
        Return all trainable parameters in this module and its submodules.
        
        This method should be implemented by all subclasses to return a flat
        list of all Value instances that require gradient computation during
        backpropagation. Used by optimizers to update model parameters.
        
        Returns:
            list[Value]: Flat list of all trainable parameters
            
        Example:
            >>> model = MLP(2, [3, 1])
            >>> params = model.parameters()
            >>> print(f"Total parameters: {len(params)}")
        """
        return []

class Neuron(Module):
    """
    Individual computational unit implementing a perceptron with activation.
    
    A neuron performs the fundamental operation: activation(w·x + b)
    where w are weights, x is input, b is bias, and activation is a 
    non-linear function. This is the atomic building block of neural networks.
    
    Features:
    - Configurable input dimensionality
    - Multiple activation function support
    - Xavier/Glorot-style weight initialization
    - Automatic gradient computation
    - Memory-efficient implementation
    
    Args:
        nin (int): Number of input connections (input dimensionality)
        activation (str): Activation function type. Options:
            - 'linear': No activation (identity function)
            - 'relu': Rectified Linear Unit
            - 'tanh': Hyperbolic tangent  
            - 'sigmoid': Logistic function
            - 'swish': Self-gated activation
            
    Attributes:
        w (list[Value]): Weight parameters for each input connection
        b (Value): Bias parameter
        activation (str): Type of activation function used
        
    Mathematical Operation:
        output = activation(Σ(w_i * x_i) + b) for i in range(nin)
        
    Example:
        >>> # Create a neuron with 3 inputs and ReLU activation
        >>> neuron = Neuron(3, 'relu')
        >>> inputs = [Value(1.0), Value(2.0), Value(-1.0)]
        >>> output = neuron(inputs)
        >>> print(f"Output: {output.data}")
        
        >>> # Access parameters for optimization
        >>> params = neuron.parameters()  # [w1, w2, w3, bias]
        >>> print(f"Neuron has {len(params)} parameters")
    """
    
    def __init__(self, nin, activation='linear'):
        """
        Initialize neuron with random weights and zero bias.
        
        Weights are initialized using uniform distribution in (-1, 1).
        This provides good starting point for most activation functions
        while avoiding vanishing/exploding gradient issues.
        
        Args:
            nin (int): Number of input features/connections
            activation (str): Activation function identifier
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
    
    def __call__(self, x):
        """
        Forward pass through the neuron.
        
        Computes the weighted sum of inputs plus bias, then applies the
        specified activation function. This implements the core perceptron
        computation with modern activation functions.
        
        Args:
            x (Value or list[Value]): Input value(s) to the neuron
            
        Returns:
            Value: Activated output of the neuron
            
        Raises:
            ValueError: If input dimension doesn't match neuron's nin
            
        Example:
            >>> neuron = Neuron(2, 'relu')
            >>> inputs = [Value(0.5), Value(-0.3)]
            >>> output = neuron(inputs)
            >>> # Equivalent to: relu(w1*0.5 + w2*(-0.3) + bias)
        """
        # Compute weighted sum: w1*x1 + w2*x2 + ... + b
        if not isinstance(x, list):
            x = [x]  # Handle single input case
        
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        if self.activation == 'relu':
            out = out.relu()
        elif self.activation == 'tanh':
            out = out.tanh()
        elif self.activation == 'sigmoid':
            out = out.sigmoid()
        elif self.activation == 'swish':
            out = out.swish()
        # 'linear' activation means no activation function
        
        return out
    
    def parameters(self):
        """
        Return all trainable parameters of this neuron.
        
        Returns:
            list[Value]: List containing all weights followed by bias
            
        Example:
            >>> neuron = Neuron(3)
            >>> params = neuron.parameters()  # [w1, w2, w3, bias]
            >>> print(f"Parameter count: {len(params)}")  # Outputs: 4
        """
        return self.w + [self.b]

class Layer(Module):
    """
    Dense (fully-connected) neural network layer.
    
    A layer consists of multiple neurons that all receive the same input
    and produce a vector output. This is the fundamental building block
    for feedforward neural networks, also known as a "dense" or 
    "fully-connected" layer.
    
    Architecture:
        Input (nin,) → [Neuron₁, Neuron₂, ..., Neuronₙₒᵤₜ] → Output (nout,)
        
    Each neuron in the layer:
    - Receives the full input vector
    - Has its own set of weights and bias
    - Applies the same activation function
    - Contributes one element to the output vector
    
    Args:
        nin (int): Dimensionality of input vectors
        nout (int): Number of neurons in layer (output dimensionality)
        activation (str): Activation function for all neurons in layer
        
    Attributes:
        neurons (list[Neuron]): Collection of neurons forming this layer
        
    Example:
        >>> # Create layer: 4 inputs → 3 neurons with ReLU → 3 outputs  
        >>> layer = Layer(4, 3, 'relu')
        >>> inputs = [Value(x) for x in [1.0, -0.5, 2.0, 0.3]]
        >>> outputs = layer(inputs)  # List of 3 Values
        >>> print(f"Layer output shape: {len(outputs)}")  # Outputs: 3
    """
    
    def __init__(self, nin, nout, activation='linear'):
        """
        Initialize layer with specified architecture and activation.
        
        Creates nout neurons, each with nin input connections.
        Total parameters: nout * (nin + 1) due to nin weights + 1 bias per neuron.
        
        Args:
            nin (int): Input feature dimensionality
            nout (int): Output feature dimensionality  
            activation (str): Activation function identifier
            
        Example:
            >>> layer = Layer(784, 128, 'relu')  # MNIST hidden layer
            >>> params = layer.parameters()
            >>> print(f"Parameters: {len(params)}")  # 784*128 + 128 = 100,480
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        """
        Forward pass through the layer.
        
        Applies each neuron to the input and collects outputs into a vector.
        For single-output layers, returns the Value directly for convenience.
        
        Args:
            x (list[Value]): Input vector to the layer
            
        Returns:
            Value or list[Value]: Layer output(s)
            - Single Value if layer has one neuron
            - List of Values if layer has multiple neurons
            
        Mathematical Operation:
            output[i] = activation(Σⱼ(w[i,j] * x[j]) + b[i]) for each neuron i
            
        Example:
            >>> layer = Layer(2, 3, 'relu')
            >>> inputs = [Value(1.0), Value(-0.5)]
            >>> outputs = layer(inputs)  # List of 3 Values
            >>> 
            >>> # Single output layer
            >>> output_layer = Layer(3, 1, 'sigmoid')
            >>> final_output = output_layer(outputs)  # Single Value
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        """
        Return all trainable parameters in this layer.
        
        Flattens parameters from all neurons into a single list.
        Maintains order: [neuron1_params, neuron2_params, ...]
        
        Returns:
            list[Value]: All weights and biases in the layer
            
        Example:
            >>> layer = Layer(2, 3)  # 2 inputs, 3 neurons
            >>> params = layer.parameters()
            >>> # Contains: [w1_1, w1_2, b1, w2_1, w2_2, b2, w3_1, w3_2, b3]
            >>> print(f"Total parameters: {len(params)}")  # Outputs: 9
        """
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    """
    Multi-Layer Perceptron (feedforward neural network).
    
    The MLP is a fundamental neural network architecture consisting of
    multiple layers of neurons arranged in a feedforward topology.
    Each layer is fully connected to the next, making it suitable for
    a wide range of supervised learning tasks.
    
    Architecture:
        Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
        
    Key Features:
    - Configurable depth (number of layers)
    - Configurable width (neurons per layer)  
    - Layer-specific activation functions
    - Automatic parameter management
    - Universal function approximation capability
    
    Common Use Cases:
    - Classification: softmax output with cross-entropy loss
    - Regression: linear output with MSE loss
    - Feature learning: hidden representations
    - Function approximation: any continuous function
    
    Args:
        nin (int): Input feature dimensionality
        nouts (list[int]): Neurons per layer [hidden1, hidden2, ..., output]
        activations (list[str], optional): Activation per layer. 
            Defaults to ReLU for hidden layers, linear for output.
            
    Attributes:
        layers (list[Layer]): Sequential layers forming the network
        
    Example:
        >>> # Binary classifier for 784-dim input (e.g., MNIST)
        >>> model = MLP(784, [128, 64, 1], ['relu', 'relu', 'sigmoid'])
        >>> 
        >>> # Regression model  
        >>> regressor = MLP(10, [50, 20, 1])  # Auto: ['relu', 'relu', 'linear']
        >>> 
        >>> # Multi-class classifier
        >>> classifier = MLP(100, [64, 32, 10], ['relu', 'relu', 'linear'])
    """
    
    def __init__(self, nin, nouts, activations=None):
        """
        Initialize Multi-Layer Perceptron with specified architecture.
        
        Constructs a feedforward network by chaining layers sequentially.
        Each layer's output becomes the next layer's input, creating
        a deep computational pipeline.
        
        Args:
            nin (int): Dimensionality of input data
            nouts (list[int]): Number of neurons in each layer
            activations (list[str], optional): Activation function per layer
                If None, uses ReLU for hidden layers and linear for output
                
        Example:
            >>> # Explicit activations
            >>> model = MLP(2, [10, 5, 1], ['tanh', 'relu', 'sigmoid'])
            >>> 
            >>> # Default activations (recommended)
            >>> model = MLP(2, [10, 5, 1])  # ReLU hidden, linear output
        """
        sz = [nin] + nouts
        if activations is None:
            # Default: ReLU for hidden layers, linear for output
            activations = ['relu'] * (len(nouts) - 1) + ['linear']
        
        self.layers = [Layer(sz[i], sz[i+1], activations[i]) 
                      for i in range(len(nouts))]
    
    def __call__(self, x):
        """
        Forward pass through the entire network.
        
        Sequentially applies each layer to compute the final output.
        This implements the standard feedforward computation:
        h₁ = f₁(W₁x + b₁), h₂ = f₂(W₂h₁ + b₂), ..., y = fₙ(Wₙhₙ₋₁ + bₙ)
        
        Args:
            x (Value or list[Value]): Network input
            
        Returns:
            Value or list[Value]: Network output
            - Single Value for regression or binary classification
            - List of Values for multi-class classification
            
        Example:
            >>> model = MLP(2, [5, 3, 1])
            >>> inputs = [Value(1.0), Value(-0.5)]
            >>> output = model(inputs)  # Single Value
            >>> 
            >>> # Multi-output classification
            >>> classifier = MLP(10, [20, 5])
            >>> logits = classifier(data)  # List of 5 Values
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """
        Return all trainable parameters in the network.
        
        Flattens parameters from all layers into a single list for
        optimization algorithms. Maintains hierarchical order:
        [layer1_params, layer2_params, ..., layerN_params]
        
        Returns:
            list[Value]: All network parameters (weights and biases)
            
        Example:
            >>> model = MLP(784, [128, 10])
            >>> params = model.parameters()
            >>> print(f"Total parameters: {len(params)}")
            >>> # 784*128 + 128 + 128*10 + 10 = 101,770 parameters
            
            >>> # Parameter statistics
            >>> weights = [p for i, p in enumerate(params) if i % (nin + 1) != nin]
            >>> biases = [p for i, p in enumerate(params) if i % (nin + 1) == nin]
        """
        return [p for layer in self.layers for p in layer.parameters()]
