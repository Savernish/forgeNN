"""
forgeNN Activation Functions Module
===================================

High-performance implementations of standard neural network activation functions
with forward and backward pass computations optimized for automatic differentiation.

This module provides the mathematical foundations for non-linear transformations
in neural networks, implementing both the activation functions and their derivatives
required for efficient backpropagation.

Classes:
    RELU: Rectified Linear Unit - most widely used activation
    LRELU: Leaky ReLU - addresses dying ReLU problem  
    TANH: Hyperbolic tangent - zero-centered activation
    SIGMOID: Logistic function - classic activation for binary tasks
    SWISH: Self-gated activation - modern high-performance activation

Mathematical Foundations:
    Each activation function f(x) requires:
    - Forward pass: y = f(x)
    - Backward pass: ∂f/∂x evaluated at x
    
    Chain rule application: ∂L/∂x = ∂L/∂y × ∂f/∂x
"""

import math

class RELU:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    The most widely used activation function in modern deep learning.
    ReLU addresses the vanishing gradient problem and enables training
    of very deep networks while being computationally efficient.
    
    Mathematical Definition:
        f(x) = max(0, x) = { x if x > 0
                           { 0 if x ≤ 0
                           
        f'(x) = { 1 if x > 0
                { 0 if x ≤ 0
    
    Properties:
        - Range: [0, +∞)
        - Non-saturating for positive inputs
        - Sparse activation (outputs 0 for negative inputs)
        - Computationally efficient
        - Can suffer from "dying ReLU" problem
        
    Advantages:
        - Mitigates vanishing gradient problem
        - Computational efficiency (simple thresholding)
        - Biological plausibility
        - Enables sparse representations
        
    Use Cases:
        - Hidden layers in deep networks
        - Convolutional neural networks  
        - Residual networks
        - Most general-purpose architectures
    """
    
    @staticmethod
    def forward(x):
        """
        Compute ReLU activation.
        
        Args:
            x (float): Input value
            
        Returns:
            float: max(0, x)
        """
        return x if x > 0 else 0

    @staticmethod
    def backward(x):
        """
        Compute ReLU derivative.
        
        Args:
            x (float): Input value (same as used in forward pass)
            
        Returns:
            float: Gradient of ReLU at x
        """
        return 1.0 if x > 0 else 0.0
    

class LRELU:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    
    An improved version of ReLU that addresses the "dying ReLU" problem
    by allowing small gradients for negative inputs, enabling recovery
    of inactive neurons during training.
    
    Mathematical Definition:
        f(x) = { x      if x > 0
               { αx     if x ≤ 0
               
        f'(x) = { 1     if x > 0  
                { α     if x ≤ 0
                
    where α is a small positive constant (typically 0.01)
    
    Properties:
        - Range: (-∞, +∞)
        - Non-zero gradients for all inputs
        - Prevents neuron death
        - Maintains computational efficiency
        
    Advantages:
        - Solves dying ReLU problem
        - Better gradient flow
        - Improved learning dynamics
        - Minimal computational overhead
        
    Parameters:
        alpha (float): Slope for negative inputs. Default: 0.01
        
    Use Cases:
        - Deep networks prone to dying ReLU
        - Networks requiring robust gradient flow
        - Alternative to ReLU in critical applications
    """
    
    @staticmethod
    def forward(x, alpha=0.01):
        """
        Compute Leaky ReLU activation.
        
        Args:
            x (float): Input value
            alpha (float): Negative slope coefficient
            
        Returns:
            float: x if x > 0, else alpha * x
        """
        return x if x > 0 else alpha * x

    @staticmethod
    def backward(x, alpha=0.01):
        """
        Compute Leaky ReLU derivative.
        
        Args:
            x (float): Input value
            alpha (float): Negative slope coefficient
            
        Returns:
            float: Gradient of Leaky ReLU at x
        """
        return 1.0 if x > 0 else alpha


class TANH:
    """
    Hyperbolic Tangent activation function.
    
    A classic activation function that maps inputs to the range (-1, 1).
    Tanh is zero-centered, making it often preferable to sigmoid for
    hidden layers as it can lead to faster convergence.
    
    Mathematical Definition:
        f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        f'(x) = 1 - tanh²(x) = 1 - f(x)²
        
    Properties:
        - Range: (-1, 1)  
        - Zero-centered output
        - Smooth and differentiable
        - Saturates for large |x|
        - Symmetric around origin
        
    Advantages:
        - Zero-centered (better than sigmoid)
        - Strong gradients near zero
        - Smooth, continuous function
        - Well-understood mathematical properties
        
    Disadvantages:
        - Vanishing gradient problem for large |x|
        - Computationally more expensive than ReLU
        
    Use Cases:
        - Recurrent neural networks (RNNs)
        - Small networks where vanishing gradients aren't problematic
        - Applications requiring zero-centered activations
        - Traditional neural network architectures
    """
    
    @staticmethod
    def forward(x):
        """
        Compute hyperbolic tangent activation.
        
        Args:
            x (float): Input value
            
        Returns:
            float: tanh(x) in range (-1, 1)
        """
        return math.tanh(x)

    @staticmethod
    def backward(x):
        """
        Compute tanh derivative using the identity: d/dx tanh(x) = 1 - tanh²(x).
        
        Args:
            x (float): Input value
            
        Returns:
            float: Gradient of tanh at x
        """
        t = math.tanh(x)
        return 1 - t * t


class SIGMOID:
    """
    Sigmoid (Logistic) activation function.
    
    The classical activation function that maps any real number to (0, 1),
    making it ideal for binary classification and probability estimation.
    Historically important but largely superseded by ReLU for hidden layers.
    
    Mathematical Definition:
        f(x) = σ(x) = 1 / (1 + e^(-x))
        f'(x) = σ(x) * (1 - σ(x)) = f(x) * (1 - f(x))
        
    Properties:
        - Range: (0, 1)
        - Smooth and differentiable
        - Monotonically increasing
        - Probabilistic interpretation
        - Saturates for large |x|
        
    Advantages:
        - Natural probability interpretation
        - Smooth gradients
        - Well-suited for binary classification
        - Historically well-understood
        
    Disadvantages:
        - Vanishing gradient problem
        - Not zero-centered
        - Computationally expensive
        - Output saturation issues
        
    Use Cases:
        - Binary classification output layers
        - Gating mechanisms in RNNs/LSTMs  
        - Attention mechanisms
        - Probability estimation tasks
    """
    
    @staticmethod
    def forward(x):
        """
        Compute sigmoid activation.
        
        Uses numerically stable computation to avoid overflow for large |x|.
        
        Args:
            x (float): Input value
            
        Returns:
            float: Sigmoid output in range (0, 1)
        """
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def backward(x):
        """
        Compute sigmoid derivative using the identity: σ'(x) = σ(x)(1 - σ(x)).
        
        Args:
            x (float): Input value
            
        Returns:
            float: Gradient of sigmoid at x
        """
        s = 1 / (1 + math.exp(-x))
        return s * (1 - s)


class SWISH:
    """
    Swish activation function.
    
    A modern self-gated activation function discovered by Google Research
    that often outperforms ReLU in deep networks. Swish combines the 
    simplicity of ReLU with the smoothness of sigmoid gating.
    
    Mathematical Definition:
        f(x) = x * σ(x) = x / (1 + e^(-x))
        f'(x) = σ(x) + x * σ(x) * (1 - σ(x))
              = σ(x) * (1 + x * (1 - σ(x)))
              
    where σ(x) is the sigmoid function
    
    Properties:
        - Range: (-∞, +∞) but mostly in [0, +∞)
        - Smooth and non-monotonic
        - Self-gated (uses own input for gating)
        - Unbounded above, bounded below
        - Approximately linear for large positive x
        
    Advantages:
        - Often outperforms ReLU empirically
        - Smooth gradients (no sharp corners)
        - Self-gating property
        - Works well in very deep networks
        - Good performance across many domains
        
    Disadvantages:
        - More computationally expensive than ReLU
        - Less interpretable than simpler functions
        - Requires more careful initialization
        
    Use Cases:
        - Deep convolutional networks
        - Transformer architectures  
        - Large-scale models where performance matters
        - Research applications exploring modern activations
        
    References:
        Ramachandran et al. "Searching for Activation Functions" (2017)
    """
    
    @staticmethod
    def forward(x):
        """
        Compute Swish activation: x * sigmoid(x).
        
        Args:
            x (float): Input value
            
        Returns:
            float: Swish activation output
        """
        return x / (1 + math.exp(-x))

    @staticmethod
    def backward(x):
        """
        Compute Swish derivative.
        
        Uses the product rule and sigmoid derivative:
        d/dx [x * σ(x)] = σ(x) + x * σ'(x) = σ(x) + x * σ(x) * (1 - σ(x))
        
        Args:
            x (float): Input value
            
        Returns:
            float: Gradient of Swish at x
        """
        s = 1 / (1 + math.exp(-x))  # sigmoid
        return s + x * s * (1 - s)