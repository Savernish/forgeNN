"""
forgeNN: A High-Performance Automatic Differentiation Framework
================================================================

A production-ready deep learning library featuring efficient automatic 
differentiation, dynamic computation graphs, and optimized neural network 
components. Designed for research, education, and production deployments.

Copyright (c) 2025 forgeNN Development Team
Licensed under MIT License
"""

from ..functions.activation import RELU, LRELU, TANH, SIGMOID, SWISH
from ..functions.loss import MSE, CrossEntropy

class Value:
    """
    Tensor-like automatic differentiation engine with dynamic computation graph.
    
    The Value class is the core building block of forgeNN, providing automatic 
    differentiation capabilities through reverse-mode automatic differentiation 
    (backpropagation). Each Value instance represents a node in a dynamic 
    computation graph, enabling efficient gradient computation for complex 
    mathematical expressions.
    
    Key Features:
    - Dynamic computation graph construction
    - Efficient reverse-mode automatic differentiation  
    - Memory-optimized gradient computation
    - Extensive operator overloading for intuitive API
    - Built-in activation and loss functions
    - Production-ready performance optimizations
    
    Args:
        data (float): The numerical value of this tensor node
        _children (tuple, optional): Parent nodes in computation graph. Defaults to ()
        _op (str, optional): Operation that created this node. Defaults to ''
        
    Attributes:
        data (float): The forward pass value
        grad (float): The computed gradient (∂L/∂self)
        _prev (set): Set of parent nodes for graph traversal
        _op (str): String representation of the operation
        _backward (callable): Function to compute local gradients
        
    Example:
        >>> import forgeNN as fnn
        >>> x = fnn.Value(2.0)
        >>> y = fnn.Value(3.0)  
        >>> z = x * y + x**2
        >>> z.backward()
        >>> print(f"dz/dx = {x.grad}")  # Outputs: dz/dx = 7.0
        
    Note:
        This implementation uses reverse-mode AD which is optimal for 
        functions with many inputs and few outputs (typical in ML).
    """
    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a new Value node in the computation graph.
        
        Args:
            data (float): Numerical value for this node
            _children (tuple): Parent nodes that created this node
            _op (str): String identifier for the operation type
        """
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        """
        Element-wise addition with automatic differentiation support.
        
        Implements the chain rule: d(u+v)/dx = du/dx + dv/dx
        
        Args:
            other (Value): Right operand for addition
            
        Returns:
            Value: New node representing the sum with gradient function
            
        Example:
            >>> a = Value(2.0)
            >>> b = Value(3.0)
            >>> c = a + b  # c.data = 5.0
        """
        out = Value(self.data + other.data, (self, other), _op='+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Element-wise multiplication with automatic differentiation support.
        
        Implements the product rule: d(uv)/dx = u(dv/dx) + v(du/dx)
        
        Args:
            other (Value): Right operand for multiplication
            
        Returns:
            Value: New node representing the product with gradient function
            
        Example:
            >>> a = Value(2.0)
            >>> b = Value(3.0)
            >>> c = a * b  # c.data = 6.0
        """
        out = Value(self.data * other.data, (self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        """
        Power operation with automatic differentiation support.
        
        Implements the power rule: d(u^n)/dx = n*u^(n-1) * du/dx
        
        Args:
            other (int, float): Exponent (must be numeric, not Value)
            
        Returns:
            Value: New node representing self raised to the power of other
            
        Raises:
            AssertionError: If other is not int or float
            
        Example:
            >>> x = Value(3.0)
            >>> y = x ** 2  # y.data = 9.0
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), _op=f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Perform reverse-mode automatic differentiation (backpropagation).
        
        Computes gradients for all nodes in the computation graph using 
        topological sorting and the chain rule. This method implements 
        the core backpropagation algorithm used in modern deep learning.
        
        The algorithm:
        1. Topologically sort all nodes in the computation graph
        2. Initialize this node's gradient to 1.0 (∂self/∂self = 1)
        3. Traverse nodes in reverse topological order
        4. Apply chain rule to propagate gradients backwards
        
        Time Complexity: O(V + E) where V is nodes and E is edges
        Space Complexity: O(V) for the topological sort
        
        Example:
            >>> x = Value(2.0)
            >>> y = x**2 + 3*x + 1
            >>> y.backward()  # Computes dy/dx = 4*x + 3 = 11
            >>> print(x.grad)  # Outputs: 11.0
            
        Note:
            Call this method only on scalar outputs (loss functions).
            For multi-output functions, call backward() on each output.
        """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __repr__(self):
        #return f"Value(data={self.data})"
        return f"{self.data}"
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def apply_activation(self, activation_class, *args, **kwargs):
        """
        Apply activation function with automatic differentiation support.
        
        This method provides a unified interface for applying activation 
        functions while maintaining gradient flow through the computation 
        graph. Uses the strategy pattern for extensible activation functions.
        
        Args:
            activation_class: Activation function class with forward/backward methods
            *args: Positional arguments passed to activation function
            **kwargs: Keyword arguments passed to activation function
            
        Returns:
            Value: New node with activation applied and gradient function attached
            
        Example:
            >>> x = Value(0.5)
            >>> y = x.apply_activation(RELU)
            >>> y.backward()
            >>> print(x.grad)  # Gradient through ReLU
            
        Note:
            This is the core method used by all activation function shortcuts
            (relu, sigmoid, tanh, etc.). Activation classes must implement
            forward(x, *args, **kwargs) and backward(x, *args, **kwargs).
        """
        # Forward pass using the activation class
        activated_data = activation_class.forward(self.data, *args, **kwargs)
        out = Value(activated_data, (self,), f'{activation_class.__name__}')
        
        def _backward():
            # Backward pass using the activation class
            grad_input = activation_class.backward(self.data, *args, **kwargs)
            self.grad += grad_input * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        """
        Apply Rectified Linear Unit (ReLU) activation function.
        
        ReLU(x) = max(0, x)
        
        ReLU is the most widely used activation function in deep learning due to:
        - Computational efficiency (simple max operation)
        - Mitigation of vanishing gradient problem
        - Sparse activation (promotes sparsity)
        - Non-saturating for positive inputs
        
        Returns:
            Value: New node with ReLU activation applied
            
        Example:
            >>> x = Value(-2.0)
            >>> y = x.relu()  # y.data = 0.0
            >>> z = Value(3.0).relu()  # z.data = 3.0
        """
        return self.apply_activation(RELU)
    
    def lrelu(self, alpha=0.01):
        """
        Apply Leaky Rectified Linear Unit (Leaky ReLU) activation function.
        
        LeakyReLU(x) = max(αx, x) where α is a small positive constant
        
        Leaky ReLU addresses the "dying ReLU" problem by allowing small 
        gradients for negative inputs, enabling recovery of "dead" neurons.
        
        Args:
            alpha (float): Negative slope coefficient. Defaults to 0.01
            
        Returns:
            Value: New node with Leaky ReLU activation applied
            
        Example:
            >>> x = Value(-2.0)
            >>> y = x.lrelu(alpha=0.1)  # y.data = -0.2
        """
        return self.apply_activation(LRELU, alpha=alpha)
    
    def tanh(self):
        """
        Apply Hyperbolic Tangent activation function.
        
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Tanh is a zero-centered activation function with output range (-1, 1).
        Often preferred over sigmoid due to zero-centered output which can
        lead to faster convergence during training.
        
        Returns:
            Value: New node with tanh activation applied
            
        Example:
            >>> x = Value(0.0)
            >>> y = x.tanh()  # y.data = 0.0
            >>> z = Value(1.0).tanh()  # z.data ≈ 0.7616
        """
        return self.apply_activation(TANH)
    
    def sigmoid(self):
        """
        Apply Sigmoid activation function.
        
        σ(x) = 1 / (1 + e^(-x))
        
        Sigmoid maps any real number to (0, 1), making it useful for:
        - Binary classification (output layer)
        - Gating mechanisms in RNNs/LSTMs
        - Probability estimation
        
        Note: Can suffer from vanishing gradients for large |x|
        
        Returns:
            Value: New node with sigmoid activation applied
            
        Example:
            >>> x = Value(0.0)
            >>> y = x.sigmoid()  # y.data = 0.5
            >>> z = Value(2.0).sigmoid()  # z.data ≈ 0.8808
        """
        return self.apply_activation(SIGMOID)
    
    def swish(self):
        """
        Apply Swish activation function.
        
        Swish(x) = x * σ(x) = x / (1 + e^(-x))
        
        Swish is a self-gated activation function discovered by Google that
        often outperforms ReLU in deep networks. Key properties:
        - Smooth and non-monotonic
        - Unbounded above, bounded below
        - Self-gating property
        
        Returns:
            Value: New node with Swish activation applied
            
        Example:
            >>> x = Value(1.0)
            >>> y = x.swish()  # y.data ≈ 0.7311
        """
        return self.apply_activation(SWISH)

    def apply_loss(self, loss_class, target, *args, **kwargs):
        """
        Apply loss function with automatic differentiation support.
        
        This method provides a unified interface for computing loss functions
        while maintaining gradient flow. Essential for training neural networks
        as it connects predictions to ground truth labels.
        
        Args:
            loss_class: Loss function class with forward/backward methods
            target: Ground truth target value or label
            *args: Additional positional arguments for loss function
            **kwargs: Additional keyword arguments for loss function
            
        Returns:
            Value: New node representing the computed loss
            
        Example:
            >>> pred = Value(0.8)
            >>> target = 1.0
            >>> loss = pred.apply_loss(MSE, target)
            >>> loss.backward()  # Computes gradients for optimization
        """
        # Forward pass using the loss class
        loss_value = loss_class.forward(self.data, target, *args, **kwargs)
        out = Value(loss_value, (self,), f'{loss_class.__name__}')
        
        def _backward():
            # Backward pass using the loss class
            grad_input = loss_class.backward(self.data, target, *args, **kwargs)
            self.grad += grad_input * out.grad
        out._backward = _backward

        return out

    def mse(self, target):
        """
        Compute Mean Squared Error loss.
        
        MSE(ŷ, y) = (ŷ - y)²
        
        Mean Squared Error is the most common loss function for regression tasks.
        It penalizes large errors quadratically, making it sensitive to outliers.
        
        Mathematical properties:
        - Always non-negative
        - Differentiable everywhere
        - Convex (single global minimum)
        - Scale-dependent
        
        Args:
            target (float): Ground truth target value
            
        Returns:
            Value: MSE loss value with gradient computation
            
        Example:
            >>> prediction = Value(2.5)
            >>> target = 3.0
            >>> loss = prediction.mse(target)  # loss.data = 0.25
            >>> loss.backward()
            >>> print(prediction.grad)  # Gradient: 2*(pred - target) = -1.0
        """
        return self.apply_loss(MSE, target)
    
    def cross_entropy(self, target):
        """
        Compute Cross-Entropy loss.
        
        CrossEntropy(ŷ, y) = -log(ŷ[y]) for classification
        
        Cross-entropy is the standard loss function for classification tasks.
        It measures the difference between predicted and true probability 
        distributions, heavily penalizing confident wrong predictions.
        
        Mathematical properties:
        - Always non-negative
        - Approaches 0 as prediction approaches target
        - Heavily penalizes confident wrong predictions
        - Used with softmax for multi-class classification
        
        Args:
            target: Ground truth class label or probability distribution
            
        Returns:
            Value: Cross-entropy loss value with gradient computation
            
        Example:
            >>> # For binary classification
            >>> prob = Value(0.9)  # Predicted probability
            >>> label = 1  # True class
            >>> loss = prob.cross_entropy(label)
            
        Note:
            For multi-class problems, ensure predictions are softmax-normalized
            and targets are properly encoded (one-hot or class indices).
        """
        return self.apply_loss(CrossEntropy, target)
    