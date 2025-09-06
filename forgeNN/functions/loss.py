"""
forgeNN Loss Functions Module
=============================

Production-ready implementations of standard loss functions for supervised
learning with automatic differentiation support. Optimized for both scalar
and tensor operations with numerical stability considerations.

This module provides the mathematical foundations for training neural networks
by measuring the difference between predictions and ground truth labels.

Classes:
    MSE: Mean Squared Error - standard regression loss
    CrossEntropy: Cross-entropy loss - standard classification loss

Design Principles:
    - Numerical stability for extreme values
    - Efficient computation and memory usage  
    - Support for both scalar and batch operations
    - Proper gradient computation for backpropagation
    - Industry-standard mathematical formulations

Mathematical Foundations:
    Loss functions L(ŷ, y) measure prediction quality:
    - Forward pass: loss_value = L(ŷ, y)  
    - Backward pass: ∂L/∂ŷ for gradient descent
"""

import numpy as np

class MSE:
    """
    Mean Squared Error loss function.
    
    The most fundamental loss function for regression tasks, measuring
    the average squared difference between predictions and targets.
    MSE provides a convex optimization landscape and is differentiable
    everywhere, making it ideal for gradient-based optimization.
    
    Mathematical Definition:
        MSE(ŷ, y) = (1/n) * Σᵢ(ŷᵢ - yᵢ)²  [batch version]
        MSE(ŷ, y) = (ŷ - y)²               [scalar version]
        
        ∂MSE/∂ŷ = (2/n) * Σᵢ(ŷᵢ - yᵢ)     [batch version] 
        ∂MSE/∂ŷ = 2(ŷ - y)                [scalar version]
    
    Properties:
        - Always non-negative: MSE ≥ 0
        - Convex function (single global minimum)
        - Differentiable everywhere
        - Scale-dependent (sensitive to output magnitude)
        - Penalizes large errors quadratically
        
    Advantages:
        - Simple and intuitive interpretation
        - Convex optimization landscape
        - Smooth gradients for optimization
        - Well-understood mathematical properties
        - Efficient computation
        
    Disadvantages:
        - Sensitive to outliers (quadratic penalty)
        - Scale-dependent (not unit-invariant)
        - May not reflect true task objectives
        
    Use Cases:
        - Regression problems
        - Continuous value prediction
        - Neural network training
        - Baseline loss for comparison
        - Problems where large errors should be heavily penalized
    """
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Compute Mean Squared Error loss.
        
        Handles both scalar and array inputs with automatic broadcasting.
        For arrays, computes the mean over all elements.
        
        Args:
            y_pred (float or array-like): Predicted values
            y_true (float or array-like): Ground truth target values
            
        Returns:
            float: MSE loss value
            
        Example:
            >>> # Scalar case
            >>> loss = MSE.forward(2.5, 3.0)  # Returns: 0.25
            >>> 
            >>> # Array case  
            >>> predictions = np.array([1.0, 2.0, 3.0])
            >>> targets = np.array([1.1, 2.2, 2.8])
            >>> loss = MSE.forward(predictions, targets)
        """
        # Handle both scalar and array inputs
        if isinstance(y_pred, (int, float)) and isinstance(y_true, (int, float)):
            return (y_pred - y_true) ** 2
        else:
            return ((y_pred - y_true) ** 2).mean()

    @staticmethod
    def backward(y_pred, y_true):
        """
        Compute gradient of MSE with respect to predictions.
        
        The gradient provides the direction for parameter updates
        during backpropagation and gradient descent optimization.
        
        Args:
            y_pred (float or array-like): Predicted values
            y_true (float or array-like): Ground truth target values
            
        Returns:
            float or array-like: Gradient ∂MSE/∂y_pred
            
        Mathematical Derivation:
            ∂/∂ŷ [(ŷ - y)²] = 2(ŷ - y)
            
        Example:
            >>> gradient = MSE.backward(2.5, 3.0)  # Returns: -1.0
            >>> # Gradient points toward target (negative for overprediction)
        """
        # Handle both scalar and array inputs
        if isinstance(y_pred, (int, float)) and isinstance(y_true, (int, float)):
            return 2 * (y_pred - y_true)
        else:
            return 2 * (y_pred - y_true) / y_true.size

class CrossEntropy:
    """
    Cross-Entropy loss function for classification tasks.
    
    The standard loss function for multi-class classification problems.
    Cross-entropy measures the difference between predicted and true
    probability distributions, providing strong gradients for confident
    wrong predictions and encouraging calibrated probability outputs.
    
    Mathematical Definition:
        For multi-class classification:
        CE(ŷ, y) = -(1/m) * Σᵢ log(ŷ[i, y[i]])
        
        For binary classification:  
        CE(ŷ, y) = -(y*log(ŷ) + (1-y)*log(1-ŷ))
        
        Gradient:
        ∂CE/∂ŷ = -(1/m) * (y - ŷ)  [after softmax]
        
    Properties:
        - Always non-negative: CE ≥ 0
        - Heavily penalizes confident wrong predictions
        - Encourages well-calibrated probabilities
        - Optimal for maximum likelihood estimation
        - Requires probability-normalized inputs
        
    Advantages:
        - Probabilistic interpretation
        - Strong gradients for wrong predictions
        - Optimal for classification under log-likelihood
        - Well-suited for softmax outputs
        - Extensive theoretical foundation
        
    Disadvantages:
        - Requires probability normalization
        - Sensitive to class imbalance
        - Can be unstable without numerical safeguards
        - Not robust to label noise
        
    Use Cases:
        - Multi-class classification
        - Binary classification  
        - Probability estimation
        - Neural network training with softmax
        - Maximum likelihood estimation
        
    Note:
        Input predictions should be probabilities (0 ≤ ŷ ≤ 1).
        For raw logits, apply softmax first to get valid probabilities.
    """
    
    @staticmethod
    def forward(y_pred, y_true):
        """
        Compute Cross-Entropy loss.
        
        Implements numerically stable cross-entropy computation with
        clipping to prevent log(0) which would cause numerical instability.
        
        Args:
            y_pred (array-like): Predicted probabilities (post-softmax)
                Shape: (batch_size, num_classes) or (batch_size,)
            y_true (array-like): True class labels
                Either class indices or one-hot encoded
                
        Returns:
            float: Cross-entropy loss value
            
        Example:
            >>> # Multi-class case
            >>> predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
            >>> labels = np.array([0, 1])  # Class indices
            >>> loss = CrossEntropy.forward(predictions, labels)
            
        Note:
            Predictions are clipped to [1e-15, 1-1e-15] to prevent
            numerical instability from log(0) or log(1).
        """
        m = y_true.shape[0]
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_likelihood = -np.log(p[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def backward(y_pred, y_true):
        """
        Compute gradient of Cross-Entropy loss.
        
        For softmax + cross-entropy combination, the gradient simplifies
        to a remarkably clean form: (predictions - targets) / batch_size.
        
        Args:
            y_pred (array-like): Predicted probabilities
            y_true (array-like): True class labels (as indices)
            
        Returns:
            array-like: Gradient ∂CE/∂y_pred
            
        Mathematical Insight:
            The combination of softmax activation and cross-entropy loss
            produces gradients proportional to prediction errors, which
            is optimal for gradient-based learning.
            
        Example:
            >>> gradient = CrossEntropy.backward(predictions, labels)
            >>> # Use gradient for parameter updates in backpropagation
        """
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad
    
