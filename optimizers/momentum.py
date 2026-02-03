"""
Momentum-based Optimizers

Implements:
1. Classical Momentum (Polyak, 1964)
2. Nesterov Accelerated Gradient (Nesterov, 1983)

Theory:
- Momentum accelerates convergence in relevant directions
- Dampens oscillations in high-curvature directions
- Nesterov momentum achieves O(1/k²) convergence for convex smooth functions

References:
- Polyak, B.T. (1964). Some methods of speeding up the convergence of iteration methods
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²)
- Sutskever et al. (2013). On the importance of initialization and momentum in deep learning
"""

import numpy as np
from typing import Dict
from .base_optimizer import BaseOptimizer


class Momentum(BaseOptimizer):
    """
    Classical Momentum optimizer (Heavy Ball method).
    
    Update rule:
        v_{t+1} = β v_t + ∇J(θ_t)
        θ_{t+1} = θ_t - η v_{t+1}
    
    where:
        v_t: velocity (momentum accumulator)
        β: momentum coefficient (typically 0.9)
        η: learning rate
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size
        momentum : float
            Momentum coefficient β ∈ [0, 1)
            - 0: equivalent to SGD
            - 0.9: typical value
            - 0.99: high momentum
        """
        super().__init__(learning_rate=learning_rate, name="Momentum")
        
        if not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform momentum update.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Current gradient
            
        Returns:
        --------
        new_params : ndarray
            Updated parameters
        """
        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Update velocity: v = β*v + ∇J
        self.velocity = self.momentum * self.velocity + gradient
        
        # Update parameters: θ = θ - η*v
        new_params = params - self.learning_rate * self.velocity
        
        return new_params
    
    def reset(self):
        """Reset optimizer state including velocity."""
        super().reset()
        self.velocity = None
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


class NesterovMomentum(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG).
    
    Update rule:
        v_{t+1} = β v_t + ∇J(θ_t - β v_t)  # Look-ahead gradient
        θ_{t+1} = θ_t - η v_{t+1}
    
    Key difference from classical momentum:
    - Gradient computed at look-ahead position (θ - β*v)
    - Provides better convergence rate: O(1/k²) vs O(1/k)
    
    Interpretation:
    - "First jump, then correct" instead of "correct, then jump"
    - More responsive to changes in gradient direction
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 momentum: float = 0.9):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size
        momentum : float
            Momentum coefficient β ∈ [0, 1)
        """
        super().__init__(learning_rate=learning_rate, name="NesterovMomentum")
        
        if not 0 <= momentum < 1:
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform Nesterov momentum update.
        
        Note: This expects gradient to be computed at the look-ahead point.
        For practical implementation, gradient should be:
            ∇J(params - momentum * velocity)
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Gradient at look-ahead point
            
        Returns:
        --------
        new_params : ndarray
            Updated parameters
        """
        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Update velocity: v = β*v + ∇J(θ - β*v)
        self.velocity = self.momentum * self.velocity + gradient
        
        # Update parameters: θ = θ - η*v
        new_params = params - self.learning_rate * self.velocity
        
        return new_params
    
    def get_lookahead_params(self, params: np.ndarray) -> np.ndarray:
        """
        Compute look-ahead parameters for gradient evaluation.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
            
        Returns:
        --------
        lookahead_params : ndarray
            θ - β*v (where gradient should be evaluated)
        """
        if self.velocity is None:
            return params
        return params - self.momentum * self.velocity
    
    def reset(self):
        """Reset optimizer state including velocity."""
        super().reset()
        self.velocity = None
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }