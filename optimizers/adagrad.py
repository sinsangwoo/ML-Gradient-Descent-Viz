"""
AdaGrad - Adaptive Gradient Algorithm

AdaGrad adapts the learning rate for each parameter based on historical gradients.
Parameters with large gradients get smaller effective learning rates.

Update rule:
    G_t = G_{t-1} + g_t ⊙ g_t  (accumulate squared gradients)
    θ_{t+1} = θ_t - η / √(G_t + ε) ⊙ g_t

where ⊙ is element-wise multiplication.

Advantages:
- Automatic learning rate adaptation
- Works well with sparse gradients
- No manual tuning of per-parameter learning rates

Disadvantages:
- Monotonically decreasing learning rate (can stop too early)
- Not suitable for non-convex optimization (deep learning)

References:
- Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
"""

import numpy as np
from typing import Dict
from .base_optimizer import BaseOptimizer


class AdaGrad(BaseOptimizer):
    """
    AdaGrad optimizer with per-parameter adaptive learning rates.
    
    The effective learning rate for parameter i at time t is:
        η_i(t) = η / √(Σ_{s=1}^t g_i(s)² + ε)
    
    Key property: Learning rate decreases proportionally to gradient magnitude.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-8,
                 initial_accumulator_value: float = 0.0):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate η
        epsilon : float
            Small constant for numerical stability
            Prevents division by zero
        initial_accumulator_value : float
            Initial value for gradient accumulator
            Can be set > 0 to prevent very large initial steps
        """
        super().__init__(learning_rate=learning_rate, name="AdaGrad")
        
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be > 0, got {epsilon}")
        
        self.epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self.accumulator = None  # G_t: accumulated squared gradients
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform AdaGrad update.
        
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
        # Initialize accumulator on first step
        if self.accumulator is None:
            self.accumulator = np.full_like(
                params, 
                self.initial_accumulator_value
            )
        
        # Accumulate squared gradients: G_t = G_{t-1} + g_t²
        self.accumulator += gradient ** 2
        
        # Compute adaptive learning rate per parameter
        # η_adapted = η / √(G_t + ε)
        adapted_lr = self.learning_rate / (np.sqrt(self.accumulator) + self.epsilon)
        
        # Update parameters: θ = θ - η_adapted ⊙ g
        new_params = params - adapted_lr * gradient
        
        return new_params
    
    def get_effective_lr(self) -> float:
        """
        Get mean effective learning rate across all parameters.
        
        Returns:
        --------
        mean_lr : float
            Average effective learning rate
        """
        if self.accumulator is None:
            return self.learning_rate
        
        # Compute mean of per-parameter learning rates
        adapted_lrs = self.learning_rate / (np.sqrt(self.accumulator) + self.epsilon)
        return np.mean(adapted_lrs)
    
    def reset(self):
        """Reset optimizer state including accumulator."""
        super().reset()
        self.accumulator = None
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'initial_accumulator_value': self.initial_accumulator_value
        }