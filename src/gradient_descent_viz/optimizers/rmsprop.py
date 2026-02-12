"""
RMSProp - Root Mean Square Propagation

RMSProp addresses AdaGrad's monotonically decreasing learning rate by using
an exponentially decaying average of squared gradients instead of cumulative sum.

Update rule:
    E[g²]_t = ρ E[g²]_{t-1} + (1-ρ) g_t²  (exponential moving average)
    θ_{t+1} = θ_t - η / √(E[g²]_t + ε) ⊙ g_t

where:
    ρ: decay rate (typically 0.9 or 0.99)
    E[g²]_t: moving average of squared gradients

Advantages over AdaGrad:
- Non-monotonic learning rate (can increase again)
- Better for non-convex optimization
- More suitable for deep learning

References:
- Tieleman & Hinton (2012). Lecture 6.5 - RMSProp, COURSERA: Neural Networks for Machine Learning
- Hinton, G. (2012). Overview of mini-batch gradient descent
"""

import numpy as np
from typing import Dict
from .base_optimizer import BaseOptimizer


class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer with exponentially decaying squared gradient average.
    
    The effective learning rate adapts based on recent gradient magnitudes,
    allowing learning rate to both increase and decrease over time.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 epsilon: float = 1e-8,
                 centered: bool = False):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate η (often 0.001 for RMSProp)
        rho : float
            Decay rate ρ ∈ [0, 1) for moving average
            - 0.9: typical value (recent history matters more)
            - 0.99: longer memory (smoother adaptation)
        epsilon : float
            Small constant for numerical stability
        centered : bool
            If True, normalize by centered second moment:
            E[g²] - E[g]² instead of E[g²]
            Can improve convergence in some cases
        """
        super().__init__(learning_rate=learning_rate, name="RMSProp")
        
        if not 0 <= rho < 1:
            raise ValueError(f"Rho must be in [0, 1), got {rho}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be > 0, got {epsilon}")
        
        self.rho = rho
        self.epsilon = epsilon
        self.centered = centered
        
        self.moving_avg_squared = None  # E[g²]
        self.moving_avg = None  # E[g] (only if centered=True)
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform RMSProp update.
        
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
        # Initialize moving averages on first step
        if self.moving_avg_squared is None:
            self.moving_avg_squared = np.zeros_like(params)
            if self.centered:
                self.moving_avg = np.zeros_like(params)
        
        # Update moving average of squared gradients
        # E[g²]_t = ρ E[g²]_{t-1} + (1-ρ) g_t²
        self.moving_avg_squared = (
            self.rho * self.moving_avg_squared + 
            (1 - self.rho) * gradient ** 2
        )
        
        if self.centered:
            # Update moving average of gradients: E[g]_t = ρ E[g]_{t-1} + (1-ρ) g_t
            self.moving_avg = (
                self.rho * self.moving_avg + 
                (1 - self.rho) * gradient
            )
            # Use centered second moment: Var[g] = E[g²] - E[g]²
            denominator = self.moving_avg_squared - self.moving_avg ** 2 + self.epsilon
        else:
            # Standard RMSProp: just E[g²]
            denominator = self.moving_avg_squared + self.epsilon
        
        # Compute adaptive learning rate
        adapted_lr = self.learning_rate / np.sqrt(denominator)
        
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
        if self.moving_avg_squared is None:
            return self.learning_rate
        
        if self.centered and self.moving_avg is not None:
            denominator = self.moving_avg_squared - self.moving_avg ** 2 + self.epsilon
        else:
            denominator = self.moving_avg_squared + self.epsilon
        
        adapted_lrs = self.learning_rate / np.sqrt(denominator)
        return np.mean(adapted_lrs)
    
    def reset(self):
        """Reset optimizer state including moving averages."""
        super().reset()
        self.moving_avg_squared = None
        self.moving_avg = None
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'rho': self.rho,
            'epsilon': self.epsilon,
            'centered': self.centered
        }