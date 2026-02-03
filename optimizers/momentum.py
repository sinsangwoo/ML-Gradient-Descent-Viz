"""
Momentum-Based Optimizers

Implements:
1. Classical Momentum (Polyak, 1964)
2. Nesterov Accelerated Gradient (NAG)

Key Innovation: Accumulate velocity to accelerate convergence
and dampen oscillations.

Theoretical Guarantees:
- Classical Momentum: Better constants than SGD
- Nesterov: Optimal O(1/k²) for smooth convex functions

References:
- Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods.
- Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²).
- Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer


class MomentumSGD(BaseOptimizer):
    """
    SGD with Classical Momentum (Polyak, 1964).
    
    Update rule:
    v_{t+1} = β*v_t - α*∇J(θ_t)
    θ_{t+1} = θ_t + v_{t+1}
    
    Where:
    - v is the velocity (momentum term)
    - β ∈ [0,1) is the momentum coefficient
    - α is the learning rate
    
    Physical Interpretation:
    Think of a ball rolling down a hill:
    - Gradient: current slope
    - Velocity: accumulated "inertia"
    - Momentum helps traverse flat regions and escape local minima
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size α
        momentum : float
            Momentum coefficient β ∈ [0,1)
            - β=0: Vanilla SGD
            - β=0.9: Standard choice
            - β=0.99: Heavy momentum (for very flat regions)
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.momentum = momentum
        
        # Velocity state
        self.v_W = None
        self.v_b = None
        
    def _initialize_state(self):
        """Initialize velocity vectors to zero."""
        self.v_W = np.zeros_like(self.W)
        self.v_b = 0.0
        
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        Momentum update:
        v ← β*v - α*∇J
        Δθ = v
        """
        # Update velocities
        self.v_W = self.momentum * self.v_W - self.learning_rate * grad_W
        self.v_b = self.momentum * self.v_b - self.learning_rate * grad_b
        
        # Track effective learning rate
        if self.track_history:
            self.learning_rate_history.append(self.learning_rate)
        
        return self.v_W, self.v_b
    
    def get_hyperparameters(self) -> Dict:
        """Return Momentum SGD hyperparameters."""
        return {
            'optimizer': 'MomentumSGD',
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


class NesterovMomentum(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG).
    
    Update rule (standard form):
    v_{t+1} = β*v_t - α*∇J(θ_t + β*v_t)
    θ_{t+1} = θ_t + v_{t+1}
    
    Key Difference from Classical Momentum:
    - Classical: Compute gradient at current position
    - Nesterov: Compute gradient at "look-ahead" position
    
    This "look-ahead" gives NAG superior theoretical convergence:
    - Smooth convex: O(1/k²) vs O(1/k) for SGD
    - Strongly convex: Better constant in linear convergence
    
    Practical Implementation:
    We use the mathematically equivalent reformulation that doesn't
    require computing gradient at look-ahead position:
    
    θ̃_t = θ_t + β*v_t  (look-ahead position)
    v_{t+1} = β*v_t - α*∇J(θ̃_t)
    θ_{t+1} = θ_t + v_{t+1}
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size α
        momentum : float
            Momentum coefficient β ∈ [0,1)
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.momentum = momentum
        
        # Velocity state
        self.v_W = None
        self.v_b = None
        
    def _initialize_state(self):
        """Initialize velocity vectors to zero."""
        self.v_W = np.zeros_like(self.W)
        self.v_b = 0.0
        
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray,
                          n_samples: int) -> Tuple[np.ndarray, float]:
        """
        Override to compute gradient at look-ahead position.
        
        θ̃ = θ + β*v
        ∇J(θ̃) instead of ∇J(θ)
        """
        # Compute look-ahead parameters
        W_lookahead = self.W + self.momentum * self.v_W
        b_lookahead = self.b + self.momentum * self.v_b
        
        # Compute gradient at look-ahead position
        y_pred = X @ W_lookahead + b_lookahead
        error = y_pred - y
        
        grad_W = (2/n_samples) * X.T @ error
        grad_b = (2/n_samples) * np.sum(error)
        
        return grad_W, grad_b
    
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        Nesterov update:
        v ← β*v - α*∇J(θ + β*v)  [gradient already computed at look-ahead]
        Δθ = v
        """
        # Update velocities (gradient already at look-ahead position)
        self.v_W = self.momentum * self.v_W - self.learning_rate * grad_W
        self.v_b = self.momentum * self.v_b - self.learning_rate * grad_b
        
        # Track effective learning rate
        if self.track_history:
            self.learning_rate_history.append(self.learning_rate)
        
        return self.v_W, self.v_b
    
    def get_hyperparameters(self) -> Dict:
        """Return Nesterov hyperparameters."""
        return {
            'optimizer': 'NesterovMomentum',
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


if __name__ == "__main__":
    # Test momentum optimizers
    print("Testing Momentum-based Optimizers\n")
    
    from data_generator import LinearDataGenerator
    
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    print("="*70)
    print("1. Classical Momentum")
    print("="*70)
    momentum_opt = MomentumSGD(
        learning_rate=0.1,
        momentum=0.9,
        epochs=300,
        random_seed=42,
        monitor_convergence=False
    )
    momentum_opt.fit(X, y, verbose=True)
    
    print("\n" + "="*70)
    print("2. Nesterov Accelerated Gradient")
    print("="*70)
    nesterov_opt = NesterovMomentum(
        learning_rate=0.1,
        momentum=0.9,
        epochs=300,
        random_seed=42,
        monitor_convergence=False
    )
    nesterov_opt.fit(X, y, verbose=True)
    
    # Compare convergence
    print("\n" + "="*70)
    print("Convergence Comparison")
    print("="*70)
    print(f"Classical Momentum final loss: {momentum_opt.get_history()['loss_history'][-1]:.6f}")
    print(f"Nesterov final loss: {nesterov_opt.get_history()['loss_history'][-1]:.6f}")
    print(f"Nesterov speedup: {momentum_opt.get_history()['loss_history'][-1] / nesterov_opt.get_history()['loss_history'][-1]:.2f}x")