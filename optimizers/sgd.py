"""
Stochastic Gradient Descent (SGD)

Implements vanilla SGD with optional mini-batch support.
This serves as the baseline optimizer for comparison.

Convergence Theory:
- For convex functions: O(1/k) convergence
- For strongly convex: Linear convergence with optimal learning rate

References:
- Robbins, H., & Monro, S. (1951). A stochastic approximation method.
- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent.
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Vanilla Stochastic Gradient Descent.
    
    Update rule:
    θ_{t+1} = θ_t - α ∇J(θ_t)
    
    Properties:
    - Memoryless (no momentum)
    - Constant learning rate
    - Simplest first-order method
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000,
                 batch_size: int = None, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size α
        epochs : int
            Number of passes through data
        batch_size : int, optional
            Mini-batch size (None = full batch)
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.batch_size = batch_size
        
    def _initialize_state(self):
        """SGD has no additional state."""
        pass
    
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        Vanilla SGD update: Δθ = -α ∇J(θ)
        """
        update_W = -self.learning_rate * grad_W
        update_b = -self.learning_rate * grad_b
        
        # Track effective learning rate (constant for vanilla SGD)
        if self.track_history:
            self.learning_rate_history.append(self.learning_rate)
        
        return update_W, update_b
    
    def get_hyperparameters(self) -> Dict:
        """Return SGD hyperparameters."""
        return {
            'optimizer': 'SGD',
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size if self.batch_size else 'full'
        }


if __name__ == "__main__":
    # Test SGD
    print("Testing SGD optimizer\n")
    
    from data_generator import LinearDataGenerator
    
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    optimizer = SGD(
        learning_rate=0.1,
        epochs=500,
        random_seed=42,
        monitor_convergence=True
    )
    
    optimizer.fit(X, y, verbose=True)
    
    print("\nTest predictions:")
    X_test = np.array([[0.5], [1.0], [1.5]])
    y_pred = optimizer.predict(X_test)
    for x, y_p in zip(X_test, y_pred):
        print(f"X={x[0]:.1f} → ŷ={y_p[0]:.3f}")
