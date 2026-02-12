"""
Adaptive Learning Rate Optimizers

Implements:
1. AdaGrad - Adaptive Gradient Algorithm
2. RMSProp - Root Mean Square Propagation
3. Adam - Adaptive Moment Estimation
4. AdamW - Adam with Decoupled Weight Decay

Key Innovation: Per-parameter adaptive learning rates based on
gradient history.

Theoretical Contributions:
- AdaGrad: Optimal for sparse gradients
- RMSProp: Fixes AdaGrad's aggressive learning rate decay
- Adam: Combines momentum + adaptive learning rates
- AdamW: Proper weight decay (not L2 regularization)

References:
- Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning
- Tieleman & Hinton (2012). RMSProp (Lecture 6.5)
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer


class AdaGrad(BaseOptimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad).
    
    Update rule:
    G_t = G_{t-1} + (∇J_t)²  (accumulated squared gradients)
    θ_{t+1} = θ_t - α * ∇J_t / (√G_t + ε)
    
    Properties:
    - Adapts learning rate per parameter
    - Large gradients → small learning rate
    - Small gradients → large learning rate
    - Performs well on sparse data (NLP, recommender systems)
    
    Weakness:
    - Aggressive learning rate decay (G_t only grows)
    - Can stop learning too early
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8,
                 epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate α
        epsilon : float
            Numerical stability constant ε (prevents division by zero)
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.epsilon = epsilon
        
        # Accumulated squared gradients
        self.G_W = None
        self.G_b = None
        
    def _initialize_state(self):
        """Initialize accumulated gradient squares to zero."""
        self.G_W = np.zeros_like(self.W)
        self.G_b = 0.0
        
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        AdaGrad update:
        G ← G + (∇J)²
        Δθ = -α * ∇J / (√G + ε)
        """
        # Accumulate squared gradients
        self.G_W += grad_W ** 2
        self.G_b += grad_b ** 2
        
        # Compute adaptive learning rates
        adapted_lr_W = self.learning_rate / (np.sqrt(self.G_W) + self.epsilon)
        adapted_lr_b = self.learning_rate / (np.sqrt(self.G_b) + self.epsilon)
        
        # Compute updates
        update_W = -adapted_lr_W * grad_W
        update_b = -adapted_lr_b * grad_b
        
        # Track effective learning rate (average)
        if self.track_history:
            avg_lr = (adapted_lr_W.mean() + adapted_lr_b) / 2
            self.learning_rate_history.append(avg_lr)
        
        return update_W, update_b
    
    def get_hyperparameters(self) -> Dict:
        """Return AdaGrad hyperparameters."""
        return {
            'optimizer': 'AdaGrad',
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon
        }


class RMSProp(BaseOptimizer):
    """
    Root Mean Square Propagation (RMSProp).
    
    Update rule:
    v_t = β*v_{t-1} + (1-β)*(∇J_t)²  (exponential moving average)
    θ_{t+1} = θ_t - α * ∇J_t / (√v_t + ε)
    
    Key Improvement over AdaGrad:
    - Uses exponential moving average instead of sum
    - Forgets old gradients (decay factor β)
    - Doesn't suffer from aggressive learning rate decay
    
    Widely used in RNNs before Adam.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8, epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate α
        beta : float
            Decay rate for moving average β ∈ [0,1)
            - β=0.9: Standard choice
            - β=0.99: Longer memory
        epsilon : float
            Numerical stability constant ε
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.beta = beta
        self.epsilon = epsilon
        
        # Exponential moving average of squared gradients
        self.v_W = None
        self.v_b = None
        
    def _initialize_state(self):
        """Initialize moving average to zero."""
        self.v_W = np.zeros_like(self.W)
        self.v_b = 0.0
        
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        RMSProp update:
        v ← β*v + (1-β)*(∇J)²
        Δθ = -α * ∇J / (√v + ε)
        """
        # Update moving average of squared gradients
        self.v_W = self.beta * self.v_W + (1 - self.beta) * (grad_W ** 2)
        self.v_b = self.beta * self.v_b + (1 - self.beta) * (grad_b ** 2)
        
        # Compute adaptive learning rates
        adapted_lr_W = self.learning_rate / (np.sqrt(self.v_W) + self.epsilon)
        adapted_lr_b = self.learning_rate / (np.sqrt(self.v_b) + self.epsilon)
        
        # Compute updates
        update_W = -adapted_lr_W * grad_W
        update_b = -adapted_lr_b * grad_b
        
        # Track effective learning rate
        if self.track_history:
            avg_lr = (adapted_lr_W.mean() + adapted_lr_b) / 2
            self.learning_rate_history.append(avg_lr)
        
        return update_W, update_b
    
    def get_hyperparameters(self) -> Dict:
        """Return RMSProp hyperparameters."""
        return {
            'optimizer': 'RMSProp',
            'learning_rate': self.learning_rate,
            'beta': self.beta,
            'epsilon': self.epsilon
        }


class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation (Adam).
    
    Update rule:
    m_t = β₁*m_{t-1} + (1-β₁)*∇J_t       (1st moment, momentum)
    v_t = β₂*v_{t-1} + (1-β₂)*(∇J_t)²    (2nd moment, RMSProp)
    
    Bias correction:
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    
    Update:
    θ_{t+1} = θ_t - α * m̂_t / (√v̂_t + ε)
    
    Combines Best of Both Worlds:
    - Momentum (m_t): Smooths gradient direction
    - Adaptive learning rate (v_t): Per-parameter scaling
    - Bias correction: Accounts for zero initialization
    
    Most popular optimizer in deep learning (as of 2020s).
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size α
        beta1 : float
            1st moment decay rate β₁ ∈ [0,1)
            - β₁=0.9: Standard choice
        beta2 : float
            2nd moment decay rate β₂ ∈ [0,1)
            - β₂=0.999: Standard choice (slower than RMSProp's 0.9)
        epsilon : float
            Numerical stability constant ε
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, epochs=epochs, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # 1st and 2nd moment estimates
        self.m_W = None
        self.m_b = None
        self.v_W = None
        self.v_b = None
        
    def _initialize_state(self):
        """Initialize moment estimates to zero."""
        self.m_W = np.zeros_like(self.W)
        self.m_b = 0.0
        self.v_W = np.zeros_like(self.W)
        self.v_b = 0.0
        
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        Adam update:
        m ← β₁*m + (1-β₁)*∇J
        v ← β₂*v + (1-β₂)*(∇J)²
        m̂ ← m / (1 - β₁^t)
        v̂ ← v / (1 - β₂^t)
        Δθ = -α * m̂ / (√v̂ + ε)
        """
        t = step + 1  # Adam uses 1-based indexing for bias correction
        
        # Update biased 1st moment estimate
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_W
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        
        # Update biased 2nd moment estimate
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_W ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
        
        # Bias correction
        m_hat_W = self.m_W / (1 - self.beta1 ** t)
        m_hat_b = self.m_b / (1 - self.beta1 ** t)
        v_hat_W = self.v_W / (1 - self.beta2 ** t)
        v_hat_b = self.v_b / (1 - self.beta2 ** t)
        
        # Compute updates
        update_W = -self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        update_b = -self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        # Track effective learning rate
        if self.track_history:
            # Effective LR = α / √v̂
            eff_lr_W = self.learning_rate / (np.sqrt(v_hat_W) + self.epsilon)
            eff_lr_b = self.learning_rate / (np.sqrt(v_hat_b) + self.epsilon)
            avg_lr = (eff_lr_W.mean() + eff_lr_b) / 2
            self.learning_rate_history.append(avg_lr)
        
        return update_W, update_b
    
    def get_hyperparameters(self) -> Dict:
        """Return Adam hyperparameters."""
        return {
            'optimizer': 'Adam',
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }


class AdamW(Adam):
    """
    Adam with Decoupled Weight Decay (AdamW).
    
    Key Insight (Loshchilov & Hutter, 2019):
    L2 regularization and weight decay are NOT equivalent in Adam!
    
    Wrong (L2 regularization in Adam):
    ∇J_reg = ∇J + λ*θ
    Then apply Adam update
    
    Right (Decoupled weight decay):
    Apply Adam update, then:
    θ_{t+1} = θ_{t+1} - η*λ*θ_t
    
    This small change significantly improves generalization!
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01, epochs: int = 1000, **kwargs):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size α
        beta1, beta2, epsilon : float
            Same as Adam
        weight_decay : float
            Weight decay coefficient λ
            - λ=0: No weight decay (becomes Adam)
            - λ=0.01: Standard choice
        epochs : int
            Number of training iterations
        """
        super().__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                        epsilon=epsilon, epochs=epochs, **kwargs)
        self.weight_decay = weight_decay
        
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        AdamW update:
        1. Compute Adam update
        2. Add decoupled weight decay
        """
        # Get Adam update
        update_W, update_b = super()._compute_update(grad_W, grad_b, step)
        
        # Add decoupled weight decay: Δθ ← Δθ - η*λ*θ
        update_W -= self.learning_rate * self.weight_decay * self.W
        update_b -= self.learning_rate * self.weight_decay * self.b
        
        return update_W, update_b
    
    def get_hyperparameters(self) -> Dict:
        """Return AdamW hyperparameters."""
        params = super().get_hyperparameters()
        params['optimizer'] = 'AdamW'
        params['weight_decay'] = self.weight_decay
        return params


if __name__ == "__main__":
    # Test adaptive optimizers
    print("Testing Adaptive Learning Rate Optimizers\n")
    
    from data_generator import LinearDataGenerator
    
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    optimizers = [
        ('AdaGrad', AdaGrad(learning_rate=0.1, epochs=300, random_seed=42)),
        ('RMSProp', RMSProp(learning_rate=0.01, epochs=300, random_seed=42)),
        ('Adam', Adam(learning_rate=0.01, epochs=300, random_seed=42)),
        ('AdamW', AdamW(learning_rate=0.01, weight_decay=0.01, epochs=300, random_seed=42))
    ]
    
    results = {}
    
    for name, optimizer in optimizers:
        print("="*70)
        print(f"Training with {name}")
        print("="*70)
        optimizer.fit(X, y, verbose=False)
        final_loss = optimizer.get_history()['loss_history'][-1]
        results[name] = final_loss
        print(f"Final loss: {final_loss:.6f}\n")
    
    print("="*70)
    print("Final Comparison")
    print("="*70)
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:12s}: {loss:.6f}")