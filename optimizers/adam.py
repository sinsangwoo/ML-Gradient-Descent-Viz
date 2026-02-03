"""
Adam and AdamW - Adaptive Moment Estimation

Adam combines ideas from Momentum and RMSProp:
- First moment estimate (momentum): E[g]
- Second moment estimate (adaptive learning rate): E[g²]
- Bias correction for initialization

Update rule:
    m_t = β₁ m_{t-1} + (1-β₁) g_t           # First moment (momentum)
    v_t = β₂ v_{t-1} + (1-β₂) g_t²          # Second moment (variance)
    
    m̂_t = m_t / (1 - β₁^t)                  # Bias correction
    v̂_t = v_t / (1 - β₂^t)                  # Bias correction
    
    θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)

AdamW modifies weight decay:
- Adam: weight decay in loss function (equivalent to L2 regularization)
- AdamW: decoupled weight decay (applied directly to parameters)

References:
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization, ICLR
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization, ICLR
"""

import numpy as np
from typing import Dict
from .base_optimizer import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam optimizer - combines momentum and adaptive learning rates.
    
    Key features:
    - Adaptive per-parameter learning rates
    - Momentum-like updates
    - Bias correction for initialization
    - Generally robust across many tasks
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 amsgrad: bool = False):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate η (default 0.001 for Adam)
        beta1 : float
            Exponential decay rate for first moment estimate
            Typical: 0.9
        beta2 : float
            Exponential decay rate for second moment estimate
            Typical: 0.999 (longer memory than beta1)
        epsilon : float
            Small constant for numerical stability
        amsgrad : bool
            Whether to use AMSGrad variant (Reddi et al., 2018)
            Maintains max of past v_t for more conservative updates
        """
        super().__init__(learning_rate=learning_rate, name="Adam")
        
        if not 0 <= beta1 < 1:
            raise ValueError(f"Beta1 must be in [0, 1), got {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Beta2 must be in [0, 1), got {beta2}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be > 0, got {epsilon}")
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.v_max = None  # Max of v_t (only for AMSGrad)
        self.t = 0  # Time step (for bias correction)
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform Adam update.
        
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
        # Initialize moments on first step
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            if self.amsgrad:
                self.v_max = np.zeros_like(params)
        
        # Increment time step
        self.t += 1
        
        # Update biased first moment estimate: m_t = β₁ m_{t-1} + (1-β₁) g_t
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update biased second moment estimate: v_t = β₂ v_{t-1} + (1-β₂) g_t²
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # AMSGrad: use max of v_hat for stability
        if self.amsgrad:
            self.v_max = np.maximum(self.v_max, v_hat)
            denominator = np.sqrt(self.v_max) + self.epsilon
        else:
            denominator = np.sqrt(v_hat) + self.epsilon
        
        # Update parameters: θ = θ - η m̂ / (√v̂ + ε)
        new_params = params - self.learning_rate * m_hat / denominator
        
        return new_params
    
    def get_effective_lr(self) -> float:
        """
        Get mean effective learning rate across all parameters.
        
        Returns:
        --------
        mean_lr : float
            Average effective learning rate
        """
        if self.m is None or self.t == 0:
            return self.learning_rate
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        if self.amsgrad and self.v_max is not None:
            denominator = np.sqrt(self.v_max) + self.epsilon
        else:
            denominator = np.sqrt(v_hat) + self.epsilon
        
        # Effective LR per parameter
        adapted_lrs = self.learning_rate * np.abs(m_hat) / denominator
        return np.mean(adapted_lrs)
    
    def reset(self):
        """Reset optimizer state including moments."""
        super().reset()
        self.m = None
        self.v = None
        self.v_max = None
        self.t = 0
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }


class AdamW(Adam):
    """
    AdamW - Adam with decoupled weight decay.
    
    Key difference from Adam:
    - Weight decay applied directly to parameters, not through gradients
    - Decouples regularization from adaptive learning rate
    - Often improves generalization
    
    Update rule:
        [Same moment updates as Adam]
        θ_{t+1} = (1 - λη) θ_t - η m̂_t / (√v̂_t + ε)
    
    where λ is the weight decay coefficient.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 weight_decay: float = 0.01,
                 amsgrad: bool = False):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate η
        beta1, beta2, epsilon : float
            Adam hyperparameters (same as Adam)
        weight_decay : float
            Weight decay coefficient λ (L2 penalty coefficient)
            Typical: 0.01 (1% decay per step)
        amsgrad : bool
            Whether to use AMSGrad variant
        """
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad
        )
        
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")
        
        self.name = "AdamW"
        self.weight_decay = weight_decay
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform AdamW update with decoupled weight decay.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Current gradient (without weight decay)
            
        Returns:
        --------
        new_params : ndarray
            Updated parameters
        """
        # Perform standard Adam update (without weight decay in gradient)
        new_params = super().step(params, gradient)
        
        # Apply decoupled weight decay: θ *= (1 - λη)
        new_params = new_params - self.learning_rate * self.weight_decay * params
        
        return new_params
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        config = super().get_config()
        config['weight_decay'] = self.weight_decay
        return config