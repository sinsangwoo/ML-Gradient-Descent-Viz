"""
Base Optimizer Class

Defines the unified interface that all optimizers must implement.
Provides common functionality for parameter tracking, gradient computation,
and convergence monitoring.

References:
- Goodfellow et al. (2016). Deep Learning, Chapter 8: Optimization
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import warnings


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    All optimizers must implement:
    - __init__: Initialize optimizer-specific hyperparameters
    - step: Perform one optimization step
    - get_config: Return optimizer configuration
    """
    
    def __init__(self, learning_rate: float = 0.01, name: str = "BaseOptimizer"):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate (may be adapted by optimizer)
        name : str
            Optimizer name for logging
        """
        self.learning_rate = learning_rate
        self.name = name
        
        # Training state
        self.iteration = 0
        self.parameters = None
        
        # History tracking
        self.param_history = []
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rate_history = []
        
    @abstractmethod
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform one optimization step.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Gradient at current parameters
            
        Returns:
        --------
        new_params : ndarray
            Updated parameters
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """
        Get optimizer configuration.
        
        Returns:
        --------
        config : dict
            Optimizer hyperparameters
        """
        pass
    
    def reset(self):
        """Reset optimizer state (e.g., momentum buffers)."""
        self.iteration = 0
        self.param_history = []
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rate_history = []
    
    def track_step(self, params: np.ndarray, gradient: np.ndarray, 
                   loss: Optional[float] = None):
        """
        Track optimization progress.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Current gradient
        loss : float, optional
            Current loss value
        """
        self.iteration += 1
        self.param_history.append(params.copy())
        self.gradient_norms.append(np.linalg.norm(gradient))
        
        if loss is not None:
            self.loss_history.append(loss)
        
        # Track effective learning rate (may be adapted)
        self.learning_rate_history.append(self.get_effective_lr())
    
    def get_effective_lr(self) -> float:
        """
        Get effective learning rate (may differ from base LR for adaptive methods).
        
        Returns:
        --------
        lr : float
            Effective learning rate
        """
        return self.learning_rate
    
    def get_statistics(self) -> Dict:
        """
        Get optimization statistics.
        
        Returns:
        --------
        stats : dict
            Training statistics
        """
        return {
            'iterations': self.iteration,
            'final_gradient_norm': self.gradient_norms[-1] if self.gradient_norms else None,
            'mean_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else None,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'optimizer': self.name
        }
    
    def __repr__(self) -> str:
        config = self.get_config()
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        return f"{self.name}({config_str})"