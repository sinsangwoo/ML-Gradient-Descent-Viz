"""
Stochastic Gradient Descent (SGD)

Implements three variants:
1. Batch GD: Uses full dataset per iteration
2. Mini-batch SGD: Uses random subsets
3. Online SGD: Single sample per iteration

Theory:
For convex functions with Lipschitz gradients (constant L):
- Batch GD converges as O(1/k) for optimal step size
- SGD converges as O(1/√k) in expectation

References:
- Robbins & Monro (1951). A Stochastic Approximation Method
- Bottou (2010). Large-Scale Machine Learning with Stochastic Gradient Descent
"""

import numpy as np
from typing import Dict, Optional
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule:
        θ_{t+1} = θ_t - η ∇J(θ_t; batch)
    
    where batch can be full dataset, mini-batch, or single sample.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 batch_size: Optional[int] = None,
                 shuffle: bool = True,
                 random_seed: Optional[int] = None):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size
        batch_size : int, optional
            Batch size for mini-batch SGD
            - None: full batch (standard GD)
            - int: mini-batch SGD
            - 1: online SGD
        shuffle : bool
            Whether to shuffle data each epoch
        random_seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(learning_rate=learning_rate, name="SGD")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Perform SGD update.
        
        Parameters:
        -----------
        params : ndarray
            Current parameters
        gradient : ndarray
            Gradient (possibly from mini-batch)
            
        Returns:
        --------
        new_params : ndarray
            Updated parameters
        """
        # Standard SGD update
        new_params = params - self.learning_rate * gradient
        return new_params
    
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle
        }
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, 
                       epoch: int = 0) -> list:
        """
        Create mini-batches from data.
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Features
        y : ndarray, shape (m, 1)
            Targets
        epoch : int
            Current epoch (for seeding)
            
        Returns:
        --------
        batches : list of tuples
            [(X_batch, y_batch), ...]
        """
        m = X.shape[0]
        
        # Full batch
        if self.batch_size is None or self.batch_size >= m:
            return [(X, y)]
        
        # Shuffle indices
        indices = np.arange(m)
        if self.shuffle:
            if self.random_seed is not None:
                np.random.seed(self.random_seed + epoch)
            np.random.shuffle(indices)
        
        # Create mini-batches
        batches = []
        for i in range(0, m, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches