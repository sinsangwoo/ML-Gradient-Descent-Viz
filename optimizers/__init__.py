"""
Optimizer Zoo - Collection of gradient-based optimization algorithms.

This module provides research-grade implementations of various optimization
algorithms with theoretical convergence guarantees and practical enhancements.

First-Order Methods:
- SGD: Stochastic Gradient Descent (batch, mini-batch, online)
- Momentum: Polyak's heavy ball method
- Nesterov: Accelerated gradient with momentum
- AdaGrad: Adaptive learning rates per parameter
- RMSProp: Root Mean Square Propagation
- Adam: Adaptive Moment Estimation
- AdamW: Adam with decoupled weight decay

All optimizers follow a unified interface for easy comparison and benchmarking.
"""

from .base_optimizer import BaseOptimizer
from .sgd import SGD
from .momentum import Momentum, NesterovMomentum
from .adagrad import AdaGrad
from .rmsprop import RMSProp
from .adam import Adam, AdamW

__all__ = [
    'BaseOptimizer',
    'SGD',
    'Momentum',
    'NesterovMomentum',
    'AdaGrad',
    'RMSProp',
    'Adam',
    'AdamW'
]