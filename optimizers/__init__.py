"""
Optimizer Zoo - Collection of optimization algorithms with theoretical guarantees.

This module implements various first-order and second-order optimization methods:
- First-order: SGD, Momentum, Nesterov, Adam, RMSProp, AdaGrad
- Second-order: Newton, Quasi-Newton (BFGS, L-BFGS)
- Variance reduction: SVRG, SARAH
- Line search: Armijo, Wolfe conditions

All optimizers inherit from BaseOptimizer and follow a unified API.
"""

from .base_optimizer import BaseOptimizer
from .sgd import SGD
from .momentum import MomentumSGD, NesterovMomentum
from .adaptive import AdaGrad, RMSProp, Adam, AdamW

__all__ = [
    'BaseOptimizer',
    'SGD',
    'MomentumSGD',
    'NesterovMomentum',
    'AdaGrad',
    'RMSProp',
    'Adam',
    'AdamW'
]