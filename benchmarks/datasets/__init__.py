"""
Large-Scale Benchmark Datasets
================================

Provides loaders for industry-standard datasets:
- MNIST (784-dim, 60k samples)
- California Housing (8-dim, 20k samples)  
- Synthetic High-Dimensional (d=1,000-10,000)
- Extreme Condition Number datasets (κ → ∞)
"""

from .mnist_loader import load_mnist
from .california_housing_loader import load_california_housing
from .synthetic_highdim import generate_highdim_regression
from .extreme_conditioning import generate_extreme_condition_data

__all__ = [
    'load_mnist',
    'load_california_housing',
    'generate_highdim_regression',
    'generate_extreme_condition_data'
]
