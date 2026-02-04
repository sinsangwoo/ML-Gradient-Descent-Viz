"""
Non-Convex Models Module

Implements models beyond linear regression:
- Polynomial regression (degree 2-10)
- Neural networks (2-layer with various activations)
- Loss landscape analysis tools
"""

from .polynomial_regression import PolynomialRegressor
from .neural_network import TwoLayerNet
from .loss_landscape import LossLandscapeAnalyzer

__all__ = [
    'PolynomialRegressor',
    'TwoLayerNet', 
    'LossLandscapeAnalyzer'
]