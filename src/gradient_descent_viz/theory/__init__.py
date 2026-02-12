"""Theory module for mathematical analysis of optimization algorithms."""

from .convergence_proof import ConvergenceAnalyzer
from .numerical_stability import NumericalStabilityAnalyzer

__all__ = ['ConvergenceAnalyzer', 'NumericalStabilityAnalyzer']