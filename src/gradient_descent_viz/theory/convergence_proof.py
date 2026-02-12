"""
Convergence Theory Implementation

This module implements theoretical convergence guarantees for gradient descent:
- Lipschitz continuity constant estimation
- Strong convexity parameter (μ) estimation  
- Theoretical convergence rate bounds
- Condition number analysis
- Optimal learning rate computation

References:
- Nesterov, Y. (2004). Introductory Lectures on Convex Optimization
- Boyd, S. & Vandenberghe, L. (2004). Convex Optimization
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings


class ConvergenceAnalyzer:
    """
    Analyzes convergence properties of gradient descent for quadratic loss functions.
    
    For a quadratic loss J(θ) = (1/2m)||Xθ - y||², this analyzer computes:
    - Lipschitz constant L of ∇J
    - Strong convexity parameter μ
    - Condition number κ = L/μ
    - Theoretical optimal learning rate
    - Convergence rate bounds
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-10):
        """
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Feature matrix
        y : ndarray, shape (m, 1)
            Target vector
        epsilon : float
            Small constant to avoid division by zero
        """
        self.X = X
        self.y = y
        self.m = X.shape[0]  # number of samples
        self.n = X.shape[1]  # number of features
        self.epsilon = epsilon
        
        # Cache for computed properties
        self._hessian = None
        self._eigenvalues = None
        self._L = None
        self._mu = None
        
    def compute_hessian(self) -> np.ndarray:
        """
        Compute the Hessian matrix for quadratic loss.
        
        For J(θ) = (1/2m)||Xθ - y||², the Hessian is:
        H = (1/m) X^T X
        
        Returns:
        --------
        H : ndarray, shape (n, n)
            Hessian matrix
        """
        if self._hessian is None:
            self._hessian = (1/self.m) * (self.X.T @ self.X)
        return self._hessian
    
    def compute_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues of the Hessian matrix.
        
        Returns:
        --------
        eigenvalues : ndarray
            Sorted eigenvalues in descending order
        """
        if self._eigenvalues is None:
            H = self.compute_hessian()
            eigvals = np.linalg.eigvalsh(H)  # For symmetric matrices
            self._eigenvalues = np.sort(eigvals)[::-1]  # Descending order
        return self._eigenvalues
    
    def compute_lipschitz_constant(self) -> float:
        """
        Compute the Lipschitz constant L of the gradient.
        
        For quadratic functions, L equals the largest eigenvalue of the Hessian:
        L = λ_max(H)
        
        This guarantees: ||∇J(θ₁) - ∇J(θ₂)|| ≤ L||θ₁ - θ₂||
        
        Returns:
        --------
        L : float
            Lipschitz constant
        """
        if self._L is None:
            eigenvalues = self.compute_eigenvalues()
            self._L = eigenvalues[0]  # Maximum eigenvalue
        return self._L
    
    def compute_strong_convexity_parameter(self) -> float:
        """
        Compute the strong convexity parameter μ.
        
        For quadratic functions, μ equals the smallest eigenvalue of the Hessian:
        μ = λ_min(H)
        
        This guarantees: J(θ₂) ≥ J(θ₁) + ∇J(θ₁)^T(θ₂-θ₁) + (μ/2)||θ₂-θ₁||²
        
        Returns:
        --------
        mu : float
            Strong convexity parameter
        """
        if self._mu is None:
            eigenvalues = self.compute_eigenvalues()
            self._mu = max(eigenvalues[-1], self.epsilon)  # Minimum eigenvalue (avoid zero)
        return self._mu
    
    def compute_condition_number(self) -> float:
        """
        Compute the condition number κ = L/μ.
        
        The condition number measures the "difficulty" of the optimization problem:
        - κ = 1: perfectly conditioned (all eigenvalues equal)
        - κ >> 1: ill-conditioned (large disparity in eigenvalues)
        
        Higher κ means slower convergence.
        
        Returns:
        --------
        kappa : float
            Condition number
        """
        L = self.compute_lipschitz_constant()
        mu = self.compute_strong_convexity_parameter()
        return L / mu
    
    def compute_optimal_learning_rate(self) -> float:
        """
        Compute the theoretically optimal learning rate.
        
        For strongly convex quadratic functions:
        η_opt = 2 / (L + μ)
        
        This minimizes the worst-case convergence rate.
        
        Returns:
        --------
        eta_opt : float
            Optimal learning rate
        """
        L = self.compute_lipschitz_constant()
        mu = self.compute_strong_convexity_parameter()
        return 2.0 / (L + mu)
    
    def compute_convergence_rate(self, learning_rate: float) -> float:
        """
        Compute the theoretical convergence rate for given learning rate.
        
        For gradient descent with learning rate η:
        ||θ_k - θ*||² ≤ ρ^k ||θ_0 - θ*||²
        
        where ρ is the convergence rate:
        ρ = max(|1 - ηL|, |1 - ημ|)
        
        For optimal η = 2/(L+μ):
        ρ = (κ-1)/(κ+1) where κ = L/μ
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate to analyze
            
        Returns:
        --------
        rho : float
            Convergence rate (< 1 for convergence)
        """
        L = self.compute_lipschitz_constant()
        mu = self.compute_strong_convexity_parameter()
        
        rho = max(abs(1 - learning_rate * L), abs(1 - learning_rate * mu))
        return rho
    
    def compute_iterations_to_accuracy(self, learning_rate: float, epsilon: float) -> int:
        """
        Compute the number of iterations needed to reach ε-accuracy.
        
        Theoretical bound:
        k ≥ log(1/ε) / log(1/ρ)
        
        where ρ is the convergence rate.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        epsilon : float
            Desired accuracy (relative error)
            
        Returns:
        --------
        k : int
            Minimum number of iterations
        """
        rho = self.compute_convergence_rate(learning_rate)
        
        if rho >= 1.0:
            warnings.warn(f"Convergence rate ρ={rho:.4f} ≥ 1. Algorithm may diverge!")
            return np.inf
        
        k = np.log(1/epsilon) / np.log(1/rho)
        return int(np.ceil(k))
    
    def verify_convergence_guarantee(self, learning_rate: float) -> Dict[str, bool]:
        """
        Verify theoretical convergence guarantees for given learning rate.
        
        Checks:
        1. 0 < η < 2/L (necessary condition)
        2. ρ < 1 (convergence condition)
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate to verify
            
        Returns:
        --------
        checks : dict
            Dictionary of verification results
        """
        L = self.compute_lipschitz_constant()
        rho = self.compute_convergence_rate(learning_rate)
        
        return {
            'positive': learning_rate > 0,
            'below_stability_limit': learning_rate < 2.0 / L,
            'converges': rho < 1.0,
            'learning_rate': learning_rate,
            'convergence_rate': rho,
            'stability_limit': 2.0 / L
        }
    
    def get_full_analysis(self) -> Dict:
        """
        Get complete convergence analysis.
        
        Returns:
        --------
        analysis : dict
            Complete analysis including all theoretical properties
        """
        eigenvalues = self.compute_eigenvalues()
        L = self.compute_lipschitz_constant()
        mu = self.compute_strong_convexity_parameter()
        kappa = self.compute_condition_number()
        eta_opt = self.compute_optimal_learning_rate()
        rho_opt = self.compute_convergence_rate(eta_opt)
        
        return {
            'eigenvalues': eigenvalues,
            'lipschitz_constant': L,
            'strong_convexity_parameter': mu,
            'condition_number': kappa,
            'optimal_learning_rate': eta_opt,
            'optimal_convergence_rate': rho_opt,
            'iterations_to_1e-6': self.compute_iterations_to_accuracy(eta_opt, 1e-6),
            'iterations_to_1e-10': self.compute_iterations_to_accuracy(eta_opt, 1e-10)
        }
    
    def print_analysis(self):
        """Print formatted convergence analysis."""
        analysis = self.get_full_analysis()
        
        print("="*70)
        print("CONVERGENCE THEORY ANALYSIS")
        print("="*70)
        print(f"\n[Problem Properties]")
        print(f"  Number of samples (m): {self.m}")
        print(f"  Number of features (n): {self.n}")
        print(f"\n[Eigenvalue Spectrum]")
        print(f"  λ_max: {analysis['eigenvalues'][0]:.6e}")
        print(f"  λ_min: {analysis['eigenvalues'][-1]:.6e}")
        if len(analysis['eigenvalues']) <= 5:
            print(f"  All eigenvalues: {analysis['eigenvalues']}")
        print(f"\n[Convergence Parameters]")
        print(f"  Lipschitz constant (L): {analysis['lipschitz_constant']:.6e}")
        print(f"  Strong convexity (μ):   {analysis['strong_convexity_parameter']:.6e}")
        print(f"  Condition number (κ):   {analysis['condition_number']:.6e}")
        print(f"\n[Optimal Configuration]")
        print(f"  Optimal learning rate:  {analysis['optimal_learning_rate']:.6e}")
        print(f"  Convergence rate (ρ):   {analysis['optimal_convergence_rate']:.6e}")
        print(f"\n[Convergence Speed]")
        print(f"  Iterations to 10⁻⁶:     {analysis['iterations_to_1e-6']}")
        print(f"  Iterations to 10⁻¹⁰:    {analysis['iterations_to_1e-10']}")
        print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Testing ConvergenceAnalyzer\n")
    
    # Generate synthetic data
    np.random.seed(42)
    m, n = 100, 1
    X = 2 * np.random.rand(m, n)
    y = 5 + 2 * X + np.random.randn(m, 1)
    
    # Analyze convergence properties
    analyzer = ConvergenceAnalyzer(X, y)
    analyzer.print_analysis()
    
    # Test with different learning rates
    print("\n" + "="*70)
    print("LEARNING RATE VERIFICATION")
    print("="*70)
    
    test_rates = [0.01, 0.1, 0.5, 1.0]
    for lr in test_rates:
        result = analyzer.verify_convergence_guarantee(lr)
        status = "✓ CONVERGES" if result['converges'] else "✗ DIVERGES"
        print(f"\nη = {lr:.2f}: {status}")
        print(f"  Convergence rate ρ = {result['convergence_rate']:.6f}")
        print(f"  Stability limit = {result['stability_limit']:.6f}")