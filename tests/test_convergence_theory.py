"""
Unit tests for convergence theory implementation.

Tests:
1. Eigenvalue computation
2. Lipschitz constant correctness
3. Strong convexity parameter
4. Optimal learning rate derivation
5. Convergence rate formulas
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theory.convergence_proof import ConvergenceAnalyzer


class TestConvergenceAnalyzer:
    """Test suite for ConvergenceAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.m, self.n = 100, 1
        self.X = 2 * np.random.rand(self.m, self.n)
        self.y = 5 + 2 * self.X + np.random.randn(self.m, 1)
        self.analyzer = ConvergenceAnalyzer(self.X, self.y)
    
    def test_hessian_shape(self):
        """Test Hessian matrix has correct shape."""
        H = self.analyzer.compute_hessian()
        assert H.shape == (self.n, self.n), f"Expected shape ({self.n}, {self.n}), got {H.shape}"
    
    def test_hessian_symmetry(self):
        """Test Hessian is symmetric."""
        H = self.analyzer.compute_hessian()
        assert np.allclose(H, H.T), "Hessian should be symmetric"
    
    def test_hessian_positive_semidefinite(self):
        """Test Hessian is positive semi-definite."""
        eigenvalues = self.analyzer.compute_eigenvalues()
        assert np.all(eigenvalues >= -1e-10), f"Eigenvalues should be non-negative, got min={eigenvalues.min()}"
    
    def test_lipschitz_constant_positive(self):
        """Test Lipschitz constant is positive."""
        L = self.analyzer.compute_lipschitz_constant()
        assert L > 0, f"Lipschitz constant should be positive, got {L}"
    
    def test_strong_convexity_positive(self):
        """Test strong convexity parameter is positive."""
        mu = self.analyzer.compute_strong_convexity_parameter()
        assert mu > 0, f"Strong convexity parameter should be positive, got {mu}"
    
    def test_condition_number_geq_one(self):
        """Test condition number is at least 1."""
        kappa = self.analyzer.compute_condition_number()
        assert kappa >= 1.0, f"Condition number should be >= 1, got {kappa}"
    
    def test_optimal_learning_rate_in_range(self):
        """Test optimal learning rate is in valid range."""
        L = self.analyzer.compute_lipschitz_constant()
        mu = self.analyzer.compute_strong_convexity_parameter()
        eta_opt = self.analyzer.compute_optimal_learning_rate()
        
        assert 0 < eta_opt < 2/L, f"Optimal eta should be in (0, 2/L), got {eta_opt}"
        assert np.isclose(eta_opt, 2/(L+mu)), "Optimal eta formula incorrect"
    
    def test_convergence_rate_less_than_one(self):
        """Test convergence rate with optimal learning rate is < 1."""
        eta_opt = self.analyzer.compute_optimal_learning_rate()
        rho = self.analyzer.compute_convergence_rate(eta_opt)
        
        assert 0 <= rho < 1, f"Convergence rate should be in [0, 1), got {rho}"
    
    def test_convergence_rate_formula(self):
        """Test convergence rate formula for optimal learning rate."""
        kappa = self.analyzer.compute_condition_number()
        eta_opt = self.analyzer.compute_optimal_learning_rate()
        rho = self.analyzer.compute_convergence_rate(eta_opt)
        
        rho_theoretical = (kappa - 1) / (kappa + 1)
        assert np.isclose(rho, rho_theoretical, rtol=1e-6), \
            f"Convergence rate formula incorrect: {rho} vs {rho_theoretical}"
    
    def test_unstable_learning_rate(self):
        """Test that large learning rate gives rho >= 1."""
        L = self.analyzer.compute_lipschitz_constant()
        eta_unstable = 2.5 / L  # Beyond stability limit
        rho = self.analyzer.compute_convergence_rate(eta_unstable)
        
        assert rho >= 1.0, f"Unstable learning rate should give rho >= 1, got {rho}"
    
    def test_iterations_to_accuracy_positive(self):
        """Test iterations to accuracy is positive."""
        eta_opt = self.analyzer.compute_optimal_learning_rate()
        k = self.analyzer.compute_iterations_to_accuracy(eta_opt, epsilon=1e-6)
        
        assert k > 0, f"Iterations should be positive, got {k}"
        assert isinstance(k, (int, float)), "Iterations should be numeric"
    
    def test_verify_convergence_stable_rate(self):
        """Test convergence verification for stable learning rate."""
        eta_opt = self.analyzer.compute_optimal_learning_rate()
        result = self.analyzer.verify_convergence_guarantee(eta_opt)
        
        assert result['positive'], "Learning rate should be positive"
        assert result['below_stability_limit'], "Should be below stability limit"
        assert result['converges'], "Should converge"
    
    def test_verify_convergence_unstable_rate(self):
        """Test convergence verification for unstable learning rate."""
        L = self.analyzer.compute_lipschitz_constant()
        eta_unstable = 3.0 / L
        result = self.analyzer.verify_convergence_guarantee(eta_unstable)
        
        assert not result['converges'], "Should not converge with unstable rate"
    
    def test_get_full_analysis_completeness(self):
        """Test that full analysis returns all expected fields."""
        analysis = self.analyzer.get_full_analysis()
        
        required_fields = [
            'eigenvalues',
            'lipschitz_constant',
            'strong_convexity_parameter',
            'condition_number',
            'optimal_learning_rate',
            'optimal_convergence_rate',
            'iterations_to_1e-6',
            'iterations_to_1e-10'
        ]
        
        for field in required_fields:
            assert field in analysis, f"Missing field: {field}"
    
    def test_eigenvalues_sorted(self):
        """Test eigenvalues are sorted in descending order."""
        eigenvalues = self.analyzer.compute_eigenvalues()
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), "Eigenvalues should be sorted descending"
    
    def test_manual_hessian_computation(self):
        """Test Hessian computation against manual calculation."""
        H_computed = self.analyzer.compute_hessian()
        H_manual = (1/self.m) * (self.X.T @ self.X)
        
        assert np.allclose(H_computed, H_manual), "Hessian computation incorrect"


class TestNumericalEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_perfectly_conditioned_problem(self):
        """Test with identity-like Hessian (kappa = 1)."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        X = X / np.linalg.norm(X) * np.sqrt(100)  # Normalize to make H ≈ I
        y = 2 * X + 5
        
        analyzer = ConvergenceAnalyzer(X, y)
        kappa = analyzer.compute_condition_number()
        
        assert 0.9 < kappa < 1.1, f"Should be well-conditioned (kappa ≈ 1), got {kappa}"
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([[2.0], [4.0], [6.0]])
        
        analyzer = ConvergenceAnalyzer(X, y)
        analysis = analyzer.get_full_analysis()
        
        assert analysis['lipschitz_constant'] > 0
        assert analysis['strong_convexity_parameter'] > 0
    
    def test_zero_mean_features(self):
        """Test with zero-mean features."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        X = X - X.mean()  # Zero mean
        y = 2 * X + np.random.randn(100, 1)
        
        analyzer = ConvergenceAnalyzer(X, y)
        H = analyzer.compute_hessian()
        
        assert not np.isnan(H).any(), "Hessian should not contain NaN"
        assert not np.isinf(H).any(), "Hessian should not contain Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])