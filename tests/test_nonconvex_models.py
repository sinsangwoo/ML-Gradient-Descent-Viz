"""
Unit Tests for Non-Convex Models

Tests:
1. Polynomial regression convergence
2. Neural network forward/backward pass
3. Loss landscape analysis
4. Hessian computation
5. Critical point classification
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import PolynomialRegressor, TwoLayerNet, LossLandscapeAnalyzer


class TestPolynomialRegression:
    """Test polynomial regression."""
    
    def test_linear_polynomial(self):
        """Test degree-1 polynomial (should behave like linear regression)."""
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.flatten() + 5 + np.random.randn(50) * 0.5
        
        model = PolynomialRegressor(degree=1, random_seed=42)
        model.fit(X, y, learning_rate=0.01, epochs=500, verbose=False)
        
        coeffs = model.get_coefficients()
        
        # Should recover approximately [5, 2]
        assert abs(coeffs[0] - 5) < 1.0, "Bias not recovered"
        assert abs(coeffs[1] - 2) < 0.5, "Slope not recovered"
    
    def test_quadratic_convergence(self):
        """Test quadratic polynomial converges."""
        np.random.seed(42)
        X = np.linspace(-2, 2, 100).reshape(-1, 1)
        y = X.flatten()**2 + np.random.randn(100) * 0.2
        
        model = PolynomialRegressor(degree=2, random_seed=42)
        model.fit(X, y, learning_rate=0.001, epochs=500, verbose=False)
        
        history = model.get_history()
        
        # Loss should decrease
        assert history['loss_history'][-1] < history['loss_history'][0]
    
    def test_high_degree_polynomial(self):
        """Test high-degree polynomial doesn't crash."""
        np.random.seed(42)
        X = np.linspace(-1, 1, 50).reshape(-1, 1)
        y = np.sin(X).flatten()
        
        model = PolynomialRegressor(degree=7, random_seed=42)
        model.fit(X, y, learning_rate=0.00001, epochs=200, verbose=False)
        
        # Should complete without errors
        assert model.weights is not None
        assert len(model.weights) == 8  # degree+1


class TestNeuralNetwork:
    """Test two-layer neural network."""
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        
        net = TwoLayerNet(n_hidden=5, activation='relu', random_seed=42)
        net._initialize_parameters(n_input=3)
        
        y_pred = net._forward(X)
        
        assert y_pred.shape == (10, 1), "Output shape incorrect"
    
    def test_relu_activation(self):
        """Test ReLU activation."""
        net = TwoLayerNet(n_hidden=10, activation='relu')
        
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        
        result = net.activation(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_different_activations(self):
        """Test network works with all activations."""
        np.random.seed(42)
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        
        for activation in ['relu', 'tanh', 'sigmoid']:
            net = TwoLayerNet(n_hidden=5, activation=activation, random_seed=42)
            net.fit(X, y, learning_rate=0.01, epochs=50, verbose=False)
            
            # Should complete without errors
            assert net.W1 is not None
            assert net.W2 is not None
    
    def test_network_convergence(self):
        """Test network reduces loss."""
        np.random.seed(42)
        X = np.linspace(-1, 1, 50).reshape(-1, 1)
        y = X.flatten()**2 + 0.1 * np.random.randn(50)
        
        net = TwoLayerNet(n_hidden=10, activation='relu', random_seed=42)
        net.fit(X, y, learning_rate=0.01, epochs=200, verbose=False)
        
        history = net.get_history()
        
        # Loss should decrease significantly
        assert history['loss_history'][-1] < 0.5 * history['loss_history'][0]
    
    def test_gradient_computation(self):
        """Test gradients are computed (non-zero)."""
        np.random.seed(42)
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        
        net = TwoLayerNet(n_hidden=5, activation='tanh', random_seed=42)
        net._initialize_parameters(n_input=2)
        
        # Forward pass
        y_pred = net._forward(X)
        
        # Backward pass
        grads = net._backward(y)
        
        # Gradients should not be all zeros
        assert np.any(grads['dW1'] != 0)
        assert np.any(grads['dW2'] != 0)


class TestLossLandscapeAnalyzer:
    """Test loss landscape analysis."""
    
    def test_2d_landscape_generation(self):
        """Test 2D landscape generation."""
        # Simple quadratic loss
        def loss_fn(params):
            return np.sum(params**2)
        
        analyzer = LossLandscapeAnalyzer(loss_fn)
        
        center = np.array([0.0, 0.0])
        d1 = np.array([1.0, 0.0])
        d2 = np.array([0.0, 1.0])
        
        Alpha, Beta, losses = analyzer.generate_2d_landscape(
            center, d1, d2, resolution=10
        )
        
        assert Alpha.shape == (10, 10)
        assert Beta.shape == (10, 10)
        assert losses.shape == (10, 10)
        
        # Minimum should be at center (0, 0)
        min_idx = np.unravel_index(np.argmin(losses), losses.shape)
        # Should be close to middle of grid
        assert 3 <= min_idx[0] <= 6
        assert 3 <= min_idx[1] <= 6
    
    def test_hessian_computation(self):
        """Test Hessian computation for quadratic function."""
        # f(x) = x^T A x where A = [[2, 0], [0, 3]]
        def quadratic_loss(params):
            A = np.array([[2, 0], [0, 3]])
            return params.T @ A @ params
        
        analyzer = LossLandscapeAnalyzer(quadratic_loss)
        
        params = np.array([0.0, 0.0])
        H = analyzer.compute_hessian(params)
        
        # Hessian of quadratic should be 2*A
        expected = np.array([[4, 0], [0, 6]])
        
        np.testing.assert_array_almost_equal(H, expected, decimal=2)
    
    def test_critical_point_classification_minimum(self):
        """Test classification of local minimum."""
        # Convex quadratic has minimum at origin
        def loss_fn(params):
            return np.sum(params**2)
        
        analyzer = LossLandscapeAnalyzer(loss_fn)
        
        result = analyzer.classify_critical_point(np.array([0.0, 0.0]))
        
        assert "Local Minimum" in result['type']
        assert np.all(result['eigenvalues'] > 0)
    
    def test_critical_point_classification_saddle(self):
        """Test classification of saddle point."""
        # Saddle: f(x, y) = x^2 - y^2
        def saddle_loss(params):
            return params[0]**2 - params[1]**2
        
        analyzer = LossLandscapeAnalyzer(saddle_loss)
        
        result = analyzer.classify_critical_point(np.array([0.0, 0.0]))
        
        assert "Saddle" in result['type']
        # Should have 1 positive, 1 negative eigenvalue
        assert np.sum(result['eigenvalues'] > 0) == 1
        assert np.sum(result['eigenvalues'] < 0) == 1


class TestNumericalStability:
    """Test numerical stability of non-convex models."""
    
    def test_no_nan_polynomial(self):
        """Test polynomial doesn't produce NaN."""
        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = np.random.randn(30)
        
        model = PolynomialRegressor(degree=3, random_seed=42)
        model.fit(X, y, learning_rate=0.001, epochs=100, verbose=False)
        
        coeffs = model.get_coefficients()
        assert not np.any(np.isnan(coeffs))
    
    def test_no_nan_neural_net(self):
        """Test neural network doesn't produce NaN."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        
        net = TwoLayerNet(n_hidden=5, activation='relu', random_seed=42)
        net.fit(X, y, learning_rate=0.01, epochs=100, verbose=False)
        
        params = net.get_parameters()
        assert not np.any(np.isnan(params['W1']))
        assert not np.any(np.isnan(params['W2']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])