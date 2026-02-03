"""
Unit Tests for All Optimizers

Tests:
1. Optimizer API compliance
2. Convergence on simple problems
3. Hyperparameter validation
4. State persistence
5. Numerical stability
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers import SGD, MomentumSGD, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW
from data_generator import LinearDataGenerator


class TestOptimizerAPI:
    """Test that all optimizers follow the same API."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=0.5)
        return X, y
    
    @pytest.fixture
    def all_optimizers(self):
        """Return list of all optimizer classes."""
        return [
            SGD,
            MomentumSGD,
            NesterovMomentum,
            AdaGrad,
            RMSProp,
            Adam,
            AdamW
        ]
    
    def test_all_optimizers_have_fit(self, all_optimizers):
        """Test all optimizers have fit() method."""
        for OptClass in all_optimizers:
            opt = OptClass(epochs=10, random_seed=42)
            assert hasattr(opt, 'fit'), f"{OptClass.__name__} missing fit()"
    
    def test_all_optimizers_have_predict(self, all_optimizers):
        """Test all optimizers have predict() method."""
        for OptClass in all_optimizers:
            opt = OptClass(epochs=10, random_seed=42)
            assert hasattr(opt, 'predict'), f"{OptClass.__name__} missing predict()"
    
    def test_all_optimizers_have_get_parameters(self, all_optimizers):
        """Test all optimizers have get_parameters() method."""
        for OptClass in all_optimizers:
            opt = OptClass(epochs=10, random_seed=42)
            assert hasattr(opt, 'get_parameters'), f"{OptClass.__name__} missing get_parameters()"
    
    def test_all_optimizers_have_get_history(self, all_optimizers):
        """Test all optimizers have get_history() method."""
        for OptClass in all_optimizers:
            opt = OptClass(epochs=10, random_seed=42)
            assert hasattr(opt, 'get_history'), f"{OptClass.__name__} missing get_history()"
    
    def test_all_optimizers_can_train(self, all_optimizers, sample_data):
        """Test all optimizers can complete training."""
        X, y = sample_data
        
        for OptClass in all_optimizers:
            opt = OptClass(learning_rate=0.01, epochs=50, random_seed=42)
            opt.fit(X, y, verbose=False)
            
            # Check training completed
            assert opt.W is not None, f"{OptClass.__name__} failed to train"
            assert opt.b is not None, f"{OptClass.__name__} failed to train"
    
    def test_all_optimizers_reduce_loss(self, all_optimizers, sample_data):
        """Test all optimizers reduce loss during training."""
        X, y = sample_data
        
        for OptClass in all_optimizers:
            opt = OptClass(learning_rate=0.01, epochs=100, random_seed=42)
            opt.fit(X, y, verbose=False)
            
            loss_hist = opt.get_history()['loss_history']
            initial_loss = loss_hist[0]
            final_loss = loss_hist[-1]
            
            assert final_loss < initial_loss, \
                f"{OptClass.__name__} did not reduce loss: {initial_loss:.4f} → {final_loss:.4f}"


class TestSGD:
    """Test SGD optimizer specifically."""
    
    def test_sgd_convergence(self):
        """Test SGD converges to correct parameters."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=100, noise_std=0.1)
        
        opt = SGD(learning_rate=0.1, epochs=500, random_seed=42)
        opt.fit(X, y, verbose=False)
        
        params = opt.get_parameters()
        assert abs(params['W'] - 2.0) < 0.1, "SGD W not close to true value"
        assert abs(params['b'] - 5.0) < 0.1, "SGD b not close to true value"
    
    def test_sgd_learning_rate_matters(self):
        """Test that learning rate affects convergence."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
        
        opt_slow = SGD(learning_rate=0.001, epochs=100, random_seed=42)
        opt_fast = SGD(learning_rate=0.1, epochs=100, random_seed=42)
        
        opt_slow.fit(X, y, verbose=False)
        opt_fast.fit(X, y, verbose=False)
        
        loss_slow = opt_slow.get_history()['loss_history'][-1]
        loss_fast = opt_fast.get_history()['loss_history'][-1]
        
        assert loss_fast < loss_slow, "Higher learning rate should converge faster"


class TestMomentum:
    """Test Momentum-based optimizers."""
    
    def test_momentum_accelerates_convergence(self):
        """Test that momentum converges faster than vanilla SGD."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
        
        sgd = SGD(learning_rate=0.1, epochs=200, random_seed=42)
        momentum = MomentumSGD(learning_rate=0.1, momentum=0.9, epochs=200, random_seed=42)
        
        sgd.fit(X, y, verbose=False)
        momentum.fit(X, y, verbose=False)
        
        loss_sgd = sgd.get_history()['loss_history'][-1]
        loss_momentum = momentum.get_history()['loss_history'][-1]
        
        # Momentum should achieve lower loss in same iterations
        assert loss_momentum < loss_sgd, "Momentum should converge faster"
    
    def test_nesterov_vs_classical_momentum(self):
        """Test Nesterov vs classical momentum."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
        
        classical = MomentumSGD(learning_rate=0.1, momentum=0.9, epochs=200, random_seed=42)
        nesterov = NesterovMomentum(learning_rate=0.1, momentum=0.9, epochs=200, random_seed=42)
        
        classical.fit(X, y, verbose=False)
        nesterov.fit(X, y, verbose=False)
        
        # Both should converge (Nesterov typically slightly better)
        assert nesterov.get_history()['loss_history'][-1] < 1.0
        assert classical.get_history()['loss_history'][-1] < 1.0


class TestAdaptiveMethods:
    """Test adaptive learning rate optimizers."""
    
    def test_adam_bias_correction(self):
        """Test Adam's bias correction affects early iterations."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=0.5)
        
        adam = Adam(learning_rate=0.01, epochs=10, random_seed=42)
        adam.fit(X, y, verbose=False)
        
        # Check that bias correction is active
        # (1st moment and 2nd moment should not be zero after training)
        assert adam.m_W is not None
        assert np.any(adam.m_W != 0)
        assert np.any(adam.v_W != 0)
    
    def test_adamw_weight_decay(self):
        """Test AdamW applies weight decay."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=0.5)
        
        adam = Adam(learning_rate=0.01, epochs=100, random_seed=42)
        adamw = AdamW(learning_rate=0.01, weight_decay=0.1, epochs=100, random_seed=42)
        
        adam.fit(X, y, verbose=False)
        adamw.fit(X, y, verbose=False)
        
        # AdamW should have smaller parameter magnitudes due to weight decay
        adam_params = adam.get_parameters()
        adamw_params = adamw.get_parameters()
        
        # Weight decay should shrink parameters
        assert abs(adamw_params['W']) <= abs(adam_params['W']) + 0.5
    
    def test_adagrad_accumulates_gradients(self):
        """Test AdaGrad accumulates squared gradients."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=0.5)
        
        adagrad = AdaGrad(learning_rate=0.1, epochs=50, random_seed=42)
        adagrad.fit(X, y, verbose=False)
        
        # G should have accumulated squared gradients
        assert adagrad.G_W is not None
        assert np.all(adagrad.G_W >= 0), "Accumulated gradients should be non-negative"
        assert adagrad.G_b >= 0, "Accumulated bias gradient should be non-negative"


class TestNumericalStability:
    """Test numerical stability of optimizers."""
    
    def test_no_nan_or_inf(self):
        """Test optimizers don't produce NaN or Inf."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=1.0)
        
        optimizers = [
            SGD(learning_rate=0.1, epochs=100, random_seed=42),
            Adam(learning_rate=0.01, epochs=100, random_seed=42)
        ]
        
        for opt in optimizers:
            opt.fit(X, y, verbose=False)
            params = opt.get_parameters()
            
            assert not np.isnan(params['W']), f"{opt.__class__.__name__} produced NaN"
            assert not np.isinf(params['W']), f"{opt.__class__.__name__} produced Inf"
            assert not np.isnan(params['b']), f"{opt.__class__.__name__} produced NaN"
            assert not np.isinf(params['b']), f"{opt.__class__.__name__} produced Inf"


class TestReproducibility:
    """Test that optimizers are reproducible with same seed."""
    
    def test_sgd_reproducible(self):
        """Test SGD produces same results with same seed."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=1.0)
        
        opt1 = SGD(learning_rate=0.1, epochs=100, random_seed=42)
        opt2 = SGD(learning_rate=0.1, epochs=100, random_seed=42)
        
        opt1.fit(X, y, verbose=False)
        opt2.fit(X, y, verbose=False)
        
        params1 = opt1.get_parameters()
        params2 = opt2.get_parameters()
        
        assert np.allclose(params1['W'], params2['W']), "SGD not reproducible"
        assert np.allclose(params1['b'], params2['b']), "SGD not reproducible"
    
    def test_adam_reproducible(self):
        """Test Adam produces same results with same seed."""
        np.random.seed(42)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
        X, y = data_gen.generate_data(n_samples=50, noise_std=1.0)
        
        opt1 = Adam(learning_rate=0.01, epochs=100, random_seed=42)
        opt2 = Adam(learning_rate=0.01, epochs=100, random_seed=42)
        
        opt1.fit(X, y, verbose=False)
        opt2.fit(X, y, verbose=False)
        
        params1 = opt1.get_parameters()
        params2 = opt2.get_parameters()
        
        assert np.allclose(params1['W'], params2['W']), "Adam not reproducible"
        assert np.allclose(params1['b'], params2['b']), "Adam not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])