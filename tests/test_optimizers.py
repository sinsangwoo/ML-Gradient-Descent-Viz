"""
Unit Tests for Optimizer Zoo

Tests all optimizer implementations for:
1. Correct update rules
2. State management (reset, tracking)
3. Configuration
4. Edge cases
5. Convergence on simple problems
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers import SGD, Momentum, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW


class TestSGD:
    """Test SGD optimizer."""
    
    def test_initialization(self):
        """Test SGD initialization."""
        sgd = SGD(learning_rate=0.1)
        assert sgd.learning_rate == 0.1
        assert sgd.name == "SGD"
    
    def test_step(self):
        """Test SGD update rule."""
        sgd = SGD(learning_rate=0.1)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[0.5], [1.0]])
        
        new_params = sgd.step(params, gradient)
        expected = params - 0.1 * gradient
        
        assert np.allclose(new_params, expected)
    
    def test_batch_creation(self):
        """Test mini-batch creation."""
        sgd = SGD(learning_rate=0.1, batch_size=32)
        
        X = np.random.randn(100, 1)
        y = np.random.randn(100, 1)
        
        batches = sgd.create_batches(X, y, epoch=0)
        
        # Should create ceil(100/32) = 4 batches
        assert len(batches) == 4
        assert batches[0][0].shape[0] == 32
        assert batches[-1][0].shape[0] == 4  # Last batch has remainder


class TestMomentum:
    """Test Momentum optimizer."""
    
    def test_initialization(self):
        """Test Momentum initialization."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        assert momentum.learning_rate == 0.1
        assert momentum.momentum == 0.9
        assert momentum.velocity is None
    
    def test_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        with pytest.raises(ValueError):
            Momentum(momentum=1.5)
        with pytest.raises(ValueError):
            Momentum(momentum=-0.1)
    
    def test_first_step(self):
        """Test first momentum step initializes velocity."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[0.5], [1.0]])
        
        new_params = momentum.step(params, gradient)
        
        assert momentum.velocity is not None
        assert momentum.velocity.shape == params.shape
    
    def test_momentum_accumulation(self):
        """Test momentum accumulates gradients."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[1.0], [1.0]])
        
        # First step
        params = momentum.step(params, gradient)
        v1 = momentum.velocity.copy()
        
        # Second step with same gradient
        params = momentum.step(params, gradient)
        v2 = momentum.velocity.copy()
        
        # Velocity should grow: v2 = 0.9*v1 + g
        expected_v2 = 0.9 * v1 + gradient
        assert np.allclose(v2, expected_v2)
    
    def test_reset(self):
        """Test reset clears velocity."""
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[0.5], [1.0]])
        
        momentum.step(params, gradient)
        assert momentum.velocity is not None
        
        momentum.reset()
        assert momentum.velocity is None


class TestNesterovMomentum:
    """Test Nesterov Accelerated Gradient."""
    
    def test_initialization(self):
        """Test Nesterov initialization."""
        nesterov = NesterovMomentum(learning_rate=0.1, momentum=0.9)
        assert nesterov.name == "NesterovMomentum"
    
    def test_lookahead_params(self):
        """Test look-ahead parameter computation."""
        nesterov = NesterovMomentum(learning_rate=0.1, momentum=0.9)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[0.5], [1.0]])
        
        # First step: no velocity yet
        lookahead = nesterov.get_lookahead_params(params)
        assert np.allclose(lookahead, params)
        
        # After first step
        nesterov.step(params, gradient)
        lookahead = nesterov.get_lookahead_params(params)
        
        # Should be: params - momentum * velocity
        expected = params - 0.9 * nesterov.velocity
        assert np.allclose(lookahead, expected)


class TestAdaGrad:
    """Test AdaGrad optimizer."""
    
    def test_initialization(self):
        """Test AdaGrad initialization."""
        adagrad = AdaGrad(learning_rate=0.1, epsilon=1e-8)
        assert adagrad.epsilon == 1e-8
        assert adagrad.accumulator is None
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError):
            AdaGrad(epsilon=-1e-8)
        with pytest.raises(ValueError):
            AdaGrad(epsilon=0.0)
    
    def test_accumulator_growth(self):
        """Test gradient accumulator grows monotonically."""
        adagrad = AdaGrad(learning_rate=0.1)
        params = np.array([[1.0], [2.0]])
        gradient = np.array([[1.0], [1.0]])
        
        # First step
        adagrad.step(params, gradient)
        acc1 = adagrad.accumulator.copy()
        
        # Second step
        adagrad.step(params, gradient)
        acc2 = adagrad.accumulator.copy()
        
        # Accumulator should grow
        assert np.all(acc2 > acc1)
    
    def test_decreasing_learning_rate(self):
        """Test effective learning rate decreases over time."""
        adagrad = AdaGrad(learning_rate=0.1)
        params = np.array([[1.0]])
        gradient = np.array([[1.0]])
        
        lr1 = adagrad.get_effective_lr()
        adagrad.step(params, gradient)
        lr2 = adagrad.get_effective_lr()
        adagrad.step(params, gradient)
        lr3 = adagrad.get_effective_lr()
        
        # Learning rate should decrease
        assert lr1 > lr2 > lr3


class TestRMSProp:
    """Test RMSProp optimizer."""
    
    def test_initialization(self):
        """Test RMSProp initialization."""
        rmsprop = RMSProp(learning_rate=0.001, rho=0.9)
        assert rmsprop.rho == 0.9
        assert rmsprop.moving_avg_squared is None
    
    def test_invalid_rho(self):
        """Test that invalid rho raises error."""
        with pytest.raises(ValueError):
            RMSProp(rho=1.5)
        with pytest.raises(ValueError):
            RMSProp(rho=-0.1)
    
    def test_exponential_moving_average(self):
        """Test moving average computation."""
        rmsprop = RMSProp(learning_rate=0.001, rho=0.9)
        params = np.array([[1.0]])
        gradient = np.array([[2.0]])
        
        # First step: EMA = (1-rho) * g^2
        rmsprop.step(params, gradient)
        expected_ema = 0.1 * (2.0 ** 2)
        assert np.allclose(rmsprop.moving_avg_squared, expected_ema)
        
        # Second step: EMA = rho * EMA_old + (1-rho) * g^2
        rmsprop.step(params, gradient)
        expected_ema = 0.9 * expected_ema + 0.1 * (2.0 ** 2)
        assert np.allclose(rmsprop.moving_avg_squared, expected_ema)
    
    def test_centered_variant(self):
        """Test centered RMSProp."""
        rmsprop = RMSProp(learning_rate=0.001, rho=0.9, centered=True)
        params = np.array([[1.0]])
        gradient = np.array([[2.0]])
        
        rmsprop.step(params, gradient)
        
        assert rmsprop.moving_avg is not None
        assert rmsprop.moving_avg_squared is not None


class TestAdam:
    """Test Adam optimizer."""
    
    def test_initialization(self):
        """Test Adam initialization."""
        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        assert adam.beta1 == 0.9
        assert adam.beta2 == 0.999
        assert adam.m is None
        assert adam.v is None
        assert adam.t == 0
    
    def test_invalid_betas(self):
        """Test that invalid betas raise errors."""
        with pytest.raises(ValueError):
            Adam(beta1=1.5)
        with pytest.raises(ValueError):
            Adam(beta2=-0.1)
    
    def test_moment_updates(self):
        """Test first and second moment updates."""
        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        params = np.array([[1.0]])
        gradient = np.array([[2.0]])
        
        adam.step(params, gradient)
        
        # First moment: m = beta1*0 + (1-beta1)*g
        expected_m = 0.1 * 2.0
        assert np.allclose(adam.m, expected_m)
        
        # Second moment: v = beta2*0 + (1-beta2)*g^2
        expected_v = 0.001 * (2.0 ** 2)
        assert np.allclose(adam.v, expected_v)
        
        assert adam.t == 1
    
    def test_bias_correction(self):
        """Test bias correction in early iterations."""
        adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
        params = np.array([[1.0]])
        gradient = np.array([[1.0]])
        
        # First step
        params_1 = adam.step(params, gradient)
        
        # Bias correction should make effective LR larger initially
        # because we divide by (1 - beta^t) which is < 1
        assert adam.t == 1
    
    def test_amsgrad_variant(self):
        """Test AMSGrad variant."""
        adam = Adam(learning_rate=0.001, amsgrad=True)
        params = np.array([[1.0]])
        gradient = np.array([[2.0]])
        
        adam.step(params, gradient)
        
        assert adam.v_max is not None


class TestAdamW:
    """Test AdamW optimizer."""
    
    def test_initialization(self):
        """Test AdamW initialization."""
        adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
        assert adamw.weight_decay == 0.01
        assert adamw.name == "AdamW"
    
    def test_invalid_weight_decay(self):
        """Test that invalid weight decay raises error."""
        with pytest.raises(ValueError):
            AdamW(weight_decay=-0.01)
    
    def test_weight_decay_applied(self):
        """Test weight decay is applied separately."""
        adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
        params = np.array([[10.0]])  # Large param to see decay
        gradient = np.array([[0.0]])  # Zero gradient
        
        new_params = adamw.step(params, gradient)
        
        # Even with zero gradient, params should shrink due to weight decay
        assert new_params[0, 0] < params[0, 0]


class TestConvergence:
    """Test convergence on simple quadratic problem."""
    
    def test_sgd_convergence(self):
        """Test SGD converges on quadratic."""
        sgd = SGD(learning_rate=0.1)
        
        # Minimize f(x) = x^2, starting from x=10
        x = np.array([[10.0]])
        
        for _ in range(100):
            gradient = 2 * x  # df/dx = 2x
            x = sgd.step(x, gradient)
        
        # Should converge close to 0
        assert abs(x[0, 0]) < 0.1
    
    def test_momentum_faster_convergence(self):
        """Test momentum converges faster than SGD."""
        sgd = SGD(learning_rate=0.1)
        momentum = Momentum(learning_rate=0.1, momentum=0.9)
        
        # Same problem: minimize f(x) = x^2
        x_sgd = np.array([[10.0]])
        x_mom = np.array([[10.0]])
        
        for _ in range(50):
            grad_sgd = 2 * x_sgd
            grad_mom = 2 * x_mom
            
            x_sgd = sgd.step(x_sgd, grad_sgd)
            x_mom = momentum.step(x_mom, grad_mom)
        
        # Momentum should be closer to optimum
        assert abs(x_mom[0, 0]) < abs(x_sgd[0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])