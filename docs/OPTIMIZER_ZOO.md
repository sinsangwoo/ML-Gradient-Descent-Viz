# Optimizer Zoo Documentation

## Overview

This document provides a comprehensive guide to all implemented optimization algorithms, their theoretical foundations, and practical usage guidelines.

---

## Table of Contents

1. [Unified Interface](#unified-interface)
2. [First-Order Methods](#first-order-methods)
3. [Algorithm Comparison](#algorithm-comparison)
4. [Usage Guidelines](#usage-guidelines)
5. [References](#references)

---

## Unified Interface

All optimizers inherit from `BaseOptimizer` and implement:

```python
class BaseOptimizer(ABC):
    @abstractmethod
    def step(self, params: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Perform one optimization step."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict:
        """Get optimizer configuration."""
        pass
```

### Common Methods

- `step(params, gradient)` - Update parameters
- `reset()` - Reset internal state (momentum, accumulators)
- `track_step(params, gradient, loss)` - Log optimization progress
- `get_effective_lr()` - Get current effective learning rate
- `get_statistics()` - Get training statistics

---

## First-Order Methods

### 1. SGD - Stochastic Gradient Descent

**Update Rule:**
$$
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)
$$

**Characteristics:**
- **Simplest** optimizer
- **Deterministic** for batch GD
- **Noisy** for mini-batch/online SGD

**Pros:**
- Easy to understand and implement
- Low memory footprint
- Proven convergence guarantees

**Cons:**
- Slow on ill-conditioned problems
- Sensitive to learning rate
- Can oscillate in steep valleys

**When to Use:**
- Convex optimization
- Baseline comparison
- Limited memory scenarios

**Code Example:**
```python
from optimizers import SGD

optimizer = SGD(learning_rate=0.1, batch_size=32)
params = optimizer.step(params, gradient)
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default |
|-----------|---------------|----------|
| learning_rate | [1e-4, 1e-1] | 0.01 |
| batch_size | [1, 256] | None (full batch) |

---

### 2. Momentum - Heavy Ball Method

**Update Rule:**
$$
\begin{align*}
v_{t+1} &= \beta v_t + \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align*}
$$

**Characteristics:**
- **Accelerates** in consistent directions
- **Dampens** oscillations
- **Smooths** noisy gradients

**Intuition:**
- Like a ball rolling down a hill
- Builds up "velocity" in steep directions
- Reduces zigzagging

**Pros:**
- Faster convergence than SGD
- Better conditioning
- Escapes shallow local minima

**Cons:**
- Extra memory for velocity
- Can overshoot minima
- One more hyperparameter

**When to Use:**
- Neural networks
- When gradients are noisy
- High-curvature problems

**Code Example:**
```python
from optimizers import Momentum

optimizer = Momentum(learning_rate=0.1, momentum=0.9)
params = optimizer.step(params, gradient)
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| learning_rate | [1e-4, 1e-1] | 0.01 | Often higher than SGD |
| momentum | [0.5, 0.99] | 0.9 | Higher = more smoothing |

---

### 3. Nesterov Accelerated Gradient

**Update Rule:**
$$
\begin{align*}
v_{t+1} &= \beta v_t + \nabla J(\theta_t - \beta v_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align*}
$$

**Key Difference from Momentum:**
- Gradient evaluated at **look-ahead point**: $\theta - \beta v$
- "First jump, then correct" vs "correct, then jump"

**Characteristics:**
- **Optimal** convergence rate for smooth convex: $O(1/k^2)$
- More **responsive** to gradient changes
- Better for **ill-conditioned** problems

**Theoretical Advantage:**

For $\mu$-strongly convex, $L$-smooth functions:
- Standard momentum: $O(1/k)$ rate
- Nesterov: $O(1/k^2)$ rate

**Pros:**
- Provably faster than standard momentum
- Better anticipation of future gradients
- Same memory as momentum

**Cons:**
- Slightly more complex implementation
- Requires look-ahead gradient computation

**When to Use:**
- Convex optimization
- When convergence speed matters
- Research/benchmarking

**Code Example:**
```python
from optimizers import NesterovMomentum

optimizer = NesterovMomentum(learning_rate=0.1, momentum=0.9)

# Compute gradient at look-ahead point
lookahead_params = optimizer.get_lookahead_params(params)
gradient = compute_gradient(lookahead_params)  # User function

params = optimizer.step(params, gradient)
```

---

### 4. AdaGrad - Adaptive Gradient

**Update Rule:**
$$
\begin{align*}
G_t &= G_{t-1} + g_t \odot g_t \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{align*}
$$

**Characteristics:**
- **Per-parameter** learning rates
- **Accumulates** all past gradients
- **Monotonically decreasing** learning rate

**Intuition:**
- Parameters with large gradients → smaller effective LR
- Parameters with small gradients → larger effective LR
- Automatically balances learning across dimensions

**Pros:**
- No manual LR tuning per parameter
- Excellent for **sparse gradients** (NLP, recommender systems)
- Robust to feature scaling

**Cons:**
- Learning rate decays too aggressively
- Can stop learning prematurely
- Not suitable for deep learning (non-convex)

**When to Use:**
- Sparse features (text, categorical)
- Convex problems
- When feature scales vary widely

**Code Example:**
```python
from optimizers import AdaGrad

optimizer = AdaGrad(learning_rate=0.1, epsilon=1e-8)
params = optimizer.step(params, gradient)

print(f"Effective LR: {optimizer.get_effective_lr():.6f}")
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| learning_rate | [0.01, 1.0] | 0.01 | Can be larger than SGD |
| epsilon | [1e-10, 1e-6] | 1e-8 | Numerical stability |

---

### 5. RMSProp - Root Mean Square Propagation

**Update Rule:**
$$
\begin{align*}
E[g^2]_t &= \rho E[g^2]_{t-1} + (1-\rho) g_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t
\end{align*}
$$

**Characteristics:**
- **Exponential moving average** of squared gradients
- **Non-monotonic** learning rate (can increase!)
- **Decay factor** controls memory

**Key Improvement over AdaGrad:**
- Uses moving average instead of cumulative sum
- Allows learning rate to recover
- Better for non-convex problems

**Pros:**
- Suitable for deep learning
- Adaptive per-parameter LR
- Works well on RNNs

**Cons:**
- Still no momentum component
- Can be unstable without proper tuning

**When to Use:**
- Neural networks
- Recurrent networks (original use case)
- Non-convex optimization

**Code Example:**
```python
from optimizers import RMSProp

# Standard RMSProp
optimizer = RMSProp(learning_rate=0.001, rho=0.9)

# Centered variant (uses variance)
optimizer = RMSProp(learning_rate=0.001, rho=0.9, centered=True)

params = optimizer.step(params, gradient)
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| learning_rate | [1e-4, 1e-2] | 0.001 | Often 0.001 |
| rho | [0.9, 0.999] | 0.9 | 0.9 for most tasks |
| epsilon | [1e-10, 1e-6] | 1e-8 | Stability constant |

---

### 6. Adam - Adaptive Moment Estimation

**Update Rule:**
$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(bias correction)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

**Characteristics:**
- **Combines** Momentum + RMSProp
- **First moment**: running average of gradient (momentum)
- **Second moment**: running average of squared gradient (adaptive LR)
- **Bias correction**: accounts for initialization at zero

**Intuition:**
- Gets benefits of both momentum and adaptive learning rates
- "Best of both worlds" optimizer

**Pros:**
- **Default choice** for deep learning
- Robust across many tasks
- Minimal hyperparameter tuning needed
- Works out-of-the-box most of the time

**Cons:**
- Can generalize worse than SGD+Momentum (sharp minima)
- More memory overhead
- Not always best for every problem

**When to Use:**
- Default optimizer for neural networks
- When you want "set and forget" optimization
- When you don't know what to use

**Code Example:**
```python
from optimizers import Adam

# Standard Adam (most common)
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

# AMSGrad variant (more conservative)
optimizer = Adam(learning_rate=0.001, amsgrad=True)

params = optimizer.step(params, gradient)
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| learning_rate | [1e-4, 1e-2] | 0.001 | Often 0.001 or 0.0001 |
| beta1 | [0.9, 0.99] | 0.9 | First moment decay |
| beta2 | [0.99, 0.9999] | 0.999 | Second moment decay |
| epsilon | [1e-8, 1e-6] | 1e-8 | Rarely needs tuning |

---

### 7. AdamW - Adam with Decoupled Weight Decay

**Update Rule:**
$$
\begin{align*}
\text{[Same moment updates as Adam]} \\
\theta_{t+1} &= (1 - \lambda \eta) \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

**Key Difference from Adam:**
- Weight decay applied **directly to parameters**
- Not through gradients (decoupled from adaptive LR)

**Why Decoupling Matters:**

In standard Adam with L2 regularization:
$$
\nabla (J + \frac{\lambda}{2}\|\theta\|^2) = \nabla J + \lambda \theta
$$

Problem: Adaptive LR scales the regularization term!

AdamW solution: Apply weight decay separately.

**Pros:**
- Better generalization than Adam
- Proper L2 regularization
- Default for transformers (BERT, GPT)

**Cons:**
- One more hyperparameter (weight_decay)

**When to Use:**
- Training transformers
- When regularization matters
- When Adam overfits

**Code Example:**
```python
from optimizers import AdamW

optimizer = AdamW(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01  # 1% decay
)

params = optimizer.step(params, gradient)
```

**Key Hyperparameters:**
| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| weight_decay | [0.0, 0.1] | 0.01 | 0.01 for transformers |

---

## Algorithm Comparison

### Convergence Rate Summary

| Optimizer | Convex Smooth | Strongly Convex | Memory | Hyperparams |
|-----------|---------------|-----------------|--------|-------------|
| SGD | O(1/k) | O(exp(-k/κ)) | O(1) | 1 (η) |
| Momentum | O(1/k) | O(exp(-k/√κ)) | O(n) | 2 (η, β) |
| Nesterov | O(1/k²) | O(exp(-k/√κ)) | O(n) | 2 (η, β) |
| AdaGrad | O(1/√k) | O(log(k)/k) | O(n) | 2 (η, ε) |
| RMSProp | - | - | O(n) | 3 (η, ρ, ε) |
| Adam | - | - | O(n) | 4 (η, β₁, β₂, ε) |
| AdamW | - | - | O(n) | 5 (η, β₁, β₂, ε, λ) |

Where:
- k: iteration number
- κ: condition number
- n: parameter dimension

### Performance Characteristics

```
Convergence Speed (Typical):
Nesterov > Momentum > Adam ≈ AdamW > RMSProp > SGD > AdaGrad

Robustness to LR:
Adam ≈ AdamW > RMSProp > AdaGrad > Nesterov ≈ Momentum > SGD

Generalization (Deep Learning):
SGD+Momentum ≥ AdamW > Adam > others

Memory Efficiency:
SGD > Momentum ≈ Nesterov > AdaGrad ≈ RMSProp ≈ Adam ≈ AdamW
```

---

## Usage Guidelines

### Quick Selection Guide

**Choose SGD when:**
- Problem is convex
- You need provable guarantees
- Memory is limited
- You have time to tune LR

**Choose Momentum when:**
- Gradients are noisy
- Problem has high curvature
- You want faster convergence than SGD

**Choose Nesterov when:**
- You want optimal convergence rate
- Problem is convex
- Research/benchmarking

**Choose AdaGrad when:**
- Features are sparse (NLP)
- Different features have different scales
- Problem is convex

**Choose RMSProp when:**
- Training RNNs
- You want adaptive LR but not full Adam

**Choose Adam when:**
- Default choice for deep learning
- You want "just works" optimizer
- Limited time for hyperparameter tuning

**Choose AdamW when:**
- Training transformers
- Regularization is important
- Adam is overfitting

### Learning Rate Guidelines

| Optimizer | Typical LR | LR Search Range |
|-----------|------------|------------------|
| SGD | 0.01-0.1 | [1e-4, 1] |
| Momentum | 0.01-0.1 | [1e-4, 1] |
| Nesterov | 0.01-0.1 | [1e-4, 1] |
| AdaGrad | 0.01-1.0 | [1e-3, 10] |
| RMSProp | 0.001 | [1e-5, 1e-2] |
| Adam | 0.001 or 0.0001 | [1e-5, 1e-2] |
| AdamW | 0.001 | [1e-5, 1e-2] |

---

## References

### Classic Papers

1. **SGD**
   - Robbins & Monro (1951). A Stochastic Approximation Method

2. **Momentum**
   - Polyak (1964). Some methods of speeding up the convergence of iteration methods

3. **Nesterov**
   - Nesterov (1983). A method for solving the convex programming problem with convergence rate O(1/k²)

4. **AdaGrad**
   - Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization

5. **RMSProp**
   - Tieleman & Hinton (2012). Lecture 6.5 - RMSProp, COURSERA

6. **Adam**
   - Kingma & Ba (2015). Adam: A Method for Stochastic Optimization, ICLR

7. **AdamW**
   - Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization, ICLR

### Survey Papers

- Ruder (2016). An overview of gradient descent optimization algorithms
- Bottou et al. (2018). Optimization Methods for Large-Scale Machine Learning

---

*Last updated: Phase 2 implementation*