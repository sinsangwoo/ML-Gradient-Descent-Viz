# Optimizer Guide: Theory and Practice

## Overview

This document provides a comprehensive guide to all optimizers implemented in this library, including:
- Mathematical formulations
- Theoretical convergence guarantees
- Practical hyperparameter guidance
- When to use each optimizer

---

## Quick Reference Table

| **Optimizer** | **Best For** | **Key Hyperparameters** | **Convergence Rate** | **Memory** |
|---------------|--------------|-------------------------|----------------------|------------|
| SGD | Baseline, simple problems | η | O(1/k) | O(1) |
| Momentum | Smooth landscapes, oscillation | η, β | Better constants | O(d) |
| Nesterov | Convex optimization | η, β | O(1/k²) | O(d) |
| AdaGrad | Sparse features (NLP) | η, ε | O(1/√k) | O(d) |
| RMSProp | Non-stationary, RNNs | η, β, ε | Empirical | O(d) |
| Adam | General deep learning | η, β₁, β₂, ε | Empirical | O(2d) |
| AdamW | Fine-tuning, transfer learning | η, β₁, β₂, ε, λ | Empirical | O(2d) |

*d = number of parameters*

---

## 1. Stochastic Gradient Descent (SGD)

### Mathematical Formulation

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
$$

### Theoretical Guarantees

**Convex functions:**
$$
J(\bar{\theta}_k) - J(\theta^*) \leq \frac{R^2 + G^2 \sum_{i=1}^k \alpha_i^2}{2\sum_{i=1}^k \alpha_i}
$$

where $R$ is the initial distance to optimum, $G$ is a gradient bound.

**With constant learning rate $\alpha = O(1/\sqrt{k})$:** Convergence rate is $O(1/\sqrt{k})$.

**Strongly convex:** With optimal $\alpha$, linear convergence $O(\rho^k)$ where $\rho = (1 - \mu/L)$.

### Hyperparameter Guidelines

- **Learning rate $\alpha$:** 
  - Theory: $\alpha \leq 2/L$ for stability
  - Practice: Start with 0.1, use learning rate schedule
  - Too large: Divergence
  - Too small: Slow convergence

### When to Use

- ✓ Baseline comparison
- ✓ Well-conditioned problems
- ✗ Ill-conditioned problems
- ✗ Plateaus or saddle points

---

## 2. Momentum SGD

### Mathematical Formulation

$$
\begin{align*}
v_{t+1} &= \beta v_t - \alpha \nabla J(\theta_t) \\
\theta_{t+1} &= \theta_t + v_{t+1}
\end{align*}
$$

### Physical Interpretation

Think of a ball rolling down a hill:
- **Gradient:** Current slope
- **Velocity $v$:** Accumulated momentum
- **$\beta$:** Friction coefficient

### Theoretical Guarantees

**Strongly convex + Smooth:**
With optimal $\beta$ and $\alpha$:
$$
\rho = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2
$$

Compare to vanilla SGD: $\rho_{SGD} = (\kappa-1)/(\kappa+1)$

Momentum achieves **quadratic improvement** in dependence on $\kappa$.

### Hyperparameter Guidelines

- **Learning rate $\alpha$:** Same as SGD
- **Momentum $\beta$:**
  - Standard: 0.9
  - Heavy momentum (very smooth): 0.99
  - No momentum (= SGD): 0.0

**Optimal $\beta$ (theory):**
$$
\beta^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2
$$

### When to Use

- ✓ Oscillating gradients
- ✓ Ravines (different scales in different directions)
- ✓ Noisy gradients
- ✗ Problems requiring quick direction changes

---

## 3. Nesterov Accelerated Gradient (NAG)

### Mathematical Formulation

$$
\begin{align*}
\tilde{\theta}_t &= \theta_t + \beta v_t \quad \text{(look-ahead)} \\
v_{t+1} &= \beta v_t - \alpha \nabla J(\tilde{\theta}_t) \\
\theta_{t+1} &= \theta_t + v_{t+1}
\end{align*}
$$

### Key Innovation: Look-Ahead

Nesterov computes gradient at the **predicted future position** $\tilde{\theta}_t$ instead of current position $\theta_t$.

This gives "corrective" momentum:
- If prediction overshoots, gradient pulls back
- If prediction undershoots, gradient accelerates

### Theoretical Guarantees

**Smooth convex functions:**
$$
J(\theta_k) - J(\theta^*) \leq \frac{2L\|\theta_0 - \theta^*\|^2}{(k+1)^2}
$$

**Convergence rate:** $O(1/k^2)$ vs $O(1/k)$ for vanilla GD.

This is **optimal** among first-order methods (Nesterov, 1983).

### Hyperparameter Guidelines

- **Learning rate $\alpha$:** Slightly more aggressive than SGD
- **Momentum $\beta$:** 0.9 (standard)

**For quadratic functions:**
$$
\beta^* = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
$$

### When to Use

- ✓ Convex optimization
- ✓ Smooth problems
- ✓ When you need theoretical guarantees
- ✗ Non-convex deep learning (marginal gain over Momentum)

---

## 4. AdaGrad

### Mathematical Formulation

$$
\begin{align*}
G_t &= G_{t-1} + (\nabla J_t)^2 \quad \text{(element-wise)} \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{G_t} + \epsilon} \odot \nabla J_t
\end{align*}
$$

### Key Innovation: Per-Parameter Learning Rates

Each parameter gets its own adaptive learning rate:
- Frequently updated parameters: **smaller** learning rate
- Infrequently updated parameters: **larger** learning rate

### Theoretical Guarantees

**Convex online learning:**
$$
\text{Regret} \leq \frac{\|\theta^*\|}{2\alpha} \sum_{i=1}^d \|g_{1:T,i}\|_2 + \frac{\alpha}{2} \sum_{i=1}^d \|g_{1:T,i}\|_2
$$

where $\|g_{1:T,i}\|_2 = \sqrt{\sum_{t=1}^T g_{t,i}^2}$.

**Convergence rate:** $O(1/\sqrt{T})$ but with better constants for sparse problems.

### Hyperparameter Guidelines

- **Learning rate $\alpha$:** 0.01 (can be larger than SGD since it adapts)
- **Epsilon $\epsilon$:** $10^{-8}$ (numerical stability)

### When to Use

- ✓ Sparse features (NLP, one-hot encodings)
- ✓ Different scales across features
- ✗ Dense features (learning rate decays too aggressively)
- ✗ Long training (stops learning)

---

## 5. RMSProp

### Mathematical Formulation

$$
\begin{align*}
v_t &= \beta v_{t-1} + (1-\beta)(\nabla J_t)^2 \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot \nabla J_t
\end{align*}
$$

### Key Improvement over AdaGrad

Uses **exponential moving average** instead of sum:
- Forgets old gradients (controlled by $\beta$)
- Doesn't suffer from aggressive decay
- Better for non-stationary problems

### Hyperparameter Guidelines

- **Learning rate $\alpha$:** 0.001 (smaller than AdaGrad)
- **Decay rate $\beta$:** 0.9 (standard)
- **Epsilon $\epsilon$:** $10^{-8}$

### When to Use

- ✓ Recurrent Neural Networks (historically)
- ✓ Non-stationary objectives
- ✓ Mini-batch training
- ✗ Modern deep learning (Adam is better)

---

## 6. Adam (Adaptive Moment Estimation)

### Mathematical Formulation

$$
\begin{align*}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\nabla J_t \quad \text{(1st moment)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(\nabla J_t)^2 \quad \text{(2nd moment)} \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(bias correction)} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align*}
$$

### Key Innovation: Combines Best of Both Worlds

1. **Momentum ($m_t$):** Smooth gradient direction
2. **Adaptive learning rate ($v_t$):** Per-parameter scaling
3. **Bias correction:** Accounts for zero initialization

### Theoretical Properties

**Regret bound (Kingma & Ba, 2015):**
$$
\mathbb{E}[\text{Regret}] = O(\sqrt{T})
$$

**Note:** Adam's convergence theory is **not as strong** as SGD/Momentum for convex problems, but works **exceptionally well in practice** for deep learning.

### Hyperparameter Guidelines

**Standard configuration (Kingma & Ba, 2015):**
- $\alpha = 0.001$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

**Tuning tips:**
- $\beta_1$: Higher = smoother updates (0.9 standard, 0.95-0.99 for noisy gradients)
- $\beta_2$: Almost always 0.999 (don't change unless you know why)
- $\alpha$: Most important hyperparameter, start with 0.001

### When to Use

- ✓ **Default choice for deep learning**
- ✓ Sparse gradients
- ✓ Non-stationary objectives
- ✓ Noisy gradients
- ✗ Convex optimization (use Nesterov instead)

---

## 7. AdamW (Adam with Decoupled Weight Decay)

### Mathematical Formulation

Same as Adam, but with **decoupled weight decay**:

$$
\theta_{t+1} = \theta_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)
$$

### Key Insight: L2 ≠ Weight Decay in Adam!

**Wrong (L2 regularization in Adam):**
$$
\nabla J_{\text{reg}} = \nabla J + \lambda \theta
$$
Then apply Adam update.

**Right (Decoupled weight decay):**
Apply Adam update, then:
$$
\theta \leftarrow \theta - \alpha \lambda \theta
$$

**Why it matters:**
In Adam, L2 regularization interacts with adaptive learning rates in complex ways. Decoupled weight decay has a **clear, predictable effect** on generalization.

### Hyperparameter Guidelines

- All Adam hyperparameters
- **Weight decay $\lambda$:** 0.01 (standard), 0.0001-0.1 (range)

**Fine-tuning recipe:**
1. Start with Adam hyperparameters
2. Add small weight decay: $\lambda = 0.01$
3. Increase if overfitting, decrease if underfitting

### When to Use

- ✓ **Fine-tuning pre-trained models**
- ✓ **Transfer learning**
- ✓ When regularization is important
- ✗ Problems where overfitting isn't an issue

---

## Decision Tree: Which Optimizer?

```
Start
│
├─ Is your problem convex?
│  ├─ Yes → Use Nesterov (optimal O(1/k²) convergence)
│  └─ No → Continue
│
├─ Do you have sparse features (e.g., NLP, one-hot)?
│  ├─ Yes → Use Adam or AdaGrad
│  └─ No → Continue
│
├─ Do you need strong regularization?
│  ├─ Yes → Use AdamW
│  └─ No → Continue
│
├─ Are you training an RNN?
│  ├─ Yes → Use RMSProp or Adam
│  └─ No → Continue
│
└─ Default → Use Adam (works well in most cases)
```

---

## Practical Tips

### Learning Rate Scheduling

All optimizers benefit from learning rate schedules:

1. **Step decay:** $\alpha_t = \alpha_0 * \gamma^{\lfloor t/s \rfloor}$
2. **Exponential decay:** $\alpha_t = \alpha_0 e^{-\lambda t}$
3. **Cosine annealing:** $\alpha_t = \alpha_{\min} + 0.5(\alpha_{\max} - \alpha_{\min})(1 + \cos(\pi t/T))$
4. **Warm restarts:** Periodic resets to high learning rate

### Gradient Clipping

Prevent gradient explosion:
```python
if ||grad|| > threshold:
    grad = grad * (threshold / ||grad||)
```

### Debugging Checklist

- [ ] Loss decreasing? (If not: learning rate too high or wrong optimizer)
- [ ] Gradients reasonable magnitude? (1e-3 to 1 typically)
- [ ] Parameters changing? (If not: learning rate too small or vanishing gradients)
- [ ] Adaptive methods learning rate adapting? (Check `learning_rate_history`)

---

## References

1. **Ruder, S.** (2016). "An overview of gradient descent optimization algorithms." arXiv:1609.04747.
2. **Kingma & Ba** (2015). "Adam: A Method for Stochastic Optimization." ICLR 2015.
3. **Loshchilov & Hutter** (2019). "Decoupled Weight Decay Regularization." ICLR 2019.
4. **Nesterov, Y.** (1983). "A method for solving the convex programming problem with convergence rate O(1/k²)."
5. **Duchi et al.** (2011). "Adaptive Subgradient Methods." JMLR.

---

*Last updated: Phase 2 implementation*