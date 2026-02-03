# Numerical Stability Analysis

## Overview

This document covers numerical precision issues that arise in gradient descent implementations, including floating-point arithmetic limitations, catastrophic cancellation, and gradient stability.

---

## 1. Floating-Point Arithmetic Fundamentals

### 1.1 IEEE 754 Standard

Modern computers use IEEE 754 floating-point representation:

| **Type** | **Sign** | **Exponent** | **Mantissa** | **Precision** | **Range** |
|----------|----------|--------------|--------------|---------------|------------|
| FP16 (half) | 1 bit | 5 bits | 10 bits | ~3 decimal digits | $\pm 6.5 \times 10^4$ |
| FP32 (single) | 1 bit | 8 bits | 23 bits | ~7 decimal digits | $\pm 3.4 \times 10^{38}$ |
| FP64 (double) | 1 bit | 11 bits | 52 bits | ~16 decimal digits | $\pm 1.8 \times 10^{308}$ |

### 1.2 Machine Epsilon

**Definition:** Machine epsilon $\epsilon_{\text{mach}}$ is the smallest number such that:

$$
1 + \epsilon_{\text{mach}} \neq 1
$$

in floating-point arithmetic.

**Values:**
- FP16: $\epsilon \approx 9.77 \times 10^{-4}$
- FP32: $\epsilon \approx 1.19 \times 10^{-7}$
- FP64: $\epsilon \approx 2.22 \times 10^{-16}$

**Implication:** Any computation result smaller than $\epsilon_{\text{mach}}$ relative to 1.0 **cannot be represented accurately**.

---

## 2. Sources of Numerical Error

### 2.1 Roundoff Error

**Example:** Computing $x = 1.0 + 10^{-20}$ in FP64:

```python
x = 1.0 + 1e-20  # Result: 1.0 (information lost!)
```

**In gradient descent:** Tiny parameter updates may be lost:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J
$$

If $\eta \|\nabla J\| < \epsilon_{\text{mach}} \|\theta_{\text{old}}\|$, the update has **no effect**.

### 2.2 Catastrophic Cancellation

**Definition:** Loss of precision when subtracting two nearly equal numbers.

**Example:**

$$
a = 1.23456789, \quad b = 1.23456700
$$

$$
a - b = 0.00000089
$$

Only 2 significant digits remain (lost 6 digits!).

**In gradient descent:** Computing residuals $y_{\text{pred}} - y_{\text{true}}$ when predictions are very accurate can lose precision.

### 2.3 Gradient Explosion/Vanishing

**Gradient Explosion:** $\|\nabla J\| \to \infty$
- Causes: Large eigenvalues, unstable learning rate
- Effect: Parameters diverge, NaN/Inf values

**Gradient Vanishing:** $\|\nabla J\| \to 0$
- Causes: Small eigenvalues, over-smoothing
- Effect: Premature convergence, stagnation

---

## 3. Condition Number Impact

### 3.1 Precision Loss Estimate

**Theorem:** For a system with condition number $\kappa$, the relative error in the solution is approximately:

$$
\frac{\|\theta_{\text{computed}} - \theta_{\text{exact}}\|}{\|\theta_{\text{exact}}\|} \approx \kappa \cdot \epsilon_{\text{mach}}
$$

**Example:**

| $\kappa$ | FP32 error bound | FP64 error bound |
|----------|------------------|------------------|
| $10^2$ | $10^{-5}$ | $10^{-14}$ |
| $10^6$ | $10^{-1}$ (10%!) | $10^{-10}$ |
| $10^{10}$ | Unreliable | $10^{-6}$ |

**Implication:** Ill-conditioned problems ($\kappa \gg 1$) require **higher precision** or **preconditioning**.

### 3.2 Digits Lost

Approximate number of decimal digits lost:

$$
\text{Digits lost} \approx \log_{10}(\kappa)
$$

**Example:** $\kappa = 10^6$ loses ~6 decimal digits.

---

## 4. Stability Monitoring

### 4.1 Gradient Norm Tracking

Monitor $\|\nabla J\|$ at each iteration:

```python
if grad_norm > 1e10:
    warnings.warn("Gradient explosion!")
if grad_norm < 1e-10:
    warnings.warn("Gradient vanishing!")
```

### 4.2 Loss Monotonicity

For convex problems, loss should **never increase** (except due to numerical errors):

```python
if loss[k] > loss[k-1]:
    warnings.warn("Loss increased - possible numerical instability")
```

### 4.3 Parameter Update Size

Check if updates are too small:

$$
\frac{\|\theta_k - \theta_{k-1}\|}{\|\theta_{k-1}\|} < \epsilon_{\text{mach}}
$$

This indicates **stagnation**.

---

## 5. Mitigation Strategies

### 5.1 Use Higher Precision

```python
# Force FP64 instead of FP32
X = X.astype(np.float64)
y = y.astype(np.float64)
```

**Trade-off:** 2x memory, ~1.5-2x slower computation.

### 5.2 Normalization/Standardization

**Feature normalization:**

$$
X_{\text{norm}} = \frac{X - \mu}{\sigma}
$$

This **improves condition number** by making features have similar scales.

### 5.3 Regularization

Add small regularization to avoid singular Hessian:

$$
J_{\text{reg}}(\theta) = J(\theta) + \frac{\lambda}{2} \|\theta\|^2
$$

This ensures $\mu \geq \lambda > 0$.

### 5.4 Kahan Summation

For summing many small numbers (e.g., in batch gradient):

```python
def kahan_sum(arr):
    s = 0.0
    c = 0.0  # Compensation for lost low-order bits
    for x in arr:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s
```

Reduces accumulation error from $O(n\epsilon)$ to $O(\epsilon)$.

### 5.5 Gradient Clipping

Prevent explosion:

```python
grad_norm = np.linalg.norm(grad)
if grad_norm > threshold:
    grad = grad * (threshold / grad_norm)
```

---

## 6. Precision Comparison

### 6.1 Experimental Setup

Compare FP16, FP32, FP64 on the same problem:

```python
for dtype in [np.float16, np.float32, np.float64]:
    X_cast = X.astype(dtype)
    y_cast = y.astype(dtype)
    # Train and measure final loss
```

### 6.2 Expected Results

| **Precision** | **Typical Final Loss** | **Convergence** |
|---------------|------------------------|------------------|
| FP16 | $10^{-3}$ to $10^{-2}$ | Often unstable |
| FP32 | $10^{-7}$ to $10^{-6}$ | Stable for most problems |
| FP64 | $10^{-15}$ to $10^{-14}$ | Very stable |

**Note:** FP16 is only viable with **mixed-precision training** and **loss scaling**.

---

## 7. Case Studies

### 7.1 Well-Conditioned Problem ($\kappa \approx 1$)

```python
X = np.random.randn(100, 1)
y = 2*X + 5 + 0.1*np.random.randn(100, 1)
# κ ≈ 1, all precisions work fine
```

**Result:** FP32 and FP64 converge to similar accuracy.

### 7.2 Ill-Conditioned Problem ($\kappa \gg 1$)

```python
X = np.concatenate([X, X + 1e-6*np.random.randn(100,1)], axis=1)
# κ >> 1, highly correlated features
```

**Result:** FP32 may fail, FP64 required.

---

## 8. Best Practices

### 8.1 Before Training

1. **Check condition number:**
   ```python
   kappa = np.linalg.cond(X.T @ X)
   if kappa > 1e10:
       warnings.warn("Ill-conditioned problem!")
   ```

2. **Normalize features:**
   ```python
   X = (X - X.mean(axis=0)) / X.std(axis=0)
   ```

3. **Choose appropriate dtype:** Use FP64 if $\kappa > 10^6$.

### 8.2 During Training

1. **Monitor gradient norms** (detect explosion/vanishing)
2. **Check loss monotonicity** (detect instability)
3. **Verify parameter updates** (detect stagnation)

### 8.3 After Training

1. **Compare with closed-form solution** (if available)
2. **Check residual magnitude:** $\|\nabla J(\theta^*)\|$ should be $< 10^{-6}$
3. **Validate on held-out data**

---

## 9. Implementation

See `theory/numerical_stability.py` for:

- `NumericalStabilityAnalyzer` class
- Machine epsilon detection
- Catastrophic cancellation checker
- Gradient/loss/parameter monitoring
- Precision comparison tools

---

## 10. References

1. **Higham, N.J.** (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

2. **Goldberg, D.** (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*, 23(1), 5-48.

3. **Micikevicius, P., et al.** (2017). "Mixed Precision Training." *ICLR 2018*.

---

## 11. Quick Reference

| **Issue** | **Detection** | **Solution** |
|-----------|---------------|-------------|
| Gradient explosion | $\|\nabla J\| > 10^{10}$ | Reduce learning rate, clip gradients |
| Gradient vanishing | $\|\nabla J\| < 10^{-10}$ | Increase learning rate, check data scaling |
| Loss stagnation | No change in loss for 100+ iterations | Check for numerical underflow |
| Ill-conditioning | $\kappa > 10^6$ | Normalize features, add regularization |
| Catastrophic cancellation | $(a-b)/a < 10^{-8}$ | Use higher precision, reformulate |

---

*Last updated: Phase 1 implementation*