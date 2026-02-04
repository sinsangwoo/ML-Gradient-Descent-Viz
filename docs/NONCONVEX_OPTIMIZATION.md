# Non-Convex Optimization Guide

## Introduction

This document explains non-convex optimization - the reality of deep learning and most practical machine learning problems.

### Why Non-Convex Matters

**Convex (Phase 1-2):**
- Linear regression: One global minimum
- Guaranteed convergence
- Simple theory

**Non-Convex (Phase 3):**
- Neural networks: Many local minima
- Saddle points everywhere  
- Complex loss landscapes
- **Mirrors reality**

---

## Models Implemented

### 1. Polynomial Regression

**Model:**
$$
y = w_0 + w_1 x + w_2 x^2 + \cdots + w_d x^d
$$

**Why Non-Convex:**
For degree $d \geq 2$, loss function is non-convex in parameters.

**Example: Cubic Polynomial**
$$
y = w_0 + w_1 x + w_2 x^2 + w_3 x^3
$$

Loss landscape has:
- Multiple local minima
- Saddle points
- Flat regions

**When to Use:**
- Controlled non-convex testbed
- Understanding polynomial fitting
- Degree tuning experiments

### 2. Two-Layer Neural Network

**Architecture:**
```
Input (d) -> Hidden (h) -> Output (1)
```

**Forward Pass:**
$$
\begin{align*}
z^{[1]} &= W^{[1]} x + b^{[1]} \\
a^{[1]} &= \sigma(z^{[1]}) \\
z^{[2]} &= W^{[2]} a^{[1]} + b^{[2]} \\
\hat{y} &= z^{[2]}
\end{align*}
$$

**Activations Supported:**

1. **ReLU:** $\sigma(z) = \max(0, z)$
   - Pros: No vanishing gradient, fast training
   - Cons: Dying ReLU problem
   - Use: Default choice

2. **Tanh:** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
   - Pros: Zero-centered, smooth
   - Cons: Vanishing gradient for large |z|
   - Use: When outputs should be bounded

3. **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
   - Pros: Output in (0,1)
   - Cons: Strong vanishing gradient
   - Use: Binary classification

**Backpropagation:**

Output layer:
$$
\frac{\partial L}{\partial W^{[2]}} = \frac{1}{m} (\hat{y} - y) (a^{[1]})^T
$$

Hidden layer:
$$
\frac{\partial L}{\partial W^{[1]}} = \frac{1}{m} \left[ (W^{[2]})^T (\hat{y} - y) \odot \sigma'(z^{[1]}) \right] x^T
$$

**Initialization Strategies:**

- **Xavier (Tanh/Sigmoid):** $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$
- **He (ReLU):** $W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$

---

## Loss Landscape Analysis

### Visualizing Non-Convexity

**Method 1: 2D Slice**

Project loss onto 2D plane:
$$
\theta(\alpha, \beta) = \theta_0 + \alpha d_1 + \beta d_2
$$

where $d_1, d_2$ are orthogonal directions.

**Method 2: 3D Surface**

Generate grid and compute loss at each point:
```python
for alpha in grid:
    for beta in grid:
        theta = center + alpha*d1 + beta*d2
        loss[alpha, beta] = compute_loss(theta)
```

### Critical Point Classification

**Using Hessian:**
$$
H_{ij} = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}
$$

**Classification Rules:**

1. **Local Minimum:** All eigenvalues > 0
2. **Local Maximum:** All eigenvalues < 0
3. **Saddle Point:** Mixed signs

**Example:**
```python
eigenvalues = [2.5, -0.3, 1.1, -1.2]
# 2 positive, 2 negative -> Saddle point
```

### Escaping Saddle Points

**Problem:**
Gradient descent can get stuck at saddle points.

**Solutions:**

1. **Momentum:** Accumulates velocity to escape
2. **Noise:** SGD randomness helps escape
3. **Negative Curvature:** Follow directions with $\lambda < 0$

**Theorem (Ge et al. 2015):**
With noise, gradient descent escapes saddle points in polynomial time.

---

## Practical Guidelines

### Training Non-Convex Models

**1. Initialization Matters**
- Use He/Xavier initialization
- Never initialize all weights to zero
- Multiple random restarts

**2. Learning Rate Tuning**
- Start large, decay over time
- Use adaptive methods (Adam) for safety
- Monitor loss landscape curvature

**3. Regularization**
- Weight decay prevents overfitting
- Dropout for neural networks
- Early stopping

**4. Diagnostic Tools**
- Plot loss landscape
- Monitor gradient norms
- Check Hessian eigenvalues

### Common Pitfalls

**Vanishing Gradients:**
- **Symptom:** Gradients ≈ 0, no learning
- **Cause:** Deep networks, sigmoid/tanh
- **Solution:** Use ReLU, skip connections

**Exploding Gradients:**
- **Symptom:** Loss/gradients → ∞
- **Cause:** Poor initialization, high LR
- **Solution:** Gradient clipping, lower LR

**Stuck at Saddle Point:**
- **Symptom:** Gradient ≈ 0 but not at minimum
- **Cause:** Hessian has negative eigenvalues
- **Solution:** Add noise, use momentum

---

## Advanced Topics

### Loss Landscape Geometry

**Linear Mode Connectivity:**
Two converged solutions often connected by low-loss path.

**Sharp vs Flat Minima:**
- **Sharp:** High curvature, poor generalization
- **Flat:** Low curvature, better generalization

**Mode Collapse:**
Different initializations find different modes.

### Theoretical Results

**Kawaguchi (2016):**
Every local minimum of a neural network is globally optimal under certain conditions.

**Dauphin et al. (2014):**
Saddle points, not local minima, are the main obstacle in high dimensions.

---

## Code Examples

### Polynomial Regression

```python
from models import PolynomialRegressor

# Fit cubic polynomial
model = PolynomialRegressor(degree=3)
model.fit(X, y, learning_rate=0.0001, epochs=1000)

coeffs = model.get_coefficients()
print(f"Coefficients: {coeffs}")
```

### Neural Network

```python
from models import TwoLayerNet

# Create network
net = TwoLayerNet(n_hidden=20, activation='relu')
net.fit(X, y, learning_rate=0.01, epochs=500)

y_pred = net.predict(X_test)
```

### Loss Landscape

```python
from models import LossLandscapeAnalyzer

# Define loss function
def loss_fn(params):
    return compute_loss(params, X, y)

# Analyze
analyzer = LossLandscapeAnalyzer(loss_fn)
Alpha, Beta, losses = analyzer.generate_2d_landscape(
    center=params, 
    direction1=d1,
    direction2=d2
)

analyzer.visualize_landscape_2d(Alpha, Beta, losses)
```

---

## References

1. **Goodfellow et al.** (2016). *Deep Learning*. MIT Press.
2. **Kawaguchi, K.** (2016). "Deep Learning without Poor Local Minima." NeurIPS.
3. **Dauphin et al.** (2014). "Identifying and attacking saddle points." NeurIPS.
4. **Ge et al.** (2015). "Escaping from saddle points." COLT.
5. **Li et al.** (2018). "Visualizing the Loss Landscape of Neural Nets." NeurIPS.
6. **Glorot & Bengio** (2010). "Understanding difficulty of training deep feedforward neural networks."
7. **He et al.** (2015). "Delving Deep into Rectifiers." ICCV.

---

*Understanding non-convexity is essential for modern deep learning.*