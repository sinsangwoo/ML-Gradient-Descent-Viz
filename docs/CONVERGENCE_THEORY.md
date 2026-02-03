# Convergence Theory Documentation

## Overview

This document provides a comprehensive mathematical treatment of convergence theory for gradient descent on quadratic loss functions. All theoretical results are implemented in `theory/convergence_proof.py` and validated experimentally.

---

## 1. Problem Formulation

### 1.1 Objective Function

We consider the linear regression problem with Mean Squared Error (MSE) loss:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (x_i^T \theta - y_i)^2 = \frac{1}{2m} \|X\theta - y\|^2
$$

where:
- $\theta \in \mathbb{R}^n$ is the parameter vector
- $X \in \mathbb{R}^{m \times n}$ is the design matrix
- $y \in \mathbb{R}^m$ is the target vector
- $m$ is the number of samples

### 1.2 Gradient Descent Update Rule

The gradient descent algorithm updates parameters as:

$$
\theta_{k+1} = \theta_k - \eta \nabla J(\theta_k)
$$

where $\eta > 0$ is the learning rate.

---

## 2. Gradient Derivation

### 2.1 Analytic Gradient

For the quadratic loss, the gradient is:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

**Derivation:**

$$
\begin{align*}
J(\theta) &= \frac{1}{2m} (X\theta - y)^T (X\theta - y) \\
&= \frac{1}{2m} (\theta^T X^T X \theta - 2\theta^T X^T y + y^T y)
\end{align*}
$$

Taking the gradient with respect to $\theta$:

$$
\nabla J(\theta) = \frac{1}{2m} (2X^T X \theta - 2X^T y) = \frac{1}{m} X^T (X\theta - y)
$$

### 2.2 Hessian Matrix

The Hessian (second derivative) is:

$$
H = \nabla^2 J(\theta) = \frac{1}{m} X^T X
$$

**Key property:** The Hessian is **constant** (independent of $\theta$), which is characteristic of quadratic functions.

---

## 3. Convexity Analysis

### 3.1 Convexity Proof

**Theorem 1 (Convexity):** The loss function $J(\theta)$ is convex.

**Proof:**

For any $v \in \mathbb{R}^n$:

$$
v^T H v = v^T \left(\frac{1}{m} X^T X\right) v = \frac{1}{m} \|Xv\|^2 \geq 0
$$

Therefore, $H$ is **positive semi-definite** ($H \succeq 0$), which implies $J(\theta)$ is convex.

### 3.2 Strong Convexity

**Definition:** A function $J$ is $\mu$-strongly convex if:

$$
J(\theta_2) \geq J(\theta_1) + \nabla J(\theta_1)^T (\theta_2 - \theta_1) + \frac{\mu}{2} \|\theta_2 - \theta_1\|^2
$$

For quadratic functions:

$$
\mu = \lambda_{\min}(H)
$$

where $\lambda_{\min}(H)$ is the smallest eigenvalue of the Hessian.

**Interpretation:** Strong convexity provides a **lower bound** on how "curved" the function is.

---

## 4. Lipschitz Continuity

### 4.1 Definition

**Definition:** The gradient $\nabla J$ is $L$-Lipschitz continuous if:

$$
\|\nabla J(\theta_1) - \nabla J(\theta_2)\| \leq L \|\theta_1 - \theta_2\|, \quad \forall \theta_1, \theta_2
$$

### 4.2 Lipschitz Constant for Quadratic Functions

For our quadratic loss:

$$
L = \lambda_{\max}(H)
$$

where $\lambda_{\max}(H)$ is the largest eigenvalue of the Hessian.

**Proof:**

$$
\begin{align*}
\|\nabla J(\theta_1) - \nabla J(\theta_2)\| &= \|H(\theta_1 - \theta_2)\| \\
&\leq \|H\| \|\theta_1 - \theta_2\| \\
&= \lambda_{\max}(H) \|\theta_1 - \theta_2\|
\end{align*}
$$

**Interpretation:** The Lipschitz constant provides an **upper bound** on how fast the gradient can change.

---

## 5. Condition Number

### 5.1 Definition

The condition number is:

$$
\kappa = \frac{L}{\mu} = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
$$

### 5.2 Interpretation

- $\kappa = 1$: **Perfectly conditioned** (all eigenvalues equal, isotropic)
- $\kappa \ll 10$: **Well-conditioned** (fast convergence)
- $\kappa \approx 100$: **Ill-conditioned** (slow convergence)
- $\kappa \gg 1000$: **Severely ill-conditioned** (very slow convergence)

**Physical meaning:** The condition number measures the **ratio of the "easiest" direction to the "hardest" direction** in the optimization landscape.

---

## 6. Convergence Analysis

### 6.1 Linear Convergence Rate

**Theorem 2 (Convergence of Gradient Descent):** For $\mu$-strongly convex and $L$-smooth functions, gradient descent with learning rate $0 < \eta < \frac{2}{L}$ satisfies:

$$
\|\theta_k - \theta^*\|^2 \leq \rho^k \|\theta_0 - \theta^*\|^2
$$

where the convergence rate is:

$$
\rho = \max\{|1 - \eta L|, |1 - \eta \mu|\}
$$

and $\theta^*$ is the unique minimizer.

### 6.2 Optimal Learning Rate

**Theorem 3 (Optimal Learning Rate):** The learning rate that minimizes $\rho$ is:

$$
\eta^* = \frac{2}{L + \mu}
$$

which gives the optimal convergence rate:

$$
\rho^* = \frac{\kappa - 1}{\kappa + 1}
$$

**Proof sketch:**

To minimize $\rho = \max\{|1 - \eta L|, |1 - \eta \mu|\}$, we want:

$$
1 - \eta L = -(1 - \eta \mu)
$$

Solving for $\eta$:

$$
\eta = \frac{2}{L + \mu}
$$

### 6.3 Number of Iterations to $\epsilon$-Accuracy

To achieve $\|\theta_k - \theta^*\|^2 \leq \epsilon \|\theta_0 - \theta^*\|^2$:

$$
k \geq \frac{\log(1/\epsilon)}{\log(1/\rho)}
$$

For optimal $\eta^*$:

$$
k \geq \frac{\kappa + 1}{2} \log\left(\frac{1}{\epsilon}\right)
$$

**Implication:** The number of iterations scales **linearly with $\kappa$** and **logarithmically with $1/\epsilon$**.

---

## 7. Stability Analysis

### 7.1 Stability Condition

Gradient descent is **stable** if:

$$
0 < \eta < \frac{2}{L}
$$

**Proof:** For quadratic functions, the error propagates as:

$$
\theta_k - \theta^* = (I - \eta H)(\theta_{k-1} - \theta^*)
$$

Stability requires all eigenvalues of $(I - \eta H)$ to have magnitude less than 1:

$$
|1 - \eta \lambda_i| < 1, \quad \forall i
$$

The critical constraint is at $\lambda_{\max} = L$:

$$
-1 < 1 - \eta L < 1 \implies 0 < \eta < \frac{2}{L}
$$

### 7.2 Practical Recommendations

1. **Conservative choice:** $\eta = \frac{1}{L}$ (guarantees $\rho \leq 1 - \frac{\mu}{L}$)
2. **Optimal choice:** $\eta = \frac{2}{L + \mu}$ (fastest convergence)
3. **Safe range:** $\eta \in [\frac{\mu}{L^2}, \frac{2}{L}]$

---

## 8. Implementation Details

### 8.1 Computing Eigenvalues

For univariate regression ($X \in \mathbb{R}^{m \times 1}$):

$$
H = \frac{1}{m} X^T X = \frac{1}{m} \sum_{i=1}^m x_i^2
$$

This is a scalar, so:
- $\lambda_{\max} = \lambda_{\min} = \frac{1}{m} \sum_{i=1}^m x_i^2$
- $\kappa = 1$ (perfectly conditioned)

### 8.2 Numerical Considerations

1. **Avoid zero eigenvalues:** Add small regularization $\mu \leftarrow \max(\mu, \epsilon)$
2. **Use `eigvalsh` for symmetric matrices:** More stable than general `eig`
3. **Check condition number:** If $\kappa > 10^{10}$, consider preconditioning

---

## 9. Experimental Validation

See `examples/convergence_theory_demo.py` for:

1. **Eigenvalue spectrum visualization**
2. **Learning rate sensitivity analysis**
3. **Empirical vs theoretical convergence rate comparison**
4. **Condition number impact on convergence speed**

---

## 10. References

1. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*. Springer.
   - Chapter 2: Smooth Convex Optimization

2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
   - Section 9.3: Gradient Descent Method

3. **Nocedal, J., & Wright, S.** (2006). *Numerical Optimization* (2nd ed.). Springer.
   - Chapter 3: Line Search Methods

4. **Bubeck, S.** (2015). *Convex Optimization: Algorithms and Complexity*. Foundations and Trends in Machine Learning.
   - Section 3.2: Gradient Descent

---

## 11. Key Takeaways

| **Concept** | **Formula** | **Meaning** |
|-------------|-------------|-------------|
| Lipschitz constant | $L = \lambda_{\max}(H)$ | Upper bound on gradient change |
| Strong convexity | $\mu = \lambda_{\min}(H)$ | Lower bound on curvature |
| Condition number | $\kappa = L/\mu$ | Problem difficulty |
| Optimal learning rate | $\eta^* = 2/(L+\mu)$ | Fastest convergence |
| Convergence rate | $\rho = (\kappa-1)/(\kappa+1)$ | Exponential decay rate |
| Iterations to $\epsilon$ | $k \geq \frac{\kappa+1}{2}\log(1/\epsilon)$ | Complexity bound |

---

*Last updated: Phase 1 implementation*