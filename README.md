# Gradient Descent Optimization Dynamics (Linear Regression)

> **Project Type**: Machine Learning Theory & Optimization
> **Focus**: Linear Regression, Gradient Descent, Convex Optimization
> **Note**: This README is intentionally written using **pure GitHub Markdown + `$$ ... $$` math blocks only** to avoid any Unicode / hidden-format issues when viewed in a web browser.

---

## 1. Overview

This repository implements **linear regression trained with gradient descent from scratch**, without relying on high-level ML libraries such as scikit-learn or PyTorch.

The goal of the project is **not prediction accuracy**, but a **clear understanding of optimization dynamics**, including:

* how gradients are derived mathematically
* why gradient descent converges for linear regression
* how learning rate and initialization affect convergence

This project serves as a **baseline research artifact** for further studies in numerical optimization and non-convex learning problems.

---

## 2. Problem Definition

We are given a dataset consisting of `m` samples:

$$
\mathcal{D} = { (x^{(i)}, y^{(i)}) }_{i=1}^{m}
$$

For simplicity, this project focuses on **univariate linear regression**.

The hypothesis function is defined as:

$$
h_\theta(x) = wx + b
$$

where:

* `w` is the weight (slope)
* `b` is the bias (intercept)

---

## 3. Objective Function (Mean Squared Error)

To train the model, we minimize the **Mean Squared Error (MSE)** loss:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$

This objective function is:

* quadratic in parameters
* convex
* guaranteed to have a **single global minimum**

---

## 4. Gradient Descent

Gradient Descent is a first-order iterative optimization algorithm that updates parameters in the direction of the **negative gradient**.

The update rule is:

$$
\theta_{t+1} = \theta_t - \eta , \nabla J(\theta_t)
$$

where `η` is the learning rate.

---

## 5. Gradient Derivation

### 5.1 Gradient with respect to `w`

$$
\frac{\partial J}{\partial w}
= \frac{2}{m} \sum_{i=1}^{m}
\left( h_\theta(x^{(i)}) - y^{(i)} \right)
x^{(i)}
$$

### 5.2 Gradient with respect to `b`

$$
\frac{\partial J}{\partial b}
= \frac{2}{m} \sum_{i=1}^{m}
\left( h_\theta(x^{(i)}) - y^{(i)} \right)
$$

These gradients are directly implemented in the optimization engine.

---

## 6. Vectorized Formulation

Let:

* `X` be the input vector of shape `(m, 1)`
* `y` be the target vector

Predictions:

$$
\hat{y} = Xw + b
$$

Error vector:

$$
e = \hat{y} - y
$$

Gradients in vectorized form:

$$
\nabla_w J = \frac{2}{m} X^T (Xw + b - y)
$$

$$
\nabla_b J = \frac{2}{m} \sum e
$$

Vectorization removes explicit loops and significantly improves computational efficiency.

---

## 7. Convexity Guarantee

The loss function can be written as:

$$
J(\theta) = \frac{1}{m} | X\theta - y |^2
$$

The Hessian matrix is:

$$
H = \nabla^2 J(\theta) = \frac{2}{m} X^T X
$$

For any non-zero vector `v`:

$$
v^T H v = \frac{2}{m} | Xv |^2 \ge 0
$$

Therefore:

* the Hessian is positive semi-definite
* the loss function is convex
* gradient descent converges to the global minimum (with a proper learning rate)

---

## 8. Project Structure

```text
.
├── main.py                # Experiment orchestration
├── data_generator.py      # Synthetic data generation
├── gradient_descent.py    # Optimization engine
├── visualizer.py          # Loss surface and trajectory visualization
└── README.md
```

---

## 9. Key Design Decisions

* No high-level ML libraries are used
* All gradients are derived and implemented manually
* Training history (loss, parameters) is fully logged
* Visualization is treated as an analysis tool, not decoration

---

## 10. Limitations and Future Work

Current implementation uses **batch gradient descent**, which scales poorly for large datasets.

Future extensions include:

* Stochastic / Mini-batch Gradient Descent
* Momentum-based optimizers
* Adaptive learning rate methods (Adam, RMSProp)
* Non-convex loss landscapes (polynomial regression, neural networks)

---

## 11. References

* Boyd, S. & Vandenberghe, L. *Convex Optimization*
* Goodfellow, Bengio, Courville. *Deep Learning*
* Nocedal & Wright. *Numerical Optimization*

---

## Note on Mathematical Derivations

Full step-by-step derivations and experimental analysis are intentionally **kept outside the README**.

If you are interested in the complete mathematical development, see:

```
docs/math_derivation.md
```

This separation is deliberate to keep the README readable while preserving research-level rigor.
