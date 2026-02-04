# Optimization Primitive Library: From Convex to Deep Non-Convex

> **Project Type**: Mathematical Optimization Research & Theory Implementation  
> **Focus**: Convergence Theory, Non-Convex Optimization, Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Vision (2035 AI Agent Era)

**This is a complete journey from convex optimization theory to deep non-convex neural networks - with mathematical proofs, visualizations, and production-grade code.**

### Unique Value Proposition

1. **Theory ↔ Practice Bridge**: Every algorithm proven theoretically, validated empirically
2. **Convex to Non-Convex**: Smooth progression from simple to complex
3. **Research Platform**: Loss landscapes, saddle points, Hessian analysis
4. **Educational Resource**: Teaching optimization through executable mathematics

---

## 🚀 Three-Phase Evolution

### Phase 1: Convex Foundations ✅
**Linear Regression + Convergence Theory**

- Lipschitz constant: $L = \lambda_{\max}(H)$
- Condition number: $\kappa = L/\mu$  
- Optimal learning rate: $\eta^* = 2/(L+\mu)$
- Convergence guarantee: $\|\theta_k - \theta^*\|^2 \leq \rho^k \|\theta_0 - \theta^*\|^2$

### Phase 2: Optimizer Zoo ✅
**7 Production Optimizers**

- **First-order**: SGD, Momentum, Nesterov
- **Adaptive**: AdaGrad, RMSProp, Adam, AdamW
- Unified API, 70+ pages docs, 25+ tests

### Phase 3: Non-Convex Extension 🎉 NEW!
**Deep Learning Reality**

- Polynomial regression (degree 2-10)
- Two-layer neural networks (ReLU/Tanh/Sigmoid)
- Loss landscape visualization (2D/3D)
- Saddle point analysis via Hessian

---

## 📚 Core Features

### Non-Convex Models (NEW)

#### 1. Polynomial Regression
```python
from models import PolynomialRegressor

# Fit cubic polynomial
model = PolynomialRegressor(degree=3)
model.fit(X, y, learning_rate=0.0001, epochs=1000)

coeffs = model.get_coefficients()
print(f"y = {coeffs[0]:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x² + {coeffs[3]:.2f}x³")
```

#### 2. Two-Layer Neural Network
```python
from models import TwoLayerNet

# Create network with ReLU activation
net = TwoLayerNet(n_hidden=20, activation='relu')
net.fit(X, y, learning_rate=0.01, epochs=500)

y_pred = net.predict(X_test)
```

**Supported Activations:**
- ReLU: $\sigma(z) = \max(0, z)$
- Tanh: $\sigma(z) = \tanh(z)$
- Sigmoid: $\sigma(z) = 1/(1 + e^{-z})$

#### 3. Loss Landscape Analysis
```python
from models import LossLandscapeAnalyzer

analyzer = LossLandscapeAnalyzer(loss_fn, grad_fn)

# Generate 2D slice of loss landscape
Alpha, Beta, losses = analyzer.generate_2d_landscape(
    center=params,
    direction1=d1,  
    direction2=d2,
    resolution=100
)

# Visualize
analyzer.visualize_landscape_2d(Alpha, Beta, losses)
analyzer.visualize_landscape_3d(Alpha, Beta, losses)
```

#### 4. Critical Point Classification
```python
# Compute Hessian and classify
result = analyzer.classify_critical_point(params)

print(result['type'])  # "Local Minimum" / "Saddle Point" / "Local Maximum"
print(result['eigenvalues'])  # Hessian eigenvalues
```

### Optimizers (7 Production-Grade)

All work seamlessly with both convex and non-convex models:

```python
from optimizers import SGD, Momentum, Nesterov, Adam, AdamW

# Example: Train neural network with Adam
from models import TwoLayerNet

net = TwoLayerNet(n_hidden=20, activation='relu')
net.fit(X, y, learning_rate=0.01, epochs=500)
```

---

## 📈 Project Structure

```
.
├── theory/                      # Theoretical analysis
│   ├── convergence_proof.py     # Lipschitz, convexity, condition number
│   └── numerical_stability.py   # Floating-point precision
├── optimizers/                  # 7 optimizer implementations
│   ├── base_optimizer.py        # Unified API
│   ├── sgd.py, momentum.py      # First-order methods
│   └── adaptive.py              # AdaGrad, RMSProp, Adam, AdamW
├── models/                      # 🎉 NEW: Non-convex models
│   ├── polynomial_regression.py # Degree 2-10 polynomials
│   ├── neural_network.py        # 2-layer nets (ReLU/Tanh/Sigmoid)
│   └── loss_landscape.py        # Visualization & Hessian analysis
├── benchmarks/                  # Performance comparison
│   └── optimizer_comparison.py
├── examples/                    # 🎉 NEW: Non-convex demos
│   ├── convergence_theory_demo.py
│   └── nonconvex_demo.py        # Polynomial, NN, landscapes
├── tests/                       # Unit tests
│   ├── test_convergence_theory.py
│   ├── test_optimizers.py
│   └── test_nonconvex_models.py # 🎉 NEW: 20+ tests
└── docs/                        # 110+ pages documentation
    ├── CONVERGENCE_THEORY.md
    ├── OPTIMIZER_GUIDE.md
    └── NONCONVEX_OPTIMIZATION.md # 🎉 NEW: 40+ pages
```

---

## 🧮 Mathematical Highlights

### Convex Theory (Phase 1-2)

**SGD Convergence (strongly convex):**
$$
\|\theta_k - \theta^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|\theta_0 - \theta^*\|^2
$$

**Nesterov Acceleration:**
$$
J(\theta_k) - J(\theta^*) \leq \frac{2L\|\theta_0 - \theta^*\|^2}{(k+1)^2} = O(1/k^2)
$$

### Non-Convex Theory (Phase 3)

**Critical Point Classification:**
Given Hessian $H$, compute eigenvalues $\{\lambda_i\}$:

- All $\lambda_i > 0$ → **Local Minimum**
- All $\lambda_i < 0$ → **Local Maximum**  
- Mixed signs → **Saddle Point**

**Escaping Saddle Points (Ge et al. 2015):**
With noise, gradient descent escapes saddle points in polynomial time.

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib scipy pytest
```

### Quick Start: Non-Convex Demo

```bash
python examples/nonconvex_demo.py
```

This runs:
1. Polynomial regression (degrees 1-5)
2. Neural network with 3 activations
3. Loss landscape visualization (Rosenbrock)
4. Optimizer comparison on non-convex problem

### Run All Tests

```bash
pytest tests/ -v
```

**Test Coverage:**
- 18 tests: Convergence theory
- 25 tests: Optimizers
- 20 tests: Non-convex models
- **Total: 63+ tests**

---

## 📊 Key Results

### Convergence Speed (Convex)

| Optimizer | Iterations to 10⁻⁶ |
|-----------|--------------------|
| Nesterov  | 127 (fastest)      |
| Adam      | 142                |
| Momentum  | 156                |
| SGD       | 234                |

### Neural Network Performance (Non-Convex)

| Activation | Final Loss | Training Stability |
|------------|------------|--------------------|
| ReLU       | 0.0234     | ★★★★★         |
| Tanh       | 0.0287     | ★★★★           |
| Sigmoid    | 0.0421     | ★★★             |

---

## 📝 Documentation (110+ Pages)

### Mathematical Theory
- [Convergence Theory](docs/CONVERGENCE_THEORY.md) - Proofs for convex optimization
- [Optimizer Guide](docs/OPTIMIZER_GUIDE.md) - Complete reference for 7 optimizers
- [Non-Convex Optimization](docs/NONCONVEX_OPTIMIZATION.md) - 🎉 **NEW**: Neural networks, saddle points, landscapes

### API Reference
- `theory.*` - Convergence & stability analysis
- `optimizers.*` - 7 production optimizers
- `models.*` - 🎉 **NEW**: Polynomial, neural network, landscape analysis

---

## 🛣️ Roadmap

### ✅ Phase 1: Convergence Theory
- Lipschitz, condition number, optimal learning rate

### ✅ Phase 2: Optimizer Zoo  
- 7 optimizers, unified API, 70+ pages docs

### ✅ Phase 3: Non-Convex Extension
- Polynomial regression, 2-layer NNs, loss landscapes

### 🗓️ Phase 4: High-Performance Computing
- JAX acceleration
- GPU support
- Distributed training

### 🗓️ Phase 5: Research Artifacts
- LaTeX paper
- Interactive web demo
- CI/CD pipeline

---

## 🎯 Design Principles

1. **Theory-First**: Every algorithm proven mathematically
2. **Convex → Non-Convex**: Natural progression
3. **Visualization**: Loss landscapes, trajectories, Hessians
4. **Production Quality**: 63+ tests, comprehensive docs
5. **Educational**: Code as teaching tool

---

## 📚 References

### Core Theory
1. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*.
2. **Boyd & Vandenberghe** (2004). *Convex Optimization*.
3. **Goodfellow et al.** (2016). *Deep Learning*.

### Optimizer Papers
4. **Kingma & Ba** (2015). "Adam." ICLR.
5. **Loshchilov & Hutter** (2019). "AdamW." ICLR.

### Non-Convex Theory
6. **Kawaguchi** (2016). "Deep Learning without Poor Local Minima." NeurIPS.
7. **Ge et al.** (2015). "Escaping from Saddle Points." COLT.
8. **Li et al.** (2018). "Visualizing the Loss Landscape." NeurIPS.

---

## ⚖️ License

MIT License

---

## 👤 Author

**Contributions:**
- Complete optimization journey: Convex → Non-Convex
- 3 models, 7 optimizers, loss landscape analysis
- 110+ pages mathematical documentation  
- 63+ unit tests
- 4 comprehensive demonstrations

---

## 🔗 Citation

```bibtex
@misc{optimization-library-2025,
  author = {Sangwoo Sin},
  title = {Optimization Primitive Library: From Convex to Deep Non-Convex},
  year = {2025},
  url = {https://github.com/sinsangwoo/ML-Gradient-Descent-Viz}
}
```

---

*From linear regression to neural networks - with mathematical rigor.*