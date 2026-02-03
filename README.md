# Optimization Primitive Library: Gradient Descent with Provable Convergence

> **Project Type**: Mathematical Optimization Research & Theory Implementation  
> **Focus**: Convergence Theory, Numerical Stability, Algorithm Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Vision (2035 AI Agent Era)

**This is not a simple gradient descent implementation—it's a research-grade optimization library that proves mathematical guarantees and validates them experimentally.**

In the AI agent era, this project serves as:
1. **Theoretical Foundation**: Mathematically rigorous implementation of optimization primitives
2. **Educational Resource**: Teaching convergence theory through executable code
3. **Research Platform**: Extensible framework for studying optimization dynamics
4. **Benchmark Suite**: Performance and stability validation tools

---

## 🚀 What Makes This Different?

### Traditional ML Libraries (PyTorch, scikit-learn)
- ❌ Black-box optimizers
- ❌ No convergence guarantees
- ❌ Limited mathematical transparency

### This Project
- ✓ **Every algorithm has mathematical proof**
- ✓ **Convergence rates computed and validated**
- ✓ **Numerical stability analyzed**
- ✓ **Theory meets implementation**

---

## 📚 Core Features

### Phase 1: Convergence Theory (Current)

#### 1. **Lipschitz Continuity Analysis**
```python
from theory.convergence_proof import ConvergenceAnalyzer

analyzer = ConvergenceAnalyzer(X, y)
L = analyzer.compute_lipschitz_constant()  # λ_max(Hessian)
print(f"Gradient is {L}-Lipschitz continuous")
```

#### 2. **Strong Convexity Parameter**
```python
mu = analyzer.compute_strong_convexity_parameter()  # λ_min(Hessian)
print(f"Loss is {mu}-strongly convex")
```

#### 3. **Condition Number & Optimal Learning Rate**
```python
kappa = analyzer.compute_condition_number()  # L / mu
eta_opt = analyzer.compute_optimal_learning_rate()  # 2 / (L + mu)

print(f"Condition number: {kappa}")
print(f"Optimal learning rate: {eta_opt}")
print(f"Convergence rate: {(kappa-1)/(kappa+1)}")
```

#### 4. **Numerical Stability Monitoring**
```python
from theory.numerical_stability import NumericalStabilityAnalyzer

stability = NumericalStabilityAnalyzer(dtype=np.float64)
stability.monitor_gradient(grad, step)
stability.check_catastrophic_cancellation(a, b)
```

#### 5. **Enhanced Gradient Descent with Monitoring**
```python
model = GradientDescentRegressor(
    learning_rate=0.1,
    epochs=1000,
    monitor_convergence=True  # Enables theory-based monitoring
)
model.fit(X, y)

# Access analyzers
conv_analyzer = model.get_convergence_analyzer()
stab_analyzer = model.get_stability_analyzer()
```

---

## 📈 Project Structure

```
.
├── theory/                      # Theoretical analysis modules
│   ├── convergence_proof.py      # Lipschitz, convexity, condition number
│   └── numerical_stability.py    # Floating-point precision analysis
├── examples/                    # Demonstration scripts
│   └── convergence_theory_demo.py
├── tests/                       # Unit tests
│   └── test_convergence_theory.py
├── docs/                        # Mathematical documentation
│   ├── CONVERGENCE_THEORY.md     # Full derivations and proofs
│   └── NUMERICAL_STABILITY.md    # Precision analysis guide
├── gradient_descent.py          # Core optimizer (enhanced)
├── data_generator.py            # Synthetic data generation
├── visualizer.py                # Loss landscape visualization
└── main.py                      # Example pipeline
```

---

## 🧮 Mathematical Foundations

### Convergence Guarantee

For μ-strongly convex, L-smooth functions, gradient descent with optimal learning rate η* = 2/(L+μ) satisfies:

$$
\|\theta_k - \theta^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|\theta_0 - \theta^*\|^2
$$

where κ = L/μ is the condition number.

**Implemented in:** `theory/convergence_proof.py`  
**Validated in:** `examples/convergence_theory_demo.py`

### Key Quantities Computed

| **Symbol** | **Name** | **Formula** | **Code** |
|------------|----------|-------------|----------|
| L | Lipschitz constant | λ_max(H) | `compute_lipschitz_constant()` |
| μ | Strong convexity | λ_min(H) | `compute_strong_convexity_parameter()` |
| κ | Condition number | L/μ | `compute_condition_number()` |
| η* | Optimal learning rate | 2/(L+μ) | `compute_optimal_learning_rate()` |
| ρ | Convergence rate | (κ-1)/(κ+1) | `compute_convergence_rate()` |

See [CONVERGENCE_THEORY.md](docs/CONVERGENCE_THEORY.md) for full derivations.

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib scipy
```

### Quick Start

```python
import numpy as np
from gradient_descent import GradientDescentRegressor
from data_generator import LinearDataGenerator

# Generate data
data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)

# Train with convergence monitoring
model = GradientDescentRegressor(
    learning_rate=0.1,
    epochs=1000,
    monitor_convergence=True
)
model.fit(X, y)
```

### Run Convergence Theory Demos

```bash
python examples/convergence_theory_demo.py
```

This generates:
1. Eigenvalue spectrum analysis
2. Learning rate comparison plots
3. Condition number impact visualization
4. Theoretical vs empirical convergence validation

### Run Tests

```bash
pytest tests/test_convergence_theory.py -v
```

---

## 📊 Experimental Validation

### Demo 1: Optimal Learning Rate

![Convergence Comparison](docs/images/convergence_comparison.png)

**Observation:** Optimal η* = 2/(L+μ) converges fastest, as predicted by theory.

### Demo 2: Condition Number Impact

![Condition Number](docs/images/condition_number_impact.png)

**Observation:** Higher κ leads to slower convergence, validating the bound:

$$
k \geq \frac{\kappa + 1}{2} \log\left(\frac{1}{\epsilon}\right)
$$

### Demo 3: Theoretical vs Empirical Convergence Rate

![Rate Validation](docs/images/convergence_rate_validation.png)

**Observation:** Empirical decay rate matches theoretical ρ = (κ-1)/(κ+1) within 1-2%.

---

## 🔬 Numerical Stability Analysis

### Machine Epsilon Awareness

```python
from theory.numerical_stability import NumericalStabilityAnalyzer

analyzer = NumericalStabilityAnalyzer(dtype=np.float64)
info = analyzer.get_machine_epsilon_info()

print(f"Machine epsilon: {info['machine_epsilon']:.2e}")
print(f"Decimal precision: ~{info['decimal_digits']} digits")
```

### Catastrophic Cancellation Detection

```python
is_catastrophic = analyzer.check_catastrophic_cancellation(1.23456789, 1.23456700)
if is_catastrophic:
    print("⚠ Precision loss detected in subtraction")
```

See [NUMERICAL_STABILITY.md](docs/NUMERICAL_STABILITY.md) for detailed analysis.

---

## 📄 Documentation

### Mathematical Theory
- [Convergence Theory](docs/CONVERGENCE_THEORY.md) - Full proofs and derivations
- [Numerical Stability](docs/NUMERICAL_STABILITY.md) - Floating-point analysis

### API Reference
- `theory.convergence_proof.ConvergenceAnalyzer` - Theoretical analysis
- `theory.numerical_stability.NumericalStabilityAnalyzer` - Precision monitoring
- `gradient_descent.GradientDescentRegressor` - Core optimizer

---

## 🛣️ Roadmap

### ✅ Phase 1: Convergence Theory (Completed)
- [x] Lipschitz constant computation
- [x] Strong convexity parameter
- [x] Condition number analysis
- [x] Optimal learning rate derivation
- [x] Numerical stability monitoring
- [x] Comprehensive documentation
- [x] Unit tests

### 🚧 Phase 2: Optimizer Zoo (In Progress)
- [ ] Momentum (Polyak, Nesterov)
- [ ] Adaptive methods (Adam, RMSProp, AdaGrad)
- [ ] Second-order methods (Newton, BFGS, L-BFGS)
- [ ] Variance reduction (SVRG, SARAH)
- [ ] Line search (Armijo, Wolfe)

### 🗓️ Phase 3: Non-Convex Extension
- [ ] Polynomial regression
- [ ] Neural networks (2-layer)
- [ ] Saddle point analysis
- [ ] Loss landscape visualization

### 🗓️ Phase 4: High-Performance Computing
- [ ] JAX acceleration
- [ ] GPU support
- [ ] Distributed training
- [ ] Large-scale benchmarks

### 🗓️ Phase 5: Research Artifacts
- [ ] LaTeX paper draft
- [ ] Interactive web demo
- [ ] CI/CD pipeline
- [ ] Docker containerization

---

## 🎯 Design Principles

1. **Theory-First:** Every algorithm comes with mathematical proof
2. **Transparency:** All derivations visible in code and docs
3. **Reproducibility:** Fixed seeds, deterministic execution
4. **Educational:** Code as a teaching tool
5. **Research-Grade:** Publication-quality implementation

---

## 📚 References

1. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*. Springer.
2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge.
3. **Nocedal, J., & Wright, S.** (2006). *Numerical Optimization* (2nd ed.). Springer.
4. **Higham, N.J.** (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
5. **Bubeck, S.** (2015). *Convex Optimization: Algorithms and Complexity*. Foundations and Trends in ML.

---

## ⚖️ License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Research Focus:** Mathematical foundations of optimization algorithms with provable guarantees.

**Contact:** Open an issue for questions or collaboration.

---

## 🔗 Citation

If you use this code in academic work, please cite:

```bibtex
@misc{gradient-descent-theory,
  author = {Sangwoo Sin},
  title = {Optimization Primitive Library: Gradient Descent with Provable Convergence},
  year = {2025},
  url = {https://github.com/sinsangwoo/ML-Gradient-Descent-Viz}
}
```

---

*Building optimization primitives with mathematical rigor and numerical precision.*