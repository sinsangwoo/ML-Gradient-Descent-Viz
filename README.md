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

### Phase 1: Convergence Theory ✅

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

### Phase 2: Optimizer Zoo ✅

#### Complete Implementation of 7 Optimizers

```python
from optimizers import SGD, Momentum, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW

# First-order methods
sgd = SGD(learning_rate=0.1)
momentum = Momentum(learning_rate=0.1, momentum=0.9)
nesterov = NesterovMomentum(learning_rate=0.1, momentum=0.9)

# Adaptive methods
adagrad = AdaGrad(learning_rate=0.1)
rmsprop = RMSProp(learning_rate=0.001, rho=0.9)
adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
adamw = AdamW(learning_rate=0.001, weight_decay=0.01)

# Unified interface
params = optimizer.step(params, gradient)
```

#### Comprehensive Benchmarking

```python
from benchmarks.optimizer_comparison import OptimizerBenchmark

benchmark = OptimizerBenchmark(X, y, true_params={'W': 2.0, 'b': 5.0})
results = benchmark.compare_all_optimizers(learning_rate=0.1, epochs=500)

# Generates comparison plots and statistics
```

**Features:**
- ✓ Unified `BaseOptimizer` interface
- ✓ Automatic history tracking
- ✓ Per-optimizer configuration
- ✓ Convergence monitoring
- ✓ Side-by-side comparison tools

---

## 📈 Project Structure

```
.
├── theory/                      # Theoretical analysis modules
│   ├── convergence_proof.py      # Lipschitz, convexity, condition number
│   └── numerical_stability.py    # Floating-point precision analysis
├── optimizers/                  # ⭐ NEW: Optimizer zoo
│   ├── base_optimizer.py         # Abstract base class
│   ├── sgd.py                    # Stochastic gradient descent
│   ├── momentum.py                # Classical & Nesterov momentum
│   ├── adagrad.py                 # Adaptive gradient
│   ├── rmsprop.py                 # RMSProp
│   └── adam.py                    # Adam & AdamW
├── benchmarks/                  # ⭐ NEW: Comparison tools
│   └── optimizer_comparison.py   # Comprehensive benchmarking
├── examples/                    # Demonstration scripts
│   └── convergence_theory_demo.py
├── tests/                       # Unit tests
│   ├── test_convergence_theory.py
│   └── test_optimizers.py         # ⭐ NEW: Optimizer tests
├── docs/                        # Mathematical documentation
│   ├── CONVERGENCE_THEORY.md     # Full derivations and proofs
│   ├── NUMERICAL_STABILITY.md    # Precision analysis guide
│   └── OPTIMIZER_ZOO.md           # ⭐ NEW: Optimizer guide
├── gradient_descent.py          # Core optimizer (enhanced)
├── data_generator.py            # Synthetic data generation
├── visualizer.py                # Loss landscape visualization
└── main.py                      # Example pipeline
```

---

## 🧮 Mathematical Foundations

### Convergence Guarantee (Phase 1)

For μ-strongly convex, L-smooth functions, gradient descent with optimal learning rate η* = 2/(L+μ) satisfies:

$$
\|\theta_k - \theta^*\|^2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|\theta_0 - \theta^*\|^2
$$

where κ = L/μ is the condition number.

**Implemented in:** `theory/convergence_proof.py`  
**Validated in:** `examples/convergence_theory_demo.py`

### Optimizer Comparison (Phase 2)

| Optimizer | Convergence Rate | Memory | When to Use |
|-----------|------------------|--------|-------------|
| SGD | O(1/k) | O(1) | Convex, provable guarantees |
| Momentum | O(1/k) | O(n) | Noisy gradients, high curvature |
| Nesterov | O(1/k²) | O(n) | Optimal convex convergence |
| AdaGrad | O(1/√k) | O(n) | Sparse features (NLP) |
| RMSProp | - | O(n) | RNNs, non-convex |
| Adam | - | O(n) | Default deep learning |
| AdamW | - | O(n) | Transformers, regularization |

See [OPTIMIZER_ZOO.md](docs/OPTIMIZER_ZOO.md) for detailed comparison.

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib scipy pytest
```

### Quick Start - Phase 1 (Convergence Theory)

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

### Quick Start - Phase 2 (Optimizer Zoo)

```python
from benchmarks.optimizer_comparison import OptimizerBenchmark

# Compare all optimizers
benchmark = OptimizerBenchmark(X, y, true_params={'W': 2.0, 'b': 5.0})
results = benchmark.compare_all_optimizers(learning_rate=0.1, epochs=500)

# Print summary table
from benchmarks.optimizer_comparison import print_summary_table
print_summary_table(results)
```

### Run Optimizer Benchmark

```bash
python benchmarks/optimizer_comparison.py
```

**Generates:**
- `optimizer_convergence.png` - Loss curves comparison
- `optimizer_trajectories.png` - Parameter space paths
- Summary statistics table

### Run Tests

```bash
# Phase 1 tests
pytest tests/test_convergence_theory.py -v

# Phase 2 tests
pytest tests/test_optimizers.py -v

# All tests
pytest tests/ -v
```

---

## 📊 Experimental Results

### Optimizer Convergence Comparison

![Optimizer Convergence](docs/images/optimizer_convergence.png)

**Key Observations:**
1. **Nesterov** converges fastest (O(1/k²) theoretical rate)
2. **Adam/AdamW** robust across learning rates
3. **AdaGrad** slows down over time (monotonic LR decay)
4. **Momentum** > vanilla SGD consistently

### Parameter Space Trajectories

![Trajectories](docs/images/optimizer_trajectories.png)

**Insights:**
- **SGD**: Direct path but slow
- **Momentum**: Overshoots then corrects
- **Nesterov**: Anticipates and corrects
- **Adam**: Smooth adaptive path

---

## 📚 Documentation

### Mathematical Theory
- [Convergence Theory](docs/CONVERGENCE_THEORY.md) - Full proofs and derivations
- [Numerical Stability](docs/NUMERICAL_STABILITY.md) - Floating-point analysis
- [Optimizer Zoo](docs/OPTIMIZER_ZOO.md) - Algorithm comparison guide

### API Reference
- `theory.convergence_proof.ConvergenceAnalyzer` - Theoretical analysis
- `theory.numerical_stability.NumericalStabilityAnalyzer` - Precision monitoring
- `optimizers.*` - All optimizer implementations
- `benchmarks.optimizer_comparison.OptimizerBenchmark` - Comparison tools

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

### ✅ Phase 2: Optimizer Zoo (Completed)
- [x] SGD (batch, mini-batch, online)
- [x] Momentum (Polyak)
- [x] Nesterov Accelerated Gradient
- [x] AdaGrad
- [x] RMSProp
- [x] Adam
- [x] AdamW
- [x] Unified optimizer interface
- [x] Comprehensive benchmarking suite
- [x] 30+ unit tests
- [x] Complete documentation

### 🚧 Phase 3: Non-Convex Extension (Next)
- [ ] Polynomial regression (degree 2-10)
- [ ] 2-layer neural networks
- [ ] Saddle point analysis
- [ ] Mode connectivity visualization
- [ ] Implicit bias studies

### 🗓️ Phase 4: High-Performance Computing
- [ ] JAX acceleration
- [ ] GPU support (CuPy)
- [ ] Distributed training
- [ ] Large-scale benchmarks (MNIST, etc.)

### 🗓️ Phase 5: Research Artifacts
- [ ] LaTeX paper draft
- [ ] Interactive web demo (Gradio/Streamlit)
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

### Foundational
1. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*. Springer.
2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge.
3. **Nocedal, J., & Wright, S.** (2006). *Numerical Optimization* (2nd ed.). Springer.

### Optimizer-Specific
4. **Polyak, B.T.** (1964). Some methods of speeding up the convergence of iteration methods.
5. **Duchi et al.** (2011). Adaptive Subgradient Methods. JMLR.
6. **Kingma & Ba** (2015). Adam: A Method for Stochastic Optimization. ICLR.
7. **Loshchilov & Hutter** (2019). Decoupled Weight Decay Regularization. ICLR.

### Numerical Analysis
8. **Higham, N.J.** (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
9. **Goldberg, D.** (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic.

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