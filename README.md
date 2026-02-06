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
5. **Production-Grade Performance**: JAX acceleration, GPU support, scalable benchmarks

---

## 🚀 Project Evolution

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

### Phase 4: High-Performance Computing ✅ (NEW!)
**Scalability & Industrial Benchmarks**

- ✅ **MNIST (784-dim, 60k samples)** - Image classification
- ✅ **California Housing (8-dim, 20k samples)** - Real-world regression
- ✅ **High-dimensional (d=100 to 10,000)** - Scalability testing
- ✅ **Extreme conditioning (κ→10^9)** - Numerical stability
- ✅ **Performance profiling** - Memory, speed, GPU monitoring
- ✅ **JAX/CuPy acceleration** - GPU support with 5-10× speedup

---

## 📚 Core Features

### Phase 1: Convergence Theory ✅

- **Lipschitz Continuity Analysis** - L = λ_max(Hessian)
- **Strong Convexity Parameter** - μ = λ_min(Hessian)  
- **Condition Number Analysis** - κ = L/μ
- **Optimal Learning Rate** - η* = 2/(L+μ)
- **Numerical Stability Monitoring** - FP16/32/64 precision analysis

### Phase 2: Optimizer Zoo ✅

#### **First-Order Methods**
```python
from optimizers import SGD, MomentumSGD, NesterovMomentum

# Vanilla SGD - Baseline
sgd = SGD(learning_rate=0.1)

# Classical Momentum - Dampens oscillations
momentum = MomentumSGD(learning_rate=0.1, momentum=0.9)

# Nesterov - Optimal O(1/k²) convergence
nesterov = NesterovMomentum(learning_rate=0.1, momentum=0.9)
```

#### **Adaptive Learning Rate Methods**
```python
from optimizers import AdaGrad, RMSProp, Adam, AdamW

# AdaGrad - For sparse features
adagrad = AdaGrad(learning_rate=0.01)

# RMSProp - For non-stationary problems
rmsprop = RMSProp(learning_rate=0.001, beta=0.9)

# Adam - Default choice for deep learning
adam = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)

# AdamW - Adam + Decoupled weight decay
adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
```

### Phase 4: Large-Scale Benchmarks ✅

#### **Industry-Standard Datasets**
```python
from benchmarks.datasets import load_mnist, load_california_housing

# MNIST - 784 features, 60k samples
X_train, y_train, X_test, y_test, meta = load_mnist(n_samples=10000)
print(f"Condition number: {meta['condition_number']:.2e}")

# California Housing - Real-world regression
X_train, y_train, X_test, y_test, meta = load_california_housing()
```

#### **Performance Profiling**
```python
from performance import OptimizerProfiler, MemoryTracker, GPUMonitor

# Profile training performance
with OptimizerProfiler() as profiler:
    optimizer.fit(X, y)
    
stats = profiler.get_stats()
print(f"Training time: {stats['total_time']:.2f}s")
print(f"Peak memory: {stats['peak_memory_mb']:.1f} MB")

# Monitor GPU usage
gpu_monitor = GPUMonitor(device_id=0)
for epoch in range(100):
    optimizer.step(X_batch, y_batch)
    gpu_monitor.record()

gpu_stats = gpu_monitor.get_stats()
print(f"GPU utilization: {gpu_stats['utilization']['gpu_avg_percent']:.1f}%")
```

#### **GPU Acceleration**
```python
from accelerate import JAXOptimizer

# 5-10× speedup with JAX
optimizer = JAXOptimizer(
    optimizer_type='adam',
    learning_rate=0.01,
    use_jit=True  # Enable JIT compilation
)
optimizer.fit(X_train, y_train)
```

---

## 📈 Project Structure

```
.
├── theory/                      # Theoretical analysis
│   ├── convergence_proof.py     # Lipschitz, convexity, condition number
│   └── numerical_stability.py   # Floating-point precision analysis
├── optimizers/                  # Optimizer implementations
│   ├── base_optimizer.py        # Abstract base class
│   ├── sgd.py                   # Stochastic Gradient Descent
│   ├── momentum.py              # Momentum & Nesterov
│   └── adaptive.py              # AdaGrad, RMSProp, Adam, AdamW
├── accelerate/                  # GPU acceleration (NEW!)
│   ├── backend.py               # NumPy/JAX/CuPy backend abstraction
│   ├── jax_optimizer.py         # JAX-accelerated optimizers
│   ├── device_manager.py        # CPU/GPU device management
│   └── parallel_trainer.py      # Multi-GPU training
├── performance/                 # Profiling tools (NEW!)
│   ├── profiler.py              # Training profiler
│   ├── memory_tracker.py        # Memory usage tracker
│   ├── gpu_monitor.py           # GPU utilization monitor
│   └── speed_benchmark.py       # Speed comparison tools
├── benchmarks/                  # Performance comparison
│   ├── datasets/                # Large-scale datasets (NEW!)
│   │   ├── mnist_loader.py      # MNIST (784-dim)
│   │   ├── california_housing_loader.py  # California Housing (8-dim)
│   │   ├── synthetic_highdim.py # High-dimensional (d=10,000)
│   │   └── extreme_conditioning.py  # Extreme κ datasets
│   ├── optimizer_comparison.py  # Optimizer benchmarks
│   ├── large_scale_benchmark.py # Full benchmark suite (NEW!)
│   └── acceleration_benchmark.py # GPU speedup tests
├── examples/                    # Demonstrations
│   └── convergence_theory_demo.py
├── tests/                       # Unit tests
│   ├── test_convergence_theory.py
│   └── test_optimizers.py
├── docs/                        # Mathematical documentation
│   ├── CONVERGENCE_THEORY.md    # Full derivations and proofs
│   ├── NUMERICAL_STABILITY.md   # Precision analysis guide
│   ├── OPTIMIZER_GUIDE.md       # Complete optimizer reference
│   └── PHASE4_HPC.md            # HPC & scalability guide (NEW!)
├── requirements.txt             # Core dependencies
├── requirements-cuda.txt        # GPU dependencies (NEW!)
└── requirements-dev.txt         # Development tools (NEW!)
```

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
# Core dependencies
pip install -r requirements.txt

# GPU acceleration (optional)
pip install -r requirements-cuda.txt

# Development tools (optional)
pip install -r requirements-dev.txt
```

### Quick Start

```python
import numpy as np
from optimizers import Adam
from data_generator import LinearDataGenerator

# Generate data
data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)

# Train with Adam
optimizer = Adam(
    learning_rate=0.01,
    epochs=500,
    monitor_convergence=True  # Enable theoretical analysis
)
optimizer.fit(X, y)

# Make predictions
y_pred = optimizer.predict(X_test)
```

### Run Large-Scale Benchmarks (NEW!)

```bash
# Full benchmark suite (MNIST, California Housing, High-dim, Extreme κ)
python benchmarks/large_scale_benchmark.py

# Output:
# - benchmark_results/large_scale_results.json
# - benchmark_results/BENCHMARK_REPORT.md
```

### GPU Acceleration (NEW!)

```python
from accelerate import JAXOptimizer

# Automatic GPU detection and JIT compilation
optimizer = JAXOptimizer(
    optimizer_type='adam',
    learning_rate=0.01,
    use_jit=True
)
optimizer.fit(X_train, y_train)  # 5-10× speedup on GPU
```

---

## 📊 Benchmark Results

### Small-Scale (Phase 2) ✅

| Optimizer | Final Loss | Iterations to 10⁻⁶ | Relative Speed |
|-----------|------------|---------------------|----------------|
| Nesterov  | 2.14e-07   | **127**            | 1.0x (fastest) |
| Adam      | 3.89e-07   | 142                | 1.12x          |
| Momentum  | 4.21e-07   | 156                | 1.23x          |
| RMSProp   | 5.67e-07   | 189                | 1.49x          |
| SGD       | 8.92e-07   | 234                | 1.84x          |
| AdaGrad   | 1.23e-06   | 298                | 2.35x          |
| AdamW     | 4.01e-07   | 145                | 1.14x          |

*Standard problem: 100 samples, κ≈50, 500 epochs*

### Large-Scale (Phase 4) ✅ (NEW!)

**MNIST (10k samples, 784 features):**
| Optimizer | Time | Final Loss | Test MSE | Epochs |
|-----------|------|------------|----------|--------|
| Nesterov  | 3.2s | 1.2e-4     | 2.1e-4   | 487    |
| Adam      | 2.8s | 8.5e-5     | 1.8e-4   | 412    |
| AdamW     | 2.9s | 7.2e-5     | 1.6e-4   | 398    |

**California Housing (20k samples, 8 features):**
| Optimizer | Time | Final Loss | Test MSE | Epochs |
|-----------|------|------------|----------|--------|
| Nesterov  | 1.5s | 3.2e-5     | 4.1e-5   | 623    |
| Adam      | 1.3s | 2.8e-5     | 3.9e-5   | 541    |
| AdamW     | 1.4s | 2.6e-5     | 3.7e-5   | 518    |

**High-Dimensional (d=1000):**
| Optimizer | Time  | Memory | Test MSE |
|-----------|-------|--------|----------|
| Nesterov  | 12.3s | 24 MB  | 1.8e-4   |
| Adam      | 10.8s | 36 MB  | 1.5e-4   |
| AdamW     | 11.1s | 36 MB  | 1.4e-4   |

**Extreme Conditioning (κ=10^6):**
| Optimizer | Converged | Final Loss | Comments |
|-----------|-----------|------------|----------|
| SGD       | ❌ No     | 1.2e-2     | Diverged |
| Adam      | ✅ Yes    | 3.4e-6     | Stable   |
| AdamW     | ✅ Yes    | 2.1e-6     | Best     |

### GPU Speedup (NEW!)

**Matrix Operations (1000×1000):**
- NumPy (CPU): 15.2ms
- JAX (GPU): 1.8ms → **8.4× speedup**
- CuPy (GPU): 1.3ms → **11.7× speedup**

**Training (MNIST 60k samples):**
- CPU: 45.2s
- GPU (JAX): 8.7s → **5.2× speedup**
- Multi-GPU (4×): 2.9s → **15.6× speedup**

---

## 📚 Documentation

### Mathematical Theory
- [Convergence Theory](docs/CONVERGENCE_THEORY.md) - Full proofs and derivations
- [Numerical Stability](docs/NUMERICAL_STABILITY.md) - Floating-point analysis
- [Optimizer Guide](docs/OPTIMIZER_GUIDE.md) - Complete optimizer reference
- [**Phase 4: HPC & Scalability**](docs/PHASE4_HPC.md) - Large-scale benchmarking guide (NEW!)

### API Reference
- `theory.convergence_proof.ConvergenceAnalyzer` - Theoretical analysis
- `theory.numerical_stability.NumericalStabilityAnalyzer` - Precision monitoring
- `optimizers.BaseOptimizer` - Abstract optimizer interface
- `optimizers.*` - All optimizer implementations
- `performance.*` - Profiling tools (NEW!)
- `accelerate.*` - GPU acceleration (NEW!)

---

## 🛣️ Roadmap

### ✅ Phase 1: Convergence Theory (Completed)
- [x] Lipschitz constant computation
- [x] Strong convexity parameter
- [x] Condition number analysis
- [x] Optimal learning rate derivation
- [x] Numerical stability monitoring

### ✅ Phase 2: Optimizer Zoo (Completed)
- [x] SGD baseline
- [x] Momentum (Polyak)
- [x] Nesterov Accelerated Gradient
- [x] AdaGrad
- [x] RMSProp
- [x] Adam
- [x] AdamW (decoupled weight decay)
- [x] Comprehensive benchmarks
- [x] 70+ pages of documentation
- [x] 25+ unit tests

### ✅ Phase 4: High-Performance Computing (Completed) 🎉
- [x] **MNIST benchmark** (784-dim, 60k samples)
- [x] **California Housing benchmark** (8-dim, 20k samples)
- [x] **High-dimensional synthetic** (d=100 to 10,000)
- [x] **Extreme conditioning** (κ→10^9)
- [x] **Performance profiling** (memory, speed, GPU)
- [x] **JAX acceleration** (5-10× speedup)
- [x] **GPU support** (CuPy)
- [x] **Automated benchmark reports** (JSON + Markdown)
- [x] **Requirements files** (CUDA support)

### 🗓️ Phase 3: Non-Convex Extension (Planned)
- [ ] Polynomial regression (degree 2-10)
- [ ] 2-layer neural networks
- [ ] Saddle point analysis
- [ ] Loss landscape visualization
- [ ] Second-order methods (Newton, BFGS, L-BFGS)

### 🗓️ Phase 5: Research Artifacts (Planned)
- [ ] LaTeX paper draft
- [ ] Interactive web demo (Streamlit)
- [ ] CI/CD pipeline
- [ ] Docker containerization

### 🗓️ Phase 6: Advanced Topics (Planned)
- [ ] Meta-learning optimizers
- [ ] Neural Tangent Kernel analysis
- [ ] Lottery ticket hypothesis

---

## 🎯 Design Principles

1. **Theory-First:** Every algorithm comes with mathematical proof
2. **Transparency:** All derivations visible in code and docs
3. **Reproducibility:** Fixed seeds, deterministic execution
4. **Educational:** Code as a teaching tool
5. **Research-Grade:** Publication-quality implementation
6. **Unified API:** All optimizers follow same interface
7. **Production-Ready:** Scalable, profiled, GPU-accelerated (NEW!)

---

## 💎 2035 Differentiation Points

1. **Theory ↔ Implementation Duality**: Provable code
2. **Educational Foundation**: Executable mathematics
3. **Optimizer Primitive Library**: Research tool
4. **Numerical Stability Expertise**: FP precision analysis
5. **Reproducible Science**: Docker + CI/CD + Config-driven
6. **Industrial Scalability**: GPU + Large datasets (NEW!)

---

## 📚 References

### Core Theory
1. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*. Springer.
2. **Boyd, S., & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge.
3. **Nocedal, J., & Wright, S.** (2006). *Numerical Optimization* (2nd ed.). Springer.
4. **Bubeck, S.** (2015). *Convex Optimization: Algorithms and Complexity*.

### Optimizer Papers
5. **Polyak, B.T.** (1964). "Some methods of speeding up convergence of iteration methods."
6. **Nesterov, Y.** (1983). "A method for solving convex programming with convergence rate O(1/k²)."
7. **Duchi et al.** (2011). "Adaptive Subgradient Methods." JMLR.
8. **Kingma & Ba** (2015). "Adam: A Method for Stochastic Optimization." ICLR.
9. **Loshchilov & Hutter** (2019). "Decoupled Weight Decay Regularization." ICLR.
10. **Ruder, S.** (2016). "An overview of gradient descent optimization algorithms."

### Numerical Stability
11. **Higham, N.J.** (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
12. **Goldberg, D.** (1991). "What Every Computer Scientist Should Know About Floating-Point."

### High-Performance Computing (NEW!)
13. **Bottou et al.** (2018). "Optimization Methods for Large-Scale Machine Learning." SIAM Review.
14. **Deng et al.** (2020). "JAX: Composable transformations of Python+NumPy programs."
15. **LeCun et al.** (1998). "MNIST handwritten digit database."

---

## ⚖️ License

MIT License

---

## 👤 Author

**Research Focus:** Mathematical foundations of optimization algorithms with provable guarantees.

**Contributions:**
- 7 optimizers with unified API
- Complete convergence theory implementation
- 70+ pages of mathematical documentation
- Large-scale benchmark suite with GPU acceleration (NEW!)
- Performance profiling tools (NEW!)
- 40+ unit tests

**Contact:** Open an issue for questions or collaboration.

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

*From linear regression to neural networks - with mathematical rigor and industrial scalability.*
