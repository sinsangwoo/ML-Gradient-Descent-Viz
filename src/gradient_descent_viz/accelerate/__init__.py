"""
High-Performance Computing Module

Provides GPU acceleration and parallelization for all models and optimizers.

Backends:
- JAX: XLA compilation, auto-differentiation, GPU/TPU support
- CuPy: NumPy-compatible GPU arrays (CUDA)
- Fallback: Pure NumPy for CPU-only environments
"""

from .backend import get_backend, set_backend, available_backends
from .jax_optimizer import JAXOptimizer
from .parallel_trainer import ParallelTrainer
from .device_manager import DeviceManager

__all__ = [
    'get_backend',
    'set_backend', 
    'available_backends',
    'JAXOptimizer',
    'ParallelTrainer',
    'DeviceManager'
]