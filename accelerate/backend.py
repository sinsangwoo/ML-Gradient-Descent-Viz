"""
Backend Management - Unified Array Interface

Automatically selects best available backend:
1. JAX (if available) - Best for GPU/TPU with auto-diff
2. CuPy (if available) - NumPy-compatible GPU arrays
3. NumPy (fallback) - CPU-only

Usage:
    import accelerate.backend as xp
    x = xp.array([1, 2, 3])  # Runs on GPU if available
"""

import numpy as np
from typing import Any, Tuple
import warnings


class Backend:
    """Wrapper for array backend (JAX/CuPy/NumPy)."""
    
    def __init__(self, name: str):
        self.name = name
        self._backend = None
        self._has_jit = False
        self._has_grad = False
        
        if name == 'jax':
            self._init_jax()
        elif name == 'cupy':
            self._init_cupy()
        elif name == 'numpy':
            self._init_numpy()
        else:
            raise ValueError(f"Unknown backend: {name}")
    
    def _init_jax(self):
        """Initialize JAX backend."""
        try:
            import jax
            import jax.numpy as jnp
            from jax import jit, grad, vmap
            
            self._backend = jnp
            self.jit = jit
            self.grad = grad
            self.vmap = vmap
            self._has_jit = True
            self._has_grad = True
            
            # Check GPU availability
            devices = jax.devices()
            self.devices = devices
            self.n_devices = len(devices)
            
            print(f"✓ JAX backend initialized")
            print(f"  Devices: {self.n_devices}x {devices[0].device_kind}")
            
        except ImportError:
            raise ImportError("JAX not installed. Install: pip install jax jaxlib")
    
    def _init_cupy(self):
        """Initialize CuPy backend."""
        try:
            import cupy as cp
            
            self._backend = cp
            self._has_jit = False
            self._has_grad = False
            
            # Check GPU
            self.n_devices = cp.cuda.runtime.getDeviceCount()
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            
            print(f"✓ CuPy backend initialized")
            print(f"  Devices: {self.n_devices}x {device_name}")
            
        except ImportError:
            raise ImportError("CuPy not installed. Install: pip install cupy-cuda11x")
    
    def _init_numpy(self):
        """Initialize NumPy backend (CPU fallback)."""
        self._backend = np
        self._has_jit = False
        self._has_grad = False
        self.n_devices = 1
        
        print("✓ NumPy backend initialized (CPU only)")
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to backend."""
        return getattr(self._backend, name)
    
    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert array to NumPy (for compatibility)."""
        if self.name == 'jax':
            return np.array(x)
        elif self.name == 'cupy':
            return x.get()
        else:
            return x
    
    def from_numpy(self, x: np.ndarray) -> Any:
        """Convert NumPy array to backend format."""
        if self.name == 'numpy':
            return x
        else:
            return self._backend.array(x)


# Global backend
_current_backend = None


def available_backends() -> list:
    """List available backends on this system."""
    backends = ['numpy']  # Always available
    
    try:
        import jax
        backends.append('jax')
    except ImportError:
        pass
    
    try:
        import cupy
        backends.append('cupy')
    except ImportError:
        pass
    
    return backends


def set_backend(name: str) -> Backend:
    """Set global backend."""
    global _current_backend
    
    available = available_backends()
    if name not in available:
        raise ValueError(
            f"Backend '{name}' not available. "
            f"Available: {available}"
        )
    
    _current_backend = Backend(name)
    return _current_backend


def get_backend() -> Backend:
    """Get current backend (auto-select if not set)."""
    global _current_backend
    
    if _current_backend is None:
        # Auto-select best backend
        available = available_backends()
        
        if 'jax' in available:
            print("Auto-selecting JAX backend...")
            _current_backend = Backend('jax')
        elif 'cupy' in available:
            print("Auto-selecting CuPy backend...")
            _current_backend = Backend('cupy')
        else:
            print("Using NumPy backend (CPU only)...")
            _current_backend = Backend('numpy')
    
    return _current_backend


if __name__ == "__main__":
    print("Available backends:", available_backends())
    
    # Test backend
    backend = get_backend()
    print(f"\nCurrent backend: {backend.name}")
    
    # Test array creation
    x = backend.array([1.0, 2.0, 3.0])
    print(f"Array type: {type(x)}")
    print(f"Array: {x}")