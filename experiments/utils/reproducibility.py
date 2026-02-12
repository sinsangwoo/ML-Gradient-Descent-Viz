"""
Reproducibility Utilities
==========================

Functions to ensure fully reproducible experiments:
- Random seed management
- System information capture
- Experiment metadata tracking
"""

import os
import sys
import random
import numpy as np
import platform
import json
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib


def set_random_seeds(
    seed: int = 42,
    numpy_seed: Optional[int] = None,
    python_hash_seed: Optional[int] = None,
    deterministic: bool = True
) -> None:
    """
    Set all random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Master seed value.
    numpy_seed : int, optional
        NumPy-specific seed (defaults to `seed`).
    python_hash_seed : int, optional
        Python hash seed (defaults to `seed`).
    deterministic : bool
        If True, enable maximum determinism (may impact performance).
    
    Notes
    -----
    Some operations may still be non-deterministic:
    - GPU atomicAdd operations
    - Floating-point associativity differences
    - Parallel reduction order
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(numpy_seed if numpy_seed is not None else seed)
    
    # Python hash seed (must be set via environment before Python starts)
    if python_hash_seed is not None:
        os.environ['PYTHONHASHSEED'] = str(python_hash_seed)
    
    # JAX (if available)
    try:
        import jax
        jax.config.update('jax_enable_x64', True)  # Use float64 for precision
        if deterministic:
            jax.config.update('jax_default_matmul_precision', 'highest')
    except ImportError:
        pass
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_system_info() -> Dict[str, Any]:
    """
    Capture system information for reproducibility.
    
    Returns
    -------
    info : dict
        System information including:
        - Python version
        - NumPy version
        - OS information
        - CPU information
        - GPU information (if available)
        - Memory
    """
    info = {
        'timestamp': datetime.now().isoformat(),
        'python': {
            'version': sys.version,
            'executable': sys.executable,
        },
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'packages': {},
    }
    
    # Package versions
    try:
        import numpy
        info['packages']['numpy'] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import scipy
        info['packages']['scipy'] = scipy.__version__
    except ImportError:
        pass
    
    try:
        import jax
        info['packages']['jax'] = jax.__version__
    except ImportError:
        pass
    
    try:
        import torch
        info['packages']['torch'] = torch.__version__
    except ImportError:
        pass
    
    # GPU information
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        info['gpu'] = {
            'available': True,
            'count': device_count,
            'devices': []
        }
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info['gpu']['devices'].append({
                'id': i,
                'name': name,
                'memory_total_mb': memory.total / 1024 / 1024,
            })
    except:
        info['gpu'] = {'available': False}
    
    # Memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
        }
    except:
        info['memory'] = {}
    
    return info


def compute_config_hash(config: Dict) -> str:
    """
    Compute deterministic hash of configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    hash : str
        SHA256 hash of config (first 16 characters).
    """
    # Convert to canonical JSON string
    config_str = json.dumps(config, sort_keys=True, default=str)
    
    # Compute hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


def save_experiment_metadata(
    config: Dict,
    results: Dict,
    output_path: str
) -> None:
    """
    Save complete experiment metadata for reproducibility.
    
    Parameters
    ----------
    config : dict
        Experiment configuration.
    results : dict
        Experiment results.
    output_path : str
        Path to save metadata JSON.
    """
    metadata = {
        'config': config,
        'config_hash': compute_config_hash(config),
        'results': results,
        'system_info': get_system_info(),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


if __name__ == '__main__':
    # Test reproducibility utilities
    print("Setting random seeds...")
    set_random_seeds(42)
    
    print("\nGenerating random numbers:")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    
    print("\nSystem information:")
    info = get_system_info()
    print(json.dumps(info, indent=2, default=str))
