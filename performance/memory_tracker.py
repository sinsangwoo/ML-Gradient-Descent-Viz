"""
Memory Tracker
==============

Detailed memory profiling for optimization algorithms.

Tracks:
- Per-array memory allocations
- Gradient accumulation memory
- Optimizer state memory (momentum, second moments)
- Peak memory usage
"""

import numpy as np
from typing import Dict, List
import psutil
import os


class MemoryTracker:
    """
    Track memory consumption of optimizer components.
    """
    
    def __init__(self):
        self.allocations: Dict[str, List[float]] = {}
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
    
    def track_array(self, name: str, array: np.ndarray):
        """Track memory of a numpy array."""
        memory_mb = array.nbytes / 1024 / 1024
        
        if name not in self.allocations:
            self.allocations[name] = []
        
        self.allocations[name].append(memory_mb)
    
    def get_current_memory(self) -> float:
        """Get current process memory in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_increase(self) -> float:
        """Get memory increase since baseline."""
        return self.get_current_memory() - self.baseline_memory
    
    def get_summary(self) -> Dict:
        """Get memory usage summary."""
        summary = {
            'baseline_mb': self.baseline_memory,
            'current_mb': self.get_current_memory(),
            'increase_mb': self.get_memory_increase(),
            'tracked_arrays': {}
        }
        
        for name, sizes in self.allocations.items():
            summary['tracked_arrays'][name] = {
                'count': len(sizes),
                'total_mb': sum(sizes),
                'avg_mb': np.mean(sizes),
                'max_mb': max(sizes)
            }
        
        return summary
    
    def estimate_optimizer_memory(self, 
                                 n_params: int, 
                                 optimizer_type: str,
                                 dtype: str = 'float32') -> Dict:
        """
        Estimate memory required by optimizer.
        
        Parameters
        ----------
        n_params : int
            Number of parameters.
        optimizer_type : str
            Type of optimizer ('sgd', 'momentum', 'adam', etc.).
        dtype : str
            Data type ('float32' or 'float64').
        
        Returns
        -------
        memory_estimate : dict
            Breakdown of memory usage.
        """
        bytes_per_param = 4 if dtype == 'float32' else 8
        param_memory_mb = n_params * bytes_per_param / 1024 / 1024
        
        # Memory multipliers for different optimizers
        multipliers = {
            'sgd': 1.0,              # Just parameters
            'momentum': 2.0,         # Parameters + velocity
            'nesterov': 2.0,         # Parameters + momentum
            'adagrad': 2.0,          # Parameters + accumulated gradients
            'rmsprop': 2.0,          # Parameters + squared gradients
            'adam': 3.0,             # Parameters + first moment + second moment
            'adamw': 3.0             # Same as Adam
        }
        
        multiplier = multipliers.get(optimizer_type.lower(), 1.0)
        total_memory_mb = param_memory_mb * multiplier
        
        return {
            'n_params': n_params,
            'dtype': dtype,
            'bytes_per_param': bytes_per_param,
            'param_memory_mb': param_memory_mb,
            'optimizer_type': optimizer_type,
            'memory_multiplier': multiplier,
            'total_memory_mb': total_memory_mb,
            'breakdown': {
                'parameters': param_memory_mb,
                'optimizer_state': param_memory_mb * (multiplier - 1)
            }
        }


if __name__ == '__main__':
    # Test memory tracking
    tracker = MemoryTracker()
    
    # Simulate optimizer state
    n = 10000
    params = np.random.randn(n).astype(np.float32)
    gradient = np.random.randn(n).astype(np.float32)
    velocity = np.random.randn(n).astype(np.float32)
    
    tracker.track_array('parameters', params)
    tracker.track_array('gradient', gradient)
    tracker.track_array('velocity', velocity)
    
    print("Memory Tracking Summary:")
    summary = tracker.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\nEstimated Memory for Adam with 1M parameters:")
    estimate = tracker.estimate_optimizer_memory(1_000_000, 'adam', 'float32')
    for key, value in estimate.items():
        print(f"  {key}: {value}")
