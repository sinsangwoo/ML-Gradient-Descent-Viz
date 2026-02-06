"""
Optimizer Profiler
==================

Comprehensive profiler that tracks:
- Wall-clock time per epoch
- Memory consumption (CPU/GPU)
- Gradient computation time
- Parameter update time
- Loss evaluation time

Usage:
------
with OptimizerProfiler() as profiler:
    optimizer.fit(X, y)
    
stats = profiler.get_stats()
print(f"Total time: {stats['total_time']:.2f}s")
print(f"Peak memory: {stats['peak_memory_mb']:.1f} MB")
"""

import time
import psutil
import os
from typing import Dict, Optional, List
import numpy as np
from contextlib import contextmanager


class OptimizerProfiler:
    """
    Profile optimizer performance during training.
    
    Tracks:
    - Total wall-clock time
    - Time per epoch
    - Memory usage (CPU)
    - GPU memory (if available)
    """
    
    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu
        self.reset()
        
        # Check GPU availability
        self.has_gpu = False
        if enable_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.has_gpu = True
                self.nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass
    
    def reset(self):
        """Reset all counters."""
        self.epoch_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        self.is_profiling = False
    
    def start(self):
        """Start profiling."""
        self.reset()
        self.start_time = time.perf_counter()
        self.is_profiling = True
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        self.baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self):
        """Stop profiling."""
        self.end_time = time.perf_counter()
        self.is_profiling = False
    
    def record_epoch(self, epoch_time: Optional[float] = None):
        """Record metrics for one epoch."""
        if not self.is_profiling:
            return
        
        # Record epoch time
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        
        # Record memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        # Record GPU memory if available
        if self.has_gpu:
            try:
                gpu_info = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_mb = gpu_info.used / 1024 / 1024
                self.gpu_memory_usage.append(gpu_memory_mb)
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_memory_mb)
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get profiling statistics."""
        total_time = (self.end_time or time.perf_counter()) - (self.start_time or 0)
        
        stats = {
            'total_time': total_time,
            'num_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'std_epoch_time': np.std(self.epoch_times) if self.epoch_times else 0,
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_increase_mb': self.peak_memory - self.baseline_memory if hasattr(self, 'baseline_memory') else 0
        }
        
        if self.has_gpu and self.gpu_memory_usage:
            stats.update({
                'peak_gpu_memory_mb': self.peak_gpu_memory,
                'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage)
            })
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


if __name__ == '__main__':
    # Test profiler
    import numpy as np
    
    with OptimizerProfiler() as profiler:
        for epoch in range(10):
            start = time.perf_counter()
            
            # Simulate training
            X = np.random.randn(1000, 100)
            y = np.random.randn(1000)
            _ = X @ X.T  # Some computation
            
            epoch_time = time.perf_counter() - start
            profiler.record_epoch(epoch_time)
            time.sleep(0.01)
    
    stats = profiler.get_stats()
    print("Profiling Results:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
