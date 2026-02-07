"""
Speed Benchmark
===============

Benchmark training speed across different configurations:
- CPU vs GPU
- NumPy vs JAX vs CuPy
- Different batch sizes
- Different problem dimensions
"""

import time
import numpy as np
from typing import Dict, List, Callable, Tuple
import json


class SpeedBenchmark:
    """
    Benchmark optimizer speed with different backends.
    """
    
    def __init__(self, warmup_runs: int = 3):
        self.warmup_runs = warmup_runs
        self.results: List[Dict] = []
    
    def benchmark_function(
        self,
        func: Callable,
        *args,
        n_runs: int = 10,
        **kwargs
    ) -> Dict:
        """
        Benchmark a function's execution time.
        
        Parameters
        ----------
        func : callable
            Function to benchmark.
        *args, **kwargs
            Arguments to pass to function.
        n_runs : int
            Number of timing runs.
        
        Returns
        -------
        timing_stats : dict
        """
        # Warmup
        for _ in range(self.warmup_runs):
            _ = func(*args, **kwargs)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'n_runs': n_runs
        }
    
    def benchmark_optimizer(
        self,
        optimizer_cls,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict,
        n_runs: int = 3
    ) -> Dict:
        """
        Benchmark an optimizer on a dataset.
        
        Parameters
        ----------
        optimizer_cls : class
            Optimizer class.
        X, y : np.ndarray
            Training data.
        config : dict
            Optimizer configuration.
        n_runs : int
            Number of runs.
        
        Returns
        -------
        results : dict
        """
        times = []
        final_losses = []
        
        for run in range(n_runs):
            optimizer = optimizer_cls(**config)
            
            start = time.perf_counter()
            optimizer.fit(X, y)
            end = time.perf_counter()
            
            times.append(end - start)
            
            # Get final loss
            history = optimizer.get_history()
            if history and 'losses' in history:
                final_losses.append(history['losses'][-1])
        
        result = {
            'optimizer': optimizer_cls.__name__,
            'config': config,
            'dataset_shape': X.shape,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput_samples_per_sec': len(X) / np.mean(times) if times else 0
        }
        
        if final_losses:
            result['mean_final_loss'] = np.mean(final_losses)
            result['std_final_loss'] = np.std(final_losses)
        
        self.results.append(result)
        return result
    
    def compare_backends(
        self,
        operation: Callable,
        data_numpy: np.ndarray,
        backends: List[str] = ['numpy'],
        n_runs: int = 10
    ) -> Dict:
        """
        Compare operation speed across different backends.
        
        Parameters
        ----------
        operation : callable
            Operation to benchmark (e.g., matrix multiply).
        data_numpy : np.ndarray
            Input data as NumPy array.
        backends : list of str
            Backends to test ('numpy', 'jax', 'cupy').
        n_runs : int
            Number of runs.
        
        Returns
        -------
        comparison : dict
        """
        results = {}
        
        for backend in backends:
            if backend == 'numpy':
                data = data_numpy
                
            elif backend == 'jax':
                try:
                    import jax.numpy as jnp
                    data = jnp.array(data_numpy)
                except ImportError:
                    print(f"Skipping {backend}: not installed")
                    continue
                    
            elif backend == 'cupy':
                try:
                    import cupy as cp
                    data = cp.array(data_numpy)
                except ImportError:
                    print(f"Skipping {backend}: not installed")
                    continue
            
            else:
                print(f"Unknown backend: {backend}")
                continue
            
            # Benchmark
            timing = self.benchmark_function(operation, data, n_runs=n_runs)
            results[backend] = timing
        
        # Compute speedups relative to NumPy
        if 'numpy' in results:
            baseline = results['numpy']['mean_time']
            for backend in results:
                if backend != 'numpy':
                    speedup = baseline / results[backend]['mean_time']
                    results[backend]['speedup_vs_numpy'] = speedup
        
        return results
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to {filepath}")
    
    def print_summary(self):
        """Print formatted summary of results."""
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "="*60)
        print("SPEED BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"\nOptimizer: {result['optimizer']}")
            print(f"  Dataset: {result['dataset_shape']}")
            print(f"  Mean time: {result['mean_time']:.4f}s (±{result['std_time']:.4f}s)")
            print(f"  Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")
            if 'mean_final_loss' in result:
                print(f"  Final loss: {result['mean_final_loss']:.6f}")


if __name__ == '__main__':
    # Test speed benchmark
    benchmark = SpeedBenchmark()
    
    # Simple matrix multiplication benchmark
    print("Benchmarking matrix multiplication...")
    
    sizes = [100, 500, 1000]
    for size in sizes:
        X = np.random.randn(size, size).astype(np.float32)
        
        def matmul(X):
            return X @ X.T
        
        results = benchmark.compare_backends(
            matmul,
            X,
            backends=['numpy'],  # Add 'jax', 'cupy' if available
            n_runs=5
        )
        
        print(f"\nSize: {size}x{size}")
        for backend, timing in results.items():
            print(f"  {backend}: {timing['mean_time']*1000:.2f}ms")
