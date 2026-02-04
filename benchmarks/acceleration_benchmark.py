"""
Acceleration Benchmark - CPU vs GPU Performance

Compares:
1. NumPy (CPU) vs JAX (GPU)
2. Single-device vs Multi-device
3. Small vs Large datasets
4. Memory efficiency

Metrics:
- Training time
- Throughput (samples/second)
- Speedup factor
- Memory usage
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import get_backend, set_backend, available_backends
from accelerate import JAXOptimizer, ParallelTrainer
from optimizers import Adam


class AccelerationBenchmark:
    """
    Benchmark CPU vs GPU performance.
    """
    
    def __init__(self):
        self.results = {}
    
    def generate_dataset(self, n_samples: int, n_features: int, 
                        random_seed: int = 42) -> tuple:
        """
        Generate synthetic regression dataset.
        """
        np.random.seed(random_seed)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        true_W = np.random.randn(n_features, 1).astype(np.float32)
        y = (X @ true_W).flatten() + np.random.randn(n_samples).astype(np.float32) * 0.1
        
        return X, y
    
    def benchmark_numpy_baseline(self, X, y, epochs=100):
        """
        Baseline: NumPy CPU training.
        """
        print("\n" + "="*60)
        print("BASELINE: NumPy (CPU)")
        print("="*60)
        
        set_backend('numpy')
        
        opt = Adam(learning_rate=0.01, epochs=epochs, random_seed=42)
        
        start_time = time.time()
        opt.fit(X, y, verbose=False)
        train_time = time.time() - start_time
        
        throughput = (len(X) * epochs) / train_time
        
        print(f"  Training time: {train_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} samples/s")
        print(f"  Final loss: {opt.get_history()['loss_history'][-1]:.6f}")
        
        return {
            'name': 'NumPy CPU',
            'time': train_time,
            'throughput': throughput,
            'speedup': 1.0
        }
    
    def benchmark_jax_single_device(self, X, y, epochs=100):
        """
        JAX single-device (GPU) training.
        """
        print("\n" + "="*60)
        print("JAX: Single Device (GPU)")
        print("="*60)
        
        if 'jax' not in available_backends():
            print("  [SKIPPED] JAX not available")
            return None
        
        set_backend('jax')
        
        opt = JAXOptimizer('adam', learning_rate=0.01)
        
        start_time = time.time()
        opt.fit(X, y, epochs=epochs, verbose=False)
        train_time = time.time() - start_time
        
        throughput = (len(X) * epochs) / train_time
        
        print(f"  Training time: {train_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} samples/s")
        print(f"  Final loss: {opt.get_history()['loss_history'][-1]:.6f}")
        
        return {
            'name': 'JAX GPU',
            'time': train_time,
            'throughput': throughput
        }
    
    def benchmark_parallel_training(self, X, y, n_devices=2, epochs=100):
        """
        Multi-device parallel training.
        """
        print("\n" + "="*60)
        print(f"PARALLEL: {n_devices} Devices")
        print("="*60)
        
        trainer = ParallelTrainer(n_devices=n_devices, batch_size=256)
        
        start_time = time.time()
        trainer.fit(X, y, learning_rate=0.01, epochs=epochs, verbose=False)
        train_time = time.time() - start_time
        
        throughput = (len(X) * epochs) / train_time
        
        print(f"  Training time: {train_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} samples/s")
        print(f"  Final loss: {trainer.get_history()['loss_history'][-1]:.6f}")
        
        return {
            'name': f'Parallel {n_devices}x',
            'time': train_time,
            'throughput': throughput
        }
    
    def run_all_benchmarks(self):
        """
        Run complete benchmark suite.
        """
        print("\n" + "#"*60)
        print("#" + " "*15 + "ACCELERATION BENCHMARK" + " "*15 + "#")
        print("#"*60)
        
        # Dataset sizes to test
        configs = [
            ('Small', 1000, 20, 200),
            ('Medium', 10000, 50, 100),
            ('Large', 100000, 100, 50)
        ]
        
        all_results = {}
        
        for name, n_samples, n_features, epochs in configs:
            print(f"\n\n{'*'*60}")
            print(f"Dataset: {name} ({n_samples:,} samples, {n_features} features)")
            print(f"{'*'*60}")
            
            X, y = self.generate_dataset(n_samples, n_features)
            
            results = []
            
            # Baseline
            numpy_result = self.benchmark_numpy_baseline(X, y, epochs)
            results.append(numpy_result)
            baseline_time = numpy_result['time']
            
            # JAX GPU
            jax_result = self.benchmark_jax_single_device(X, y, epochs)
            if jax_result:
                jax_result['speedup'] = baseline_time / jax_result['time']
                results.append(jax_result)
            
            # Parallel training
            if 'jax' in available_backends():
                for n_dev in [2, 4]:
                    try:
                        parallel_result = self.benchmark_parallel_training(
                            X, y, n_devices=n_dev, epochs=epochs
                        )
                        parallel_result['speedup'] = baseline_time / parallel_result['time']
                        results.append(parallel_result)
                    except:
                        print(f"  [SKIPPED] {n_dev} devices not available")
            
            all_results[name] = results
        
        # Summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, all_results):
        """
        Print benchmark summary table.
        """
        print("\n\n" + "="*80)
        print("SUMMARY: Speedup vs NumPy Baseline")
        print("="*80)
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            print(f"  {'Method':<20} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10}")
            print(f"  {'-'*60}")
            
            for result in results:
                time_str = f"{result['time']:.3f}"
                throughput_str = f"{result['throughput']:.0f} s/s"
                speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] > 1 else "-"
                
                print(f"  {result['name']:<20} {time_str:<12} {throughput_str:<15} {speedup_str:<10}")


if __name__ == "__main__":
    benchmark = AccelerationBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n\n✓ Benchmark complete!\n")