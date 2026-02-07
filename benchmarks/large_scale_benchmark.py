"""
Large-Scale Benchmark Runner
=============================

Comprehensive benchmark suite for Phase 4:
1. MNIST (784-dim, 60k samples)
2. California Housing (8-dim, 20k samples)
3. Synthetic High-Dim (d=100, 1000, 5000, 10000)
4. Extreme Condition Numbers (κ = 10^3, 10^6, 10^9)

For each dataset:
- Test all 7 optimizers
- Profile memory and speed
- Track convergence rates
- Generate comparison plots
- Export results to JSON and PDF
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import time
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.datasets import (
    load_mnist,
    load_california_housing,
    generate_highdim_regression,
    generate_extreme_condition_data
)
from performance import OptimizerProfiler, MemoryTracker, SpeedBenchmark

# Import optimizers
try:
    from optimizers import SGD, MomentumSGD, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW
    OPTIMIZERS = {
        'SGD': SGD,
        'Momentum': MomentumSGD,
        'Nesterov': NesterovMomentum,
        'AdaGrad': AdaGrad,
        'RMSProp': RMSProp,
        'Adam': Adam,
        'AdamW': AdamW
    }
except ImportError as e:
    print(f"Warning: Could not import optimizers: {e}")
    OPTIMIZERS = {}


class LargeScaleBenchmark:
    """
    Run comprehensive benchmarks on large-scale datasets.
    """
    
    def __init__(self, output_dir: str = './benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
    
    def benchmark_dataset(
        self,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metadata: Dict,
        optimizer_configs: Dict[str, Dict],
        max_epochs: int = 1000,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Benchmark all optimizers on a single dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        X_train, y_train : np.ndarray
            Training data.
        X_test, y_test : np.ndarray
            Test data.
        metadata : dict
            Dataset metadata.
        optimizer_configs : dict
            Optimizer configurations {name: config}.
        max_epochs : int
            Maximum training epochs.
        tolerance : float
            Convergence tolerance.
        
        Returns
        -------
        results : dict
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {dataset_name}")
        print(f"Dataset shape: {X_train.shape}")
        print(f"Condition number: {metadata.get('condition_number', 'N/A')}")
        print(f"{'='*60}\n")
        
        dataset_results = {
            'dataset_name': dataset_name,
            'metadata': metadata,
            'optimizers': {}
        }
        
        for opt_name, opt_config in optimizer_configs.items():
            if opt_name not in OPTIMIZERS:
                print(f"Skipping {opt_name}: not available")
                continue
            
            print(f"Testing {opt_name}...")
            
            try:
                # Initialize optimizer
                optimizer_cls = OPTIMIZERS[opt_name]
                config = opt_config.copy()
                config['epochs'] = max_epochs
                config['tolerance'] = tolerance
                config['verbose'] = False
                
                optimizer = optimizer_cls(**config)
                
                # Profile training
                memory_tracker = MemoryTracker()
                
                with OptimizerProfiler() as profiler:
                    start_time = time.perf_counter()
                    
                    # Track initial memory
                    memory_tracker.track_array('X_train', X_train)
                    memory_tracker.track_array('y_train', y_train)
                    
                    # Train
                    optimizer.fit(X_train, y_train)
                    
                    end_time = time.perf_counter()
                
                # Get training history
                history = optimizer.get_history()
                
                # Compute test error
                y_pred_test = optimizer.predict(X_test)
                test_mse = np.mean((y_test - y_pred_test) ** 2)
                
                # Get profiling stats
                profile_stats = profiler.get_stats()
                memory_stats = memory_tracker.get_summary()
                
                # Store results
                dataset_results['optimizers'][opt_name] = {
                    'config': opt_config,
                    'training_time': end_time - start_time,
                    'final_train_loss': history['losses'][-1] if history.get('losses') else None,
                    'test_mse': float(test_mse),
                    'num_epochs': len(history.get('losses', [])),
                    'converged': history['losses'][-1] < tolerance if history.get('losses') else False,
                    'profile': profile_stats,
                    'memory': memory_stats,
                    'loss_history': history.get('losses', [])[-100:]  # Last 100 epochs
                }
                
                print(f"  ✓ {opt_name}: {end_time - start_time:.2f}s, "
                      f"Final loss: {history['losses'][-1]:.2e}, "
                      f"Test MSE: {test_mse:.2e}")
                
            except Exception as e:
                print(f"  ✗ {opt_name} failed: {e}")
                dataset_results['optimizers'][opt_name] = {'error': str(e)}
        
        self.results.append(dataset_results)
        return dataset_results
    
    def run_mnist_benchmark(self, n_samples: int = 10000):
        """Benchmark on MNIST dataset."""
        print("\n" + "#"*60)
        print("# MNIST BENCHMARK")
        print("#"*60)
        
        X_train, y_train, X_test, y_test, metadata = load_mnist(
            n_samples=n_samples,
            binary_classification=True
        )
        
        # Optimizer configs
        configs = {
            'SGD': {'learning_rate': 0.01},
            'Momentum': {'learning_rate': 0.01, 'momentum': 0.9},
            'Nesterov': {'learning_rate': 0.01, 'momentum': 0.9},
            'AdaGrad': {'learning_rate': 0.1},
            'RMSProp': {'learning_rate': 0.001, 'beta': 0.9},
            'Adam': {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999},
            'AdamW': {'learning_rate': 0.001, 'weight_decay': 0.01}
        }
        
        return self.benchmark_dataset(
            'MNIST',
            X_train, y_train,
            X_test, y_test,
            metadata,
            configs,
            max_epochs=500
        )
    
    def run_california_housing_benchmark(self):
        """Benchmark on California Housing dataset."""
        print("\n" + "#"*60)
        print("# CALIFORNIA HOUSING BENCHMARK")
        print("#"*60)
        
        X_train, y_train, X_test, y_test, metadata = load_california_housing()
        
        configs = {
            'SGD': {'learning_rate': 0.1},
            'Momentum': {'learning_rate': 0.1, 'momentum': 0.9},
            'Nesterov': {'learning_rate': 0.1, 'momentum': 0.9},
            'AdaGrad': {'learning_rate': 0.5},
            'RMSProp': {'learning_rate': 0.01, 'beta': 0.9},
            'Adam': {'learning_rate': 0.01, 'beta1': 0.9, 'beta2': 0.999},
            'AdamW': {'learning_rate': 0.01, 'weight_decay': 0.01}
        }
        
        return self.benchmark_dataset(
            'California Housing',
            X_train, y_train,
            X_test, y_test,
            metadata,
            configs,
            max_epochs=1000
        )
    
    def run_highdim_benchmark(self, dimensions: List[int] = [100, 1000, 5000]):
        """Benchmark on high-dimensional synthetic data."""
        print("\n" + "#"*60)
        print("# HIGH-DIMENSIONAL BENCHMARK")
        print("#"*60)
        
        for d in dimensions:
            print(f"\n--- Dimension: {d} ---")
            
            X, y, true_params, metadata = generate_highdim_regression(
                n_samples=1000,
                n_features=d,
                condition_number=50.0
            )
            
            # Train-test split
            n_train = int(0.8 * len(X))
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            
            configs = {
                'Nesterov': {'learning_rate': 0.01, 'momentum': 0.9},
                'Adam': {'learning_rate': 0.001},
                'AdamW': {'learning_rate': 0.001, 'weight_decay': 0.01}
            }
            
            self.benchmark_dataset(
                f'Synthetic-HighDim-d{d}',
                X_train, y_train,
                X_test, y_test,
                metadata,
                configs,
                max_epochs=500
            )
    
    def run_extreme_conditioning_benchmark(self, kappas: List[float] = [1e3, 1e6, 1e9]):
        """Benchmark on extreme condition number datasets."""
        print("\n" + "#"*60)
        print("# EXTREME CONDITIONING BENCHMARK")
        print("#"*60)
        
        for kappa in kappas:
            print(f"\n--- Condition Number: {kappa:.0e} ---")
            
            X, y, metadata = generate_extreme_condition_data(
                n_samples=200,
                n_features=50,
                condition_type='exponential',
                target_kappa=kappa
            )
            
            # Train-test split
            n_train = int(0.8 * len(X))
            X_train, y_train = X[:n_train], y[:n_train]
            X_test, y_test = X[n_train:], y[n_train:]
            
            configs = {
                'SGD': {'learning_rate': 0.001},
                'Adam': {'learning_rate': 0.0001},
                'AdamW': {'learning_rate': 0.0001, 'weight_decay': 0.01}
            }
            
            self.benchmark_dataset(
                f'Extreme-κ{kappa:.0e}',
                X_train, y_train,
                X_test, y_test,
                metadata,
                configs,
                max_epochs=2000,
                tolerance=1e-4
            )
    
    def save_results(self, filename: str = 'large_scale_results.json'):
        """Save results to JSON."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {filepath}")
    
    def generate_summary_report(self):
        """Generate markdown summary report."""
        report_path = self.output_dir / 'BENCHMARK_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write("# Large-Scale Benchmark Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for dataset_result in self.results:
                f.write(f"## {dataset_result['dataset_name']}\n\n")
                
                metadata = dataset_result['metadata']
                f.write(f"**Dataset Properties:**\n")
                f.write(f"- Samples: {metadata.get('n_train', 'N/A')}\n")
                f.write(f"- Features: {metadata.get('n_features', 'N/A')}\n")
                f.write(f"- Condition Number: {metadata.get('condition_number', 'N/A'):.2e}\n\n")
                
                f.write("| Optimizer | Training Time | Final Loss | Test MSE | Epochs |\n")
                f.write("|-----------|---------------|------------|----------|--------|\n")
                
                for opt_name, opt_result in dataset_result['optimizers'].items():
                    if 'error' not in opt_result:
                        f.write(f"| {opt_name} | "
                               f"{opt_result['training_time']:.2f}s | "
                               f"{opt_result['final_train_loss']:.2e} | "
                               f"{opt_result['test_mse']:.2e} | "
                               f"{opt_result['num_epochs']} |\n")
                    else:
                        f.write(f"| {opt_name} | ERROR | - | - | - |\n")
                
                f.write("\n")
        
        print(f"✓ Report saved to {report_path}")


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("PHASE 4: LARGE-SCALE BENCHMARK SUITE")
    print("="*60)
    
    benchmark = LargeScaleBenchmark()
    
    # 1. MNIST
    benchmark.run_mnist_benchmark(n_samples=10000)
    
    # 2. California Housing
    benchmark.run_california_housing_benchmark()
    
    # 3. High-dimensional
    benchmark.run_highdim_benchmark(dimensions=[100, 1000])
    
    # 4. Extreme conditioning
    benchmark.run_extreme_conditioning_benchmark(kappas=[1e3, 1e6])
    
    # Save results
    benchmark.save_results()
    benchmark.generate_summary_report()
    
    print("\n" + "="*60)
    print("✓ ALL BENCHMARKS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
