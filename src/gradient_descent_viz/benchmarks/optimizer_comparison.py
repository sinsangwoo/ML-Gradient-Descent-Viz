"""
Comprehensive Optimizer Benchmark Suite

Compares all implemented optimizers on:
1. Convergence speed
2. Final accuracy
3. Stability (gradient/loss variance)
4. Hyperparameter sensitivity
5. Condition number robustness

Generates publication-quality comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers import SGD, MomentumSGD, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW
from data_generator import LinearDataGenerator
from theory.convergence_proof import ConvergenceAnalyzer


class OptimizerBenchmark:
    """
    Comprehensive benchmark suite for optimizer comparison.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results = {}
        
    def benchmark_convergence_speed(self, epochs: int = 500) -> Dict:
        """
        Benchmark 1: Compare convergence speed on standard problem.
        
        Metrics:
        - Iterations to 1e-3 loss
        - Iterations to 1e-6 loss
        - Final loss at fixed epochs
        """
        print("\n" + "="*70)
        print("BENCHMARK 1: Convergence Speed")
        print("="*70)
        
        # Generate data
        np.random.seed(self.seed)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=self.seed)
        X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
        
        # Define optimizers with tuned hyperparameters
        optimizers = [
            ('SGD', SGD(learning_rate=0.1, epochs=epochs, random_seed=self.seed)),
            ('Momentum', MomentumSGD(learning_rate=0.1, momentum=0.9, epochs=epochs, random_seed=self.seed)),
            ('Nesterov', NesterovMomentum(learning_rate=0.1, momentum=0.9, epochs=epochs, random_seed=self.seed)),
            ('AdaGrad', AdaGrad(learning_rate=0.5, epochs=epochs, random_seed=self.seed)),
            ('RMSProp', RMSProp(learning_rate=0.01, beta=0.9, epochs=epochs, random_seed=self.seed)),
            ('Adam', Adam(learning_rate=0.01, epochs=epochs, random_seed=self.seed)),
            ('AdamW', AdamW(learning_rate=0.01, weight_decay=0.01, epochs=epochs, random_seed=self.seed))
        ]
        
        results = {}
        
        for name, opt in optimizers:
            print(f"\nTraining {name}...")
            opt.fit(X, y, verbose=False)
            
            history = opt.get_history()
            loss_hist = np.array(history['loss_history'])
            
            # Find iterations to thresholds
            iter_to_1e3 = np.where(loss_hist < 1e-3)[0]
            iter_to_1e6 = np.where(loss_hist < 1e-6)[0]
            
            results[name] = {
                'loss_history': loss_hist,
                'final_loss': loss_hist[-1],
                'iter_to_1e-3': iter_to_1e3[0] if len(iter_to_1e3) > 0 else epochs,
                'iter_to_1e-6': iter_to_1e6[0] if len(iter_to_1e6) > 0 else epochs,
                'gradient_norms': history['gradient_norms']
            }
            
            print(f"  Final loss: {results[name]['final_loss']:.6e}")
            print(f"  Iter to 10⁻³: {results[name]['iter_to_1e-3']}")
            print(f"  Iter to 10⁻⁶: {results[name]['iter_to_1e-6']}")
        
        self.results['convergence_speed'] = results
        return results
    
    def benchmark_ill_conditioned(self, epochs: int = 1000) -> Dict:
        """
        Benchmark 2: Robustness to ill-conditioning.
        
        Test on problems with varying condition numbers:
        - κ ≈ 1 (well-conditioned)
        - κ ≈ 100 (moderately ill-conditioned)
        - κ ≈ 1000 (severely ill-conditioned)
        """
        print("\n" + "="*70)
        print("BENCHMARK 2: Ill-Conditioning Robustness")
        print("="*70)
        
        # Create datasets with different condition numbers
        datasets = []
        
        # Well-conditioned
        np.random.seed(self.seed)
        X1 = np.random.randn(100, 1)
        y1 = 2 * X1 + 5 + 0.1 * np.random.randn(100, 1)
        datasets.append(('Well (\u03ba≈1)', X1, y1))
        
        # Moderately ill-conditioned
        X2 = np.random.randn(100, 1) * 5
        y2 = 2 * X2 + 5 + 2.0 * np.random.randn(100, 1)
        datasets.append(('Moderate (\u03ba≈100)', X2, y2))
        
        # Severely ill-conditioned
        X3 = np.random.randn(100, 1) * 20
        y3 = 2 * X3 + 5 + 10.0 * np.random.randn(100, 1)
        datasets.append(('Severe (\u03ba≈1000)', X3, y3))
        
        results = {}
        
        for dataset_name, X, y in datasets:
            print(f"\n{dataset_name}:")
            analyzer = ConvergenceAnalyzer(X, y)
            kappa = analyzer.compute_condition_number()
            print(f"  Actual κ = {kappa:.2f}")
            
            dataset_results = {}
            
            # Test adaptive methods (should be robust)
            optimizers = [
                ('SGD', SGD(learning_rate=0.01, epochs=epochs, random_seed=self.seed)),
                ('Adam', Adam(learning_rate=0.01, epochs=epochs, random_seed=self.seed)),
                ('AdamW', AdamW(learning_rate=0.01, epochs=epochs, random_seed=self.seed))
            ]
            
            for name, opt in optimizers:
                opt.fit(X, y, verbose=False)
                final_loss = opt.get_history()['loss_history'][-1]
                dataset_results[name] = final_loss
                print(f"  {name}: {final_loss:.6e}")
            
            results[dataset_name] = {
                'kappa': kappa,
                'optimizers': dataset_results
            }
        
        self.results['ill_conditioned'] = results
        return results
    
    def benchmark_hyperparameter_sensitivity(self) -> Dict:
        """
        Benchmark 3: Sensitivity to hyperparameters.
        
        Test:
        - Learning rate sensitivity
        - Momentum sensitivity (for Momentum/Nesterov)
        - Beta sensitivity (for adaptive methods)
        """
        print("\n" + "="*70)
        print("BENCHMARK 3: Hyperparameter Sensitivity")
        print("="*70)
        
        np.random.seed(self.seed)
        data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=self.seed)
        X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
        
        epochs = 300
        results = {}
        
        # SGD: Learning rate sensitivity
        print("\nSGD Learning Rate Sweep:")
        learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
        sgd_results = {}
        for lr in learning_rates:
            opt = SGD(learning_rate=lr, epochs=epochs, random_seed=self.seed)
            opt.fit(X, y, verbose=False)
            final_loss = opt.get_history()['loss_history'][-1]
            sgd_results[lr] = final_loss
            print(f"  α={lr:.3f}: {final_loss:.6e}")
        results['SGD_lr_sweep'] = sgd_results
        
        # Adam: Beta2 sensitivity
        print("\nAdam Beta2 Sweep:")
        beta2_values = [0.9, 0.99, 0.999, 0.9999]
        adam_results = {}
        for beta2 in beta2_values:
            opt = Adam(learning_rate=0.01, beta2=beta2, epochs=epochs, random_seed=self.seed)
            opt.fit(X, y, verbose=False)
            final_loss = opt.get_history()['loss_history'][-1]
            adam_results[beta2] = final_loss
            print(f"  β₂={beta2:.4f}: {final_loss:.6e}")
        results['Adam_beta2_sweep'] = adam_results
        
        self.results['hyperparam_sensitivity'] = results
        return results
    
    def visualize_convergence_comparison(self, save_path: str = 'optimizer_comparison.png'):
        """
        Generate comprehensive comparison visualization.
        """
        if 'convergence_speed' not in self.results:
            print("Run benchmark_convergence_speed() first!")
            return
        
        results = self.results['convergence_speed']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss convergence (log scale)
        ax = axes[0, 0]
        for name, data in results.items():
            ax.plot(data['loss_history'], label=name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Final loss comparison (bar chart)
        ax = axes[0, 1]
        names = list(results.keys())
        final_losses = [results[name]['final_loss'] for name in names]
        bars = ax.bar(names, final_losses, color='steelblue', alpha=0.7)
        ax.set_ylabel('Final Loss', fontsize=12)
        ax.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annotate bars
        for bar, loss in zip(bars, final_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{loss:.2e}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Iterations to threshold
        ax = axes[1, 0]
        iter_1e3 = [results[name]['iter_to_1e-3'] for name in names]
        iter_1e6 = [results[name]['iter_to_1e-6'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, iter_1e3, width, label='To 10⁻³', alpha=0.7)
        ax.bar(x + width/2, iter_1e6, width, label='To 10⁻⁶', alpha=0.7)
        ax.set_xlabel('Optimizer', fontsize=12)
        ax.set_ylabel('Iterations', fontsize=12)
        ax.set_title('Convergence Speed to Threshold', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Gradient norm evolution
        ax = axes[1, 1]
        for name, data in results.items():
            if 'gradient_norms' in data and len(data['gradient_norms']) > 0:
                ax.plot(data['gradient_norms'], label=name, linewidth=2, alpha=0.8)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Magnitude Evolution', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to '{save_path}'")
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate comprehensive text report.
        """
        report = []
        report.append("="*70)
        report.append("OPTIMIZER BENCHMARK REPORT")
        report.append("="*70)
        
        # Convergence speed summary
        if 'convergence_speed' in self.results:
            report.append("\n[1] CONVERGENCE SPEED")
            report.append("-"*70)
            results = self.results['convergence_speed']
            
            # Rank by final loss
            ranked = sorted(results.items(), key=lambda x: x[1]['final_loss'])
            report.append("\nRanking by Final Loss:")
            for rank, (name, data) in enumerate(ranked, 1):
                report.append(f"  {rank}. {name:12s} - {data['final_loss']:.6e}")
            
            # Rank by speed to 1e-6
            ranked_speed = sorted(results.items(), key=lambda x: x[1]['iter_to_1e-6'])
            report.append("\nRanking by Speed (iterations to 10⁻⁶):")
            for rank, (name, data) in enumerate(ranked_speed, 1):
                iters = data['iter_to_1e-6']
                if iters < 10000:
                    report.append(f"  {rank}. {name:12s} - {iters} iterations")
                else:
                    report.append(f"  {rank}. {name:12s} - Did not reach 10⁻⁶")
        
        # Ill-conditioning robustness
        if 'ill_conditioned' in self.results:
            report.append("\n[2] ILL-CONDITIONING ROBUSTNESS")
            report.append("-"*70)
            for dataset_name, data in self.results['ill_conditioned'].items():
                report.append(f"\n{dataset_name} (κ={data['kappa']:.2f}):")
                for opt_name, loss in data['optimizers'].items():
                    report.append(f"  {opt_name:12s}: {loss:.6e}")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        return "\n".join(report)


def main():
    """Run complete benchmark suite."""
    print("\n" + "#"*70)
    print("#" + " "*16 + "OPTIMIZER BENCHMARK SUITE" + " "*17 + "#")
    print("#"*70)
    
    benchmark = OptimizerBenchmark(seed=42)
    
    # Run benchmarks
    benchmark.benchmark_convergence_speed(epochs=500)
    benchmark.benchmark_ill_conditioned(epochs=1000)
    benchmark.benchmark_hyperparameter_sensitivity()
    
    # Generate visualizations
    benchmark.visualize_convergence_comparison()
    
    # Print report
    report = benchmark.generate_report()
    print("\n" + report)
    
    # Save report
    with open('benchmark_report.txt', 'w') as f:
        f.write(report)
    print("\n✓ Saved detailed report to 'benchmark_report.txt'")
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "BENCHMARKS COMPLETE" + " "*21 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()