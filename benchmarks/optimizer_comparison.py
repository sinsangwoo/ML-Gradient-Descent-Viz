"""
Optimizer Comparison Benchmark

Compares all implemented optimizers on the same problem:
1. Linear regression (convex)
2. Convergence speed
3. Final accuracy
4. Robustness to learning rate

Generates comprehensive comparison plots and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers import SGD, Momentum, NesterovMomentum, AdaGrad, RMSProp, Adam, AdamW
from data_generator import LinearDataGenerator


class OptimizerBenchmark:
    """Benchmark suite for optimizer comparison."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 true_params: Dict = None):
        """
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target vector
        true_params : dict, optional
            True parameters for error computation
        """
        self.X = X
        self.y = y
        self.true_params = true_params
        self.m = X.shape[0]
    
    def compute_loss(self, W: float, b: float) -> float:
        """Compute MSE loss."""
        y_pred = self.X * W + b
        return np.mean((y_pred - self.y) ** 2)
    
    def compute_gradient(self, W: float, b: float) -> np.ndarray:
        """Compute gradient of MSE loss."""
        y_pred = self.X * W + b
        error = y_pred - self.y
        
        grad_W = (2/self.m) * np.sum(self.X * error)
        grad_b = (2/self.m) * np.sum(error)
        
        return np.array([[grad_W], [grad_b]])
    
    def run_optimizer(self, optimizer, epochs: int = 500, 
                     init_params: np.ndarray = None,
                     verbose: bool = False) -> Dict:
        """
        Run single optimizer and track history.
        
        Parameters:
        -----------
        optimizer : BaseOptimizer
            Optimizer instance
        epochs : int
            Number of training iterations
        init_params : ndarray, optional
            Initial parameters [W, b]
        verbose : bool
            Print progress
            
        Returns:
        --------
        results : dict
            Training history and statistics
        """
        # Initialize parameters
        if init_params is None:
            np.random.seed(42)
            params = np.random.randn(2, 1)
        else:
            params = init_params.copy()
        
        # Reset optimizer
        optimizer.reset()
        
        # Training history
        loss_history = []
        param_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Compute loss and gradient
            W, b = params[0, 0], params[1, 0]
            loss = self.compute_loss(W, b)
            gradient = self.compute_gradient(W, b)
            
            # Store history
            loss_history.append(loss)
            param_history.append(params.copy())
            
            # Optimizer step
            if hasattr(optimizer, 'get_lookahead_params'):
                # Nesterov momentum: gradient at look-ahead point
                lookahead = optimizer.get_lookahead_params(params)
                W_la, b_la = lookahead[0, 0], lookahead[1, 0]
                gradient = self.compute_gradient(W_la, b_la)
            
            params = optimizer.step(params, gradient)
            
            # Track in optimizer
            optimizer.track_step(params, gradient, loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        # Final evaluation
        W_final, b_final = params[0, 0], params[1, 0]
        final_loss = self.compute_loss(W_final, b_final)
        
        results = {
            'optimizer': optimizer.name,
            'loss_history': loss_history,
            'param_history': param_history,
            'final_loss': final_loss,
            'final_params': {'W': W_final, 'b': b_final},
            'config': optimizer.get_config(),
            'stats': optimizer.get_statistics()
        }
        
        # Compute error if true parameters known
        if self.true_params is not None:
            W_true = self.true_params['W']
            b_true = self.true_params['b']
            results['param_error'] = {
                'W_error': abs(W_final - W_true),
                'b_error': abs(b_final - b_true)
            }
        
        return results
    
    def compare_all_optimizers(self, learning_rate: float = 0.1, 
                               epochs: int = 500,
                               verbose: bool = True) -> Dict:
        """
        Run all optimizers and compare results.
        
        Parameters:
        -----------
        learning_rate : float
            Base learning rate for all optimizers
        epochs : int
            Number of training epochs
        verbose : bool
            Print progress
            
        Returns:
        --------
        all_results : dict
            Results for all optimizers
        """
        # Define all optimizers
        optimizers = [
            SGD(learning_rate=learning_rate),
            Momentum(learning_rate=learning_rate, momentum=0.9),
            NesterovMomentum(learning_rate=learning_rate, momentum=0.9),
            AdaGrad(learning_rate=learning_rate),
            RMSProp(learning_rate=learning_rate * 10, rho=0.9),  # Higher LR for RMSProp
            Adam(learning_rate=learning_rate * 10, beta1=0.9, beta2=0.999),  # Higher LR for Adam
            AdamW(learning_rate=learning_rate * 10, beta1=0.9, beta2=0.999, weight_decay=0.01)
        ]
        
        # Same initialization for fair comparison
        np.random.seed(42)
        init_params = np.random.randn(2, 1)
        
        all_results = {}
        
        for optimizer in optimizers:
            if verbose:
                print(f"\nRunning {optimizer.name}...")
            
            results = self.run_optimizer(
                optimizer, 
                epochs=epochs,
                init_params=init_params,
                verbose=False
            )
            
            all_results[optimizer.name] = results
            
            if verbose:
                print(f"  Final loss: {results['final_loss']:.6e}")
                if 'param_error' in results:
                    print(f"  W error: {results['param_error']['W_error']:.6e}")
                    print(f"  b error: {results['param_error']['b_error']:.6e}")
        
        return all_results


def plot_convergence_comparison(results: Dict, save_path: str = None):
    """
    Plot convergence curves for all optimizers.
    
    Parameters:
    -----------
    results : dict
        Results from compare_all_optimizers
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Loss curves (log scale)
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['loss_history'], label=name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final loss comparison (bar plot)
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = plt.bar(range(len(names)), final_losses, color=colors, alpha=0.8)
    
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Final Loss', fontsize=12)
    plt.title('Final Loss Comparison', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, final_losses)):
        plt.text(bar.get_x() + bar.get_width()/2, loss, 
                f'{loss:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to {save_path}")
    
    plt.show()


def plot_parameter_trajectories(results: Dict, save_path: str = None):
    """
    Plot parameter space trajectories.
    
    Parameters:
    -----------
    results : dict
        Results from compare_all_optimizers
    save_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        param_hist = result['param_history']
        W_hist = [p[0, 0] for p in param_hist]
        b_hist = [p[1, 0] for p in param_hist]
        
        plt.plot(W_hist, b_hist, '-', label=name, linewidth=2, alpha=0.7)
        plt.plot(W_hist[0], b_hist[0], 'o', markersize=10, alpha=0.8)  # Start
        plt.plot(W_hist[-1], b_hist[-1], '*', markersize=15, alpha=0.8)  # End
    
    plt.xlabel('Weight (W)', fontsize=12)
    plt.ylabel('Bias (b)', fontsize=12)
    plt.title('Optimization Trajectories in Parameter Space', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def print_summary_table(results: Dict):
    """
    Print formatted summary table.
    
    Parameters:
    -----------
    results : dict
        Results from compare_all_optimizers
    """
    print("\n" + "="*80)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Optimizer':<20} {'Final Loss':<15} {'W Error':<12} {'b Error':<12}")
    print("-"*80)
    
    for name, result in results.items():
        final_loss = result['final_loss']
        
        if 'param_error' in result:
            w_err = result['param_error']['W_error']
            b_err = result['param_error']['b_error']
            print(f"{name:<20} {final_loss:<15.6e} {w_err:<12.6e} {b_err:<12.6e}")
        else:
            print(f"{name:<20} {final_loss:<15.6e} {'N/A':<12} {'N/A':<12}")
    
    print("="*80)
    
    # Find best optimizer
    best_name = min(results.keys(), key=lambda k: results[k]['final_loss'])
    best_loss = results[best_name]['final_loss']
    
    print(f"\n🏆 Best Optimizer: {best_name} (Final Loss: {best_loss:.6e})")
    print("="*80)


def main():
    """Run complete benchmark."""
    print("\n" + "#"*80)
    print("#" + " "*25 + "OPTIMIZER COMPARISON BENCHMARK" + " "*24 + "#")
    print("#"*80 + "\n")
    
    # Generate data
    print("[1/4] Generating data...")
    W_true, b_true = 2.0, 5.0
    data_gen = LinearDataGenerator(W_true=W_true, b_true=b_true, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    print(f"  ✓ Dataset: {len(X)} samples")
    print(f"  ✓ True parameters: W={W_true}, b={b_true}")
    
    # Create benchmark
    benchmark = OptimizerBenchmark(
        X, y, 
        true_params={'W': W_true, 'b': b_true}
    )
    
    # Run comparison
    print("\n[2/4] Running all optimizers...")
    results = benchmark.compare_all_optimizers(
        learning_rate=0.1,
        epochs=500,
        verbose=True
    )
    
    # Print summary
    print("\n[3/4] Generating summary...")
    print_summary_table(results)
    
    # Generate plots
    print("\n[4/4] Creating visualizations...")
    plot_convergence_comparison(results, save_path='optimizer_convergence.png')
    plot_parameter_trajectories(results, save_path='optimizer_trajectories.png')
    
    print("\n" + "#"*80)
    print("#" + " "*28 + "BENCHMARK COMPLETED" + " "*29 + "#")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()