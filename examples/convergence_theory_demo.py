"""
Convergence Theory Demonstration

This script demonstrates the theoretical convergence analysis capabilities:
1. Lipschitz constant computation
2. Strong convexity parameter
3. Condition number analysis
4. Optimal learning rate derivation
5. Convergence rate verification

Usage:
    python examples/convergence_theory_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gradient_descent_viz.theory.convergence_proof import ConvergenceAnalyzer
from gradient_descent_viz.optimizers.sgd import SGD as GradientDescentRegressor
from gradient_descent_viz.data.generator import LinearDataGenerator


def demo_basic_analysis():
    """Demonstrate basic convergence analysis."""
    print("="*70)
    print("DEMO 1: Basic Convergence Analysis")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    # Analyze convergence properties
    analyzer = ConvergenceAnalyzer(X, y)
    analyzer.print_analysis()
    
    return analyzer


def demo_learning_rate_comparison(analyzer):
    """Compare different learning rates."""
    print("\n" + "="*70)
    print("DEMO 2: Learning Rate Comparison")
    print("="*70)
    
    # Test various learning rates
    test_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
    
    results = []
    for lr in test_rates:
        result = analyzer.verify_convergence_guarantee(lr)
        results.append(result)
        
        status = "✓ CONVERGES" if result['converges'] else "✗ DIVERGES"
        print(f"\nη = {lr:.3f}: {status}")
        print(f"  Convergence rate ρ = {result['convergence_rate']:.6f}")
        print(f"  Stability limit: η < {result['stability_limit']:.6f}")
    
    return test_rates, results


def demo_optimal_vs_suboptimal():
    """Compare optimal vs suboptimal learning rates in practice."""
    print("\n" + "="*70)
    print("DEMO 3: Optimal vs Suboptimal Learning Rate")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    # Get optimal learning rate
    analyzer = ConvergenceAnalyzer(X, y)
    eta_opt = analyzer.compute_optimal_learning_rate()
    
    print(f"\nTheoretical optimal learning rate: η* = {eta_opt:.6f}")
    
    # Train with different rates
    learning_rates = {
        'Too Small (0.1η*)': 0.1 * eta_opt,
        'Suboptimal (0.5η*)': 0.5 * eta_opt,
        'Optimal (η*)': eta_opt,
        'Aggressive (1.5η*)': 1.5 * eta_opt
    }
    
    histories = {}
    epochs = 200
    
    for name, lr in learning_rates.items():
        print(f"\nTraining with {name} = {lr:.6f}")
        model = GradientDescentRegressor(
            learning_rate=lr,
            epochs=epochs,
            random_seed=42,
            monitor_convergence=False
        )
        model.fit(X, y, verbose=False)
        histories[name] = model.get_history()['loss_history']
    
    # Visualize convergence comparison
    plt.figure(figsize=(12, 6))
    
    for name, loss_hist in histories.items():
        plt.plot(loss_hist, label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Convergence Speed: Optimal vs Suboptimal Learning Rates', fontsize=14)
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=150)
    print("\n✓ Saved plot to 'convergence_comparison.png'")
    plt.show()


def demo_condition_number_impact():
    """Demonstrate impact of condition number on convergence."""
    print("\n" + "="*70)
    print("DEMO 4: Condition Number Impact")
    print("="*70)
    
    # Create datasets with different condition numbers
    datasets = []
    
    # Well-conditioned: κ ≈ 1
    np.random.seed(42)
    X1 = np.random.randn(100, 1)
    y1 = 2 * X1 + 5 + 0.1 * np.random.randn(100, 1)
    datasets.append(('Well-conditioned (κ≈1)', X1, y1))
    
    # Moderately conditioned: κ ≈ 10
    X2 = np.random.randn(100, 1) * 3
    y2 = 2 * X2 + 5 + 1.0 * np.random.randn(100, 1)
    datasets.append(('Moderately conditioned (κ≈10)', X2, y2))
    
    # Ill-conditioned: κ > 100 (add correlated features - for future multi-feature)
    X3 = np.random.randn(100, 1) * 10
    y3 = 2 * X3 + 5 + 5.0 * np.random.randn(100, 1)
    datasets.append(('Ill-conditioned (κ>100)', X3, y3))
    
    histories = {}
    
    for name, X, y in datasets:
        analyzer = ConvergenceAnalyzer(X, y)
        analysis = analyzer.get_full_analysis()
        
        print(f"\n{name}:")
        print(f"  Condition number κ = {analysis['condition_number']:.2f}")
        print(f"  Optimal learning rate η* = {analysis['optimal_learning_rate']:.6f}")
        print(f"  Iterations to 10⁻⁶ accuracy: {analysis['iterations_to_1e-6']}")
        
        # Train with optimal rate
        model = GradientDescentRegressor(
            learning_rate=analysis['optimal_learning_rate'],
            epochs=500,
            random_seed=42,
            monitor_convergence=False
        )
        model.fit(X, y, verbose=False)
        histories[name] = model.get_history()['loss_history']
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    for name, loss_hist in histories.items():
        plt.plot(loss_hist, label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Impact of Condition Number on Convergence Speed', fontsize=14)
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('condition_number_impact.png', dpi=150)
    print("\n✓ Saved plot to 'condition_number_impact.png'")
    plt.show()


def demo_convergence_rate_validation():
    """Validate theoretical convergence rate against empirical results."""
    print("\n" + "="*70)
    print("DEMO 5: Theoretical vs Empirical Convergence Rate")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    data_gen = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    # Analyze
    analyzer = ConvergenceAnalyzer(X, y)
    eta_opt = analyzer.compute_optimal_learning_rate()
    rho_theoretical = analyzer.compute_convergence_rate(eta_opt)
    
    print(f"\nTheoretical convergence rate ρ = {rho_theoretical:.6f}")
    print(f"This means: ||θ_k - θ*||² ≤ ρ^k ||θ_0 - θ*||²")
    
    # Train
    model = GradientDescentRegressor(
        learning_rate=eta_opt,
        epochs=300,
        random_seed=42,
        monitor_convergence=False
    )
    model.fit(X, y, verbose=False)
    
    # Compute empirical convergence rate
    loss_hist = model.get_history()['loss_history']
    
    # Fit exponential decay to loss
    # loss_k ≈ loss_∞ + C * ρ^k
    # log(loss_k - loss_∞) ≈ log(C) + k * log(ρ)
    
    loss_final = loss_hist[-1]
    loss_residual = np.array(loss_hist[10:]) - loss_final  # Skip initial transient
    loss_residual = loss_residual[loss_residual > 1e-10]  # Remove numerical zeros
    
    if len(loss_residual) > 10:
        k_vals = np.arange(len(loss_residual))
        log_residual = np.log(loss_residual)
        
        # Linear fit: log(residual) = a + b*k, where b = log(ρ)
        coeffs = np.polyfit(k_vals, log_residual, 1)
        rho_empirical = np.exp(coeffs[0])
        
        print(f"Empirical convergence rate ρ ≈ {rho_empirical:.6f}")
        print(f"Relative error: {abs(rho_theoretical - rho_empirical) / rho_theoretical * 100:.2f}%")
        
        # Visualize
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_hist, 'b-', linewidth=2, label='Empirical')
        plt.plot(loss_final + (loss_hist[10] - loss_final) * (rho_theoretical ** np.arange(len(loss_hist))), 
                'r--', linewidth=2, label='Theoretical ρ^k decay')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss Convergence', fontsize=14)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_vals, log_residual, 'bo', alpha=0.5, label='Empirical')
        plt.plot(k_vals, coeffs[1] + coeffs[0] * k_vals, 'r-', linewidth=2, 
                label=f'Fit: ρ={rho_empirical:.4f}')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('log(Loss - Loss_final)', fontsize=12)
        plt.title('Linear Convergence Rate', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('convergence_rate_validation.png', dpi=150)
        print("\n✓ Saved plot to 'convergence_rate_validation.png'")
        plt.show()
    else:
        print("Not enough data points for empirical rate estimation")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "CONVERGENCE THEORY DEMOS" + " "*20 + "#")
    print("#"*70 + "\n")
    
    # Demo 1: Basic analysis
    analyzer = demo_basic_analysis()
    
    # Demo 2: Learning rate comparison
    demo_learning_rate_comparison(analyzer)
    
    # Demo 3: Optimal vs suboptimal
    demo_optimal_vs_suboptimal()
    
    # Demo 4: Condition number impact
    demo_condition_number_impact()
    
    # Demo 5: Convergence rate validation
    demo_convergence_rate_validation()
    
    print("\n" + "#"*70)
    print("#" + " "*22 + "ALL DEMOS COMPLETED" + " "*23 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()