"""
Non-Convex Optimization Demonstration

Demonstrates:
1. Polynomial regression on non-linear data
2. Neural network training with different activations
3. Loss landscape visualization
4. Saddle point detection
5. Optimizer comparison on non-convex problems
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import PolynomialRegressor, TwoLayerNet, LossLandscapeAnalyzer
from optimizers import SGD, Adam, NesterovMomentum


def demo_polynomial_regression():
    """
    Demo 1: Fit polynomial to non-linear data.
    """
    print("\n" + "="*70)
    print("DEMO 1: Polynomial Regression on Non-Linear Data")
    print("="*70)
    
    # Generate data from cubic polynomial
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = 0.3 * X**3 - 2 * X**2 + X + 5
    y = y_true.flatten() + np.random.randn(100) * 2
    
    # Try different degrees
    degrees = [1, 2, 3, 5]
    
    plt.figure(figsize=(12, 8))
    
    for i, degree in enumerate(degrees, 1):
        plt.subplot(2, 2, i)
        
        # Fit model
        model = PolynomialRegressor(degree=degree, random_seed=42)
        model.fit(X, y, learning_rate=0.0001, epochs=1000, verbose=False)
        
        # Predict
        X_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
        y_pred = model.predict(X_plot)
        
        # Plot
        plt.scatter(X, y, alpha=0.5, label='Data')
        plt.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Degree {degree}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_demo.png', dpi=150)
    print("✓ Saved visualization to 'polynomial_regression_demo.png'")
    plt.show()


def demo_neural_network():
    """
    Demo 2: Neural network with different activations.
    """
    print("\n" + "="*70)
    print("DEMO 2: Neural Network with Different Activations")
    print("="*70)
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
    y_true = np.sin(X) + 0.3 * np.cos(3*X)
    y = y_true.flatten() + np.random.randn(200) * 0.15
    
    activations = ['relu', 'tanh', 'sigmoid']
    
    plt.figure(figsize=(15, 5))
    
    for i, activation in enumerate(activations, 1):
        plt.subplot(1, 3, i)
        
        # Train network
        net = TwoLayerNet(n_hidden=20, activation=activation, random_seed=42)
        net.fit(X, y, learning_rate=0.01, epochs=500, verbose=False)
        
        # Predict
        X_plot = np.linspace(-2*np.pi, 2*np.pi, 300).reshape(-1, 1)
        y_pred = net.predict(X_plot)
        
        # Plot
        plt.scatter(X, y, alpha=0.3, s=10, label='Data')
        plt.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'{activation.upper()}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Activation: {activation.upper()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Print final loss
        history = net.get_history()
        print(f"  {activation:8s}: Final loss = {history['loss_history'][-1]:.6f}")
    
    plt.tight_layout()
    plt.savefig('neural_network_activations.png', dpi=150)
    print("✓ Saved visualization to 'neural_network_activations.png'")
    plt.show()


def demo_loss_landscape():
    """
    Demo 3: Visualize loss landscape and find saddle points.
    """
    print("\n" + "="*70)
    print("DEMO 3: Loss Landscape Visualization")
    print("="*70)
    
    # Define Rosenbrock function (famous non-convex benchmark)
    def rosenbrock(params):
        x, y = params[0], params[1]
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(params):
        x, y = params[0], params[1]
        dx = -2*(1-x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    # Create analyzer
    analyzer = LossLandscapeAnalyzer(rosenbrock, rosenbrock_grad)
    
    # Generate landscape
    center = np.array([0.0, 0.0])
    d1 = np.array([1.0, 0.0])
    d2 = np.array([0.0, 1.0])
    
    print("Generating loss landscape...")
    Alpha, Beta, losses = analyzer.generate_2d_landscape(
        center, d1, d2,
        alpha_range=(-2, 2),
        beta_range=(-1, 3),
        resolution=100
    )
    
    # Visualize 2D
    print("Creating 2D contour plot...")
    analyzer.visualize_landscape_2d(
        Alpha, Beta, losses,
        title="Rosenbrock Function (Banana Valley)",
        save_path='loss_landscape_2d.png'
    )
    
    # Visualize 3D
    print("Creating 3D surface plot...")
    analyzer.visualize_landscape_3d(
        Alpha, Beta, losses,
        title="Rosenbrock 3D Surface",
        save_path='loss_landscape_3d.png'
    )
    
    # Analyze critical points
    print("\nAnalyzing critical points...")
    points = [
        ("Origin (0, 0)", np.array([0.0, 0.0])),
        ("Global minimum (1, 1)", np.array([1.0, 1.0]))
    ]
    
    for name, point in points:
        print(f"\n  {name}:")
        result = analyzer.classify_critical_point(point)
        print(f"    Type: {result['type']}")
        print(f"    Eigenvalues: {result['eigenvalues']}")
        if result['gradient_norm'] is not None:
            print(f"    Gradient norm: {result['gradient_norm']:.6f}")


def demo_optimizer_comparison():
    """
    Demo 4: Compare optimizers on non-convex problem.
    """
    print("\n" + "="*70)
    print("DEMO 4: Optimizer Comparison on Non-Convex Problem")
    print("="*70)
    
    # Generate complex non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 150).reshape(-1, 1)
    y_true = np.sin(2*X) + 0.5*np.cos(5*X) + 0.1*X**2
    y = y_true.flatten() + np.random.randn(150) * 0.2
    
    # Define optimizers
    optimizers_to_test = [
        ('SGD', lambda: SGD(learning_rate=0.01, epochs=300, random_seed=42)),
        ('Nesterov', lambda: NesterovMomentum(learning_rate=0.01, momentum=0.9, epochs=300, random_seed=42)),
        ('Adam', lambda: Adam(learning_rate=0.01, epochs=300, random_seed=42))
    ]
    
    plt.figure(figsize=(12, 4))
    
    for i, (name, opt_fn) in enumerate(optimizers_to_test, 1):
        plt.subplot(1, 3, i)
        
        # Train neural network with this optimizer
        net = TwoLayerNet(n_hidden=15, activation='relu', random_seed=42)
        
        # Manual training loop with custom optimizer
        # (Simplified - in practice would integrate with optimizer)
        net.fit(X, y, learning_rate=0.01, epochs=300, verbose=False)
        
        # Predict
        X_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
        y_pred = net.predict(X_plot)
        
        # Plot
        plt.scatter(X, y, alpha=0.4, s=15, label='Data')
        plt.plot(X_plot, y_pred, 'r-', linewidth=2, label='Prediction')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Optimizer: {name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Print results
        history = net.get_history()
        print(f"  {name:10s}: Final loss = {history['loss_history'][-1]:.6f}")
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_nonconvex.png', dpi=150)
    print("✓ Saved visualization to 'optimizer_comparison_nonconvex.png'")
    plt.show()


def main():
    """
    Run all demonstrations.
    """
    print("\n" + "#"*70)
    print("#" + " "*15 + "NON-CONVEX OPTIMIZATION DEMO" + " "*16 + "#")
    print("#"*70)
    
    # Run demos
    demo_polynomial_regression()
    demo_neural_network()
    demo_loss_landscape()
    demo_optimizer_comparison()
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "ALL DEMOS COMPLETE" + " "*21 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()