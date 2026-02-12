"""
Gradient Descent Visualization Main Entry Point
"""

from .data.generator import LinearDataGenerator
from .optimizers.sgd import SGD
from .visualization.visualizer import GradientDescentVisualizer
import numpy as np

def main():
    """Main execution function"""
    
    print("="*60)
    print("Gradient Descent Linear Regression Visualization")
    print("="*60)
    
    # ========================================
    # 1. Data Generation
    # ========================================
    print("\n[Step 1] Generating Data...")
    print("-"*60)
    
    # True parameters
    W_true = 2
    b_true = 5
    
    # Initialize generator
    data_gen = LinearDataGenerator(W_true=W_true, b_true=b_true, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    print(f"✓ Data generation complete")
    print(f"  - Samples: {len(X)}")
    print(f"  - X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  - y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  - True parameters: W={W_true}, b={b_true}")
    
    # ========================================
    # 2. Model Training
    # ========================================
    print("\n[Step 2] Training Model (SGD)...")
    print("-"*60)
    
    # Hyperparameters
    learning_rate = 0.1
    epochs = 1000
    
    # Create and train model
    # Note using SGD optimizer which effectively acts as the regressor in this context
    model = SGD(
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=42,
        monitor_convergence=True  # Enable new features
    )
    
    print(f"Hyperparameters:")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Epochs: {epochs}")
    print()
    
    model.fit(X, y, verbose=True)
    
    # Check results
    params = model.get_parameters()
    print(f"\nTraining Results:")
    print(f"  True W: {W_true:.4f} | Learned W: {params['W']:.4f} | Error: {abs(W_true - params['W']):.4f}")
    print(f"  True b: {b_true:.4f} | Learned b: {params['b']:.4f} | Error: {abs(b_true - params['b']):.4f}")
    
    # ========================================
    # 3. Visualization
    # ========================================
    print("\n[Step 3] Visualizing Results...")
    print("-"*60)
    
    # Create visualizer
    visualizer = GradientDescentVisualizer()
    
    # Show all plots
    visualizer.visualize_all(X, y, model, W_true, b_true)
    
    # ========================================
    # 4. Prediction Test
    # ========================================
    print("\n[Step 4] Prediction Test...")
    print("-"*60)
    
    X_test = np.array([[0.5], [1.0], [1.5], [2.0]])
    y_pred = model.predict(X_test)
    
    print("Predictions on new data:")
    for x, y_p in zip(X_test, y_pred):
        y_true_calc = b_true + W_true * x[0]
        print(f"  X={x[0]:.1f} -> Pred: {y_p[0]:.3f} | True: {y_true_calc:.3f} | Error: {abs(y_p[0] - y_true_calc):.3f}")
    
    print("\n" + "="*60)
    print("Program execution complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
