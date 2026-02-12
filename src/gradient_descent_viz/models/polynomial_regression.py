"""
Polynomial Regression - First Non-Convex Extension

For degree d >= 2, polynomial regression exhibits:
- Multiple local minima
- Saddle points
- Non-convex loss landscape

This serves as a controlled testbed for non-convex optimization.

References:
- Ge et al. (2015). "Escaping from saddle points"
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings


class PolynomialRegressor:
    """
    Polynomial regression with arbitrary degree.
    
    Model: y = w_0 + w_1*x + w_2*x^2 + ... + w_d*x^d
    
    For d >= 2, the loss landscape is non-convex.
    """
    
    def __init__(self, degree: int = 2, random_seed: int = None):
        """
        Parameters:
        -----------
        degree : int
            Polynomial degree (d >= 1)
        random_seed : int
            For reproducible initialization
        """
        if degree < 1:
            raise ValueError("Degree must be >= 1")
        
        self.degree = degree
        self.random_seed = random_seed
        
        # Coefficients: [w_0, w_1, ..., w_d]
        self.weights = None
        
        # Training history
        self.loss_history = []
        self.gradient_norms = []
        
    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X into polynomial features [1, x, x^2, ..., x^d].
        
        Parameters:
        -----------
        X : ndarray, shape (m, 1)
            Input features
            
        Returns:
        --------
        X_poly : ndarray, shape (m, d+1)
            Polynomial features
        """
        m = X.shape[0]
        X_poly = np.ones((m, self.degree + 1))
        
        for i in range(1, self.degree + 1):
            X_poly[:, i] = (X[:, 0] ** i)
        
        return X_poly
    
    def _initialize_weights(self):
        """Initialize polynomial coefficients."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Xavier initialization scaled for polynomial degree
        self.weights = np.random.randn(self.degree + 1) * 0.01
    
    def _compute_loss(self, X_poly: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MSE loss.
        
        J(w) = (1/2m) ||X_poly @ w - y||^2
        """
        y_pred = X_poly @ self.weights
        error = y_pred - y.flatten()
        loss = 0.5 * np.mean(error ** 2)
        return loss
    
    def _compute_gradient(self, X_poly: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. weights.
        
        ∇J(w) = (1/m) X_poly^T (X_poly @ w - y)
        """
        m = X_poly.shape[0]
        y_pred = X_poly @ self.weights
        error = y_pred - y.flatten()
        gradient = (1/m) * (X_poly.T @ error)
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            learning_rate: float = 0.01, epochs: int = 1000,
            verbose: bool = True) -> 'PolynomialRegressor':
        """
        Train polynomial regression using gradient descent.
        
        Parameters:
        -----------
        X : ndarray, shape (m, 1)
            Input features
        y : ndarray, shape (m, 1) or (m,)
            Target values
        learning_rate : float
            Step size
        epochs : int
            Number of iterations
        verbose : bool
            Print training progress
        """
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        
        # Initialize weights
        self._initialize_weights()
        
        # Reset history
        self.loss_history = []
        self.gradient_norms = []
        
        # Training loop
        for epoch in range(epochs + 1):
            # Compute loss
            loss = self._compute_loss(X_poly, y)
            self.loss_history.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
            
            if epoch == epochs:
                break
            
            # Compute gradient
            gradient = self._compute_gradient(X_poly, y)
            grad_norm = np.linalg.norm(gradient)
            self.gradient_norms.append(grad_norm)
            
            # Update weights
            self.weights -= learning_rate * gradient
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final loss: {loss:.6f}")
            print(f"Coefficients: {self.weights}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : ndarray, shape (m, 1)
            Input features
            
        Returns:
        --------
        y_pred : ndarray, shape (m,)
            Predictions
        """
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.weights
    
    def get_coefficients(self) -> np.ndarray:
        """Return polynomial coefficients."""
        return self.weights
    
    def get_history(self) -> Dict:
        """Return training history."""
        return {
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms
        }


if __name__ == "__main__":
    # Test polynomial regression
    print("Testing Polynomial Regression\n")
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = 0.5 * X**3 - 2 * X**2 + X + 1
    y = y_true.flatten() + np.random.randn(100) * 2
    
    # Fit polynomial of degree 3
    model = PolynomialRegressor(degree=3, random_seed=42)
    model.fit(X, y, learning_rate=0.0001, epochs=1000, verbose=True)
    
    # Test prediction
    X_test = np.array([[-2], [0], [2]])
    y_pred = model.predict(X_test)
    
    print("\nPredictions:")
    for x, y_p in zip(X_test, y_pred):
        print(f"  x={x[0]:.1f} → y={y_p:.3f}")