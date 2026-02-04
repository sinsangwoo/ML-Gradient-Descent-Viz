"""
Two-Layer Neural Network - Deep Non-Convex Model

Architecture:
Input -> Hidden(n_hidden) -> Output(1)

Activations supported:
- ReLU: max(0, x)
- Tanh: (e^x - e^-x)/(e^x + e^-x)  
- Sigmoid: 1/(1 + e^-x)

Loss landscape properties:
- Multiple local minima
- Saddle points at every layer
- Non-convex with complex geometry

References:
- Goodfellow et al. (2016). Deep Learning.
- Kawaguchi (2016). Deep learning without poor local minima.
"""

import numpy as np
from typing import Tuple, Dict, Callable


class TwoLayerNet:
    """
    Two-layer feedforward neural network for regression.
    
    Architecture:
    x -> W1, b1 -> activation -> W2, b2 -> y
    
    Parameters:
    - W1: (n_hidden, n_input)
    - b1: (n_hidden, 1)
    - W2: (1, n_hidden)
    - b2: (1, 1)
    """
    
    def __init__(self, n_hidden: int = 10, activation: str = 'relu',
                 random_seed: int = None):
        """
        Parameters:
        -----------
        n_hidden : int
            Number of hidden units
        activation : str
            Activation function ('relu', 'tanh', 'sigmoid')
        random_seed : int
            For reproducible initialization
        """
        self.n_hidden = n_hidden
        self.activation_name = activation
        self.random_seed = random_seed
        
        # Parameters
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        # Cache for backprop
        self._cache = {}
        
        # Training history
        self.loss_history = []
        self.gradient_norms = []
        
        # Set activation function
        self._set_activation()
    
    def _set_activation(self):
        """Set activation and its derivative."""
        if self.activation_name == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: (x > 0).astype(float)
        
        elif self.activation_name == 'tanh':
            self.activation = lambda x: np.tanh(x)
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        
        elif self.activation_name == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            self.activation_derivative = lambda x: (
                self.activation(x) * (1 - self.activation(x))
            )
        
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def _initialize_parameters(self, n_input: int):
        """
        Initialize weights using Xavier/He initialization.
        
        Xavier (tanh/sigmoid): std = sqrt(2 / (n_in + n_out))
        He (ReLU): std = sqrt(2 / n_in)
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if self.activation_name == 'relu':
            # He initialization
            self.W1 = np.random.randn(self.n_hidden, n_input) * np.sqrt(2 / n_input)
            self.W2 = np.random.randn(1, self.n_hidden) * np.sqrt(2 / self.n_hidden)
        else:
            # Xavier initialization  
            self.W1 = np.random.randn(self.n_hidden, n_input) * np.sqrt(2 / (n_input + self.n_hidden))
            self.W2 = np.random.randn(1, self.n_hidden) * np.sqrt(2 / (self.n_hidden + 1))
        
        self.b1 = np.zeros((self.n_hidden, 1))
        self.b2 = np.zeros((1, 1))
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        X: (m, n_input)
        Z1 = W1 @ X.T + b1  # (n_hidden, m)
        A1 = activation(Z1)   # (n_hidden, m)
        Z2 = W2 @ A1 + b2     # (1, m)
        Y = Z2                # (1, m)
        """
        m = X.shape[0]
        
        # Layer 1
        Z1 = self.W1 @ X.T + self.b1
        A1 = self.activation(Z1)
        
        # Layer 2 (output)
        Z2 = self.W2 @ A1 + self.b2
        
        # Cache for backprop
        self._cache = {
            'X': X,
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'm': m
        }
        
        return Z2.T  # (m, 1)
    
    def _backward(self, y: np.ndarray) -> Dict:
        """
        Backward pass (compute gradients).
        
        Returns gradients: dW1, db1, dW2, db2
        """
        X = self._cache['X']
        Z1 = self._cache['Z1']
        A1 = self._cache['A1']
        Z2 = self._cache['Z2']
        m = self._cache['m']
        
        # Output layer gradient
        dZ2 = (Z2 - y.T) / m  # (1, m)
        
        dW2 = dZ2 @ A1.T      # (1, n_hidden)
        db2 = np.sum(dZ2, axis=1, keepdims=True)  # (1, 1)
        
        # Hidden layer gradient
        dA1 = self.W2.T @ dZ2  # (n_hidden, m)
        dZ1 = dA1 * self.activation_derivative(Z1)  # (n_hidden, m)
        
        dW1 = dZ1 @ X          # (n_hidden, n_input)
        db1 = np.sum(dZ1, axis=1, keepdims=True)  # (n_hidden, 1)
        
        return {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
    
    def _compute_loss(self, y_pred: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss."""
        error = y_pred.flatten() - y.flatten()
        loss = 0.5 * np.mean(error ** 2)
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.01, epochs: int = 1000,
            verbose: bool = True) -> 'TwoLayerNet':
        """
        Train neural network using gradient descent.
        """
        n_input = X.shape[1]
        
        # Initialize parameters
        self._initialize_parameters(n_input)
        
        # Reset history
        self.loss_history = []
        self.gradient_norms = []
        
        # Training loop
        for epoch in range(epochs + 1):
            # Forward pass
            y_pred = self._forward(X)
            
            # Compute loss
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
            
            if epoch == epochs:
                break
            
            # Backward pass
            grads = self._backward(y)
            
            # Compute gradient norm
            grad_norm = np.sqrt(
                np.sum(grads['dW1']**2) + np.sum(grads['db1']**2) +
                np.sum(grads['dW2']**2) + np.sum(grads['db2']**2)
            )
            self.gradient_norms.append(grad_norm)
            
            # Update parameters
            self.W1 -= learning_rate * grads['dW1']
            self.b1 -= learning_rate * grads['db1']
            self.W2 -= learning_rate * grads['dW2']
            self.b2 -= learning_rate * grads['db2']
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final loss: {loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.W1 is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self._forward(X).flatten()
    
    def get_parameters(self) -> Dict:
        """Return all parameters."""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
    
    def get_history(self) -> Dict:
        """Return training history."""
        return {
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms
        }


if __name__ == "__main__":
    # Test neural network
    print("Testing Two-Layer Neural Network\n")
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_true = np.sin(X) + 0.1 * X**2
    y = y_true.flatten() + np.random.randn(200) * 0.1
    
    # Test different activations
    for activation in ['relu', 'tanh', 'sigmoid']:
        print(f"\n{'='*50}")
        print(f"Activation: {activation}")
        print('='*50)
        
        model = TwoLayerNet(n_hidden=20, activation=activation, random_seed=42)
        model.fit(X, y, learning_rate=0.01, epochs=500, verbose=False)
        
        history = model.get_history()
        print(f"Final loss: {history['loss_history'][-1]:.6f}")
        
        # Test prediction
        X_test = np.array([[-2], [0], [2]])
        y_pred = model.predict(X_test)
        print(f"Predictions: {y_pred}")