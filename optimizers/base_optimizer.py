"""
Base Optimizer Abstract Class

Defines the unified API that all optimizers must implement.
Provides common functionality like parameter tracking, convergence monitoring,
and gradient computation.

References:
- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization methods for large-scale machine learning.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import warnings


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    All optimizers must implement:
    - _initialize_state(): Initialize optimizer-specific state
    - _compute_update(): Compute parameter update from gradient
    - get_hyperparameters(): Return dict of hyperparameters
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000,
                 random_seed: Optional[int] = None, track_history: bool = True,
                 monitor_convergence: bool = False):
        """
        Parameters:
        -----------
        learning_rate : float
            Base learning rate (α or η)
        epochs : int
            Maximum number of training iterations
        random_seed : int, optional
            Random seed for reproducibility
        track_history : bool
            Whether to store training history
        monitor_convergence : bool
            Whether to enable convergence theory monitoring
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.track_history = track_history
        self.monitor_convergence = monitor_convergence
        
        # Model parameters (shared across all optimizers)
        self.W = None
        self.b = None
        
        # Training history
        self.w_history = []
        self.b_history = []
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rate_history = []  # For adaptive methods
        
        # Convergence analyzers (lazy initialization)
        self._convergence_analyzer = None
        self._stability_analyzer = None
        
        # Optimizer state (subclass-specific)
        self._state = {}
        
    def _initialize_parameters(self):
        """Initialize model parameters W and b."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.W = np.random.randn(1, 1) * 0.01
        self.b = np.random.randn(1, 1) * 0.01
        
    @abstractmethod
    def _initialize_state(self):
        """
        Initialize optimizer-specific state variables.
        
        Examples:
        - Momentum: velocity vectors
        - Adam: first and second moment estimates
        - L-BFGS: history of gradients and parameters
        """
        pass
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.
        
        J(θ) = (1/2m)||Xθ - y||²
        """
        y_pred = X @ self.W + self.b
        error = y_pred - y
        loss = np.mean(error**2)
        return loss
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray,
                          n_samples: int) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of MSE loss.
        
        ∇_W J = (2/m) X^T (Xθ - y)
        ∇_b J = (2/m) Σ(Xθ - y)
        """
        y_pred = X @ self.W + self.b
        error = y_pred - y
        
        grad_W = (2/n_samples) * X.T @ error
        grad_b = (2/n_samples) * np.sum(error)
        
        return grad_W, grad_b
    
    @abstractmethod
    def _compute_update(self, grad_W: np.ndarray, grad_b: float,
                       step: int) -> Tuple[np.ndarray, float]:
        """
        Compute parameter updates from gradients.
        
        Different optimizers use different update rules:
        - SGD: Δθ = -α∇J
        - Momentum: Δθ = β*v - α∇J
        - Adam: Δθ = -α*m̂/(√v̂ + ε)
        
        Parameters:
        -----------
        grad_W : ndarray
            Gradient w.r.t. weight
        grad_b : float
            Gradient w.r.t. bias
        step : int
            Current iteration number
            
        Returns:
        --------
        update_W : ndarray
            Update for weight parameter
        update_b : float
            Update for bias parameter
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict:
        """
        Return dictionary of optimizer hyperparameters.
        
        Used for logging and comparison across optimizers.
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'BaseOptimizer':
        """
        Train model using the optimizer.
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Feature matrix
        y : ndarray, shape (m, 1)
            Target vector
        verbose : bool
            Print training progress
            
        Returns:
        --------
        self : BaseOptimizer
            Trained optimizer instance
        """
        # Initialize convergence monitoring if enabled
        if self.monitor_convergence:
            from theory.convergence_proof import ConvergenceAnalyzer
            from theory.numerical_stability import NumericalStabilityAnalyzer
            
            self._convergence_analyzer = ConvergenceAnalyzer(X, y)
            self._stability_analyzer = NumericalStabilityAnalyzer(dtype=X.dtype)
            
            if verbose:
                print("\n" + "="*70)
                print(f"THEORETICAL ANALYSIS - {self.__class__.__name__}")
                print("="*70)
                self._convergence_analyzer.print_analysis()
                
                # Print optimizer-specific hyperparameters
                print("\n[Optimizer Hyperparameters]")
                for key, value in self.get_hyperparameters().items():
                    print(f"  {key}: {value}")
                print("="*70 + "\n")
        
        # Initialize parameters and state
        self._initialize_parameters()
        self._initialize_state()
        n_samples = len(X)
        
        # Reset history
        if self.track_history:
            self.w_history = []
            self.b_history = []
            self.loss_history = []
            self.gradient_norms = []
            self.learning_rate_history = []
        
        # Training loop
        for step in range(self.epochs + 1):
            # Track current parameters
            if self.track_history:
                self.w_history.append(self.W.item())
                self.b_history.append(self.b.item())
            
            # Compute and track loss
            loss = self._compute_loss(X, y)
            if self.track_history:
                self.loss_history.append(loss)
            
            # Monitor numerical stability
            if self.monitor_convergence and self._stability_analyzer:
                self._stability_analyzer.monitor_loss(loss, step)
            
            # Print progress
            if verbose and step % 100 == 0:
                print(f"Step {step:4d} | Loss: {loss:.6f} | "
                      f"W: {self.W.item():.4f} | b: {self.b.item():.4f}")
            
            # Final step: no update
            if step == self.epochs:
                break
            
            # Compute gradients
            grad_W, grad_b = self._compute_gradients(X, y, n_samples)
            
            # Track gradient norm
            grad_norm = np.sqrt(np.sum(grad_W**2) + grad_b**2)
            if self.track_history:
                self.gradient_norms.append(grad_norm)
            
            # Monitor gradient stability
            if self.monitor_convergence and self._stability_analyzer:
                gradient_vector = np.array([[grad_W.item()], [grad_b]])
                self._stability_analyzer.monitor_gradient(gradient_vector, step)
            
            # Store old parameters for monitoring
            if self.monitor_convergence:
                theta_old = np.array([[self.W.item()], [self.b.item()]])
            
            # Compute updates (optimizer-specific)
            update_W, update_b = self._compute_update(grad_W, grad_b, step)
            
            # Apply updates
            self.W += update_W
            self.b += update_b
            
            # Monitor parameter update
            if self.monitor_convergence and self._stability_analyzer:
                theta_new = np.array([[self.W.item()], [self.b.item()]])
                self._stability_analyzer.monitor_parameter_update(
                    theta_old, theta_new, step, self.learning_rate
                )
        
        # Final summary
        if verbose:
            print("\n" + "="*50)
            print("Training Complete!")
            print("="*50)
            print(f"Final W: {self.W.item():.4f}")
            print(f"Final b: {self.b.item():.4f}")
            print(f"Final Loss: {loss:.6f}")
            print("="*50)
            
            if self.monitor_convergence and self._stability_analyzer:
                print("\n")
                self._stability_analyzer.print_stability_report()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.W is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        return X @ self.W + self.b
    
    def get_parameters(self) -> Dict:
        """Return current model parameters."""
        return {
            'W': self.W.item() if self.W is not None else None,
            'b': self.b.item() if self.b is not None else None
        }
    
    def get_history(self) -> Dict:
        """Return training history."""
        return {
            'w_history': self.w_history,
            'b_history': self.b_history,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms,
            'learning_rate_history': self.learning_rate_history
        }
    
    def get_convergence_analyzer(self):
        """Get convergence analyzer (if monitoring enabled)."""
        return self._convergence_analyzer
    
    def get_stability_analyzer(self):
        """Get stability analyzer (if monitoring enabled)."""
        return self._stability_analyzer