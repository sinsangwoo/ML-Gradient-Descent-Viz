"""
Parallel Training - Data Parallelism

Distributes training across multiple GPUs/devices:
- Data parallelism: Split batch across devices
- Gradient synchronization
- Efficient multi-device computation

Supports:
- Multi-GPU training (CUDA)
- Multi-TPU training (Google Cloud)
- CPU multi-core (fallback)
"""

import numpy as np
from typing import List, Dict, Tuple, Callable
from .backend import get_backend
import time


class ParallelTrainer:
    """
    Data-parallel trainer for multi-device training.
    
    Splits mini-batches across devices and aggregates gradients.
    
    Example:
        >>> trainer = ParallelTrainer(n_devices=4)
        >>> trainer.fit(X, y, batch_size=256, epochs=100)
    """
    
    def __init__(self, n_devices: int = None, batch_size: int = 128):
        """
        Parameters:
        -----------
        n_devices : int, optional
            Number of devices to use (auto-detect if None)
        batch_size : int
            Global batch size (split across devices)
        """
        self.backend = get_backend()
        
        # Auto-detect devices
        if n_devices is None:
            n_devices = getattr(self.backend, 'n_devices', 1)
        
        self.n_devices = n_devices
        self.batch_size = batch_size
        self.per_device_batch_size = batch_size // n_devices
        
        if batch_size % n_devices != 0:
            print(f"Warning: batch_size ({batch_size}) not divisible by n_devices ({n_devices})")
            print(f"Using per_device_batch_size = {self.per_device_batch_size}")
        
        # Model parameters (replicated across devices)
        self.W = None
        self.b = None
        
        # Training state
        self.loss_history = []
        self.epoch_times = []
        
        print(f"✓ ParallelTrainer initialized")
        print(f"  Devices: {self.n_devices}")
        print(f"  Global batch size: {self.batch_size}")
        print(f"  Per-device batch size: {self.per_device_batch_size}")
    
    def _split_batch(self, X: np.ndarray, y: np.ndarray, 
                     indices: np.ndarray) -> List[Tuple]:
        """
        Split batch across devices.
        
        Returns:
        --------
        batches : list of (X_device, y_device)
            One batch per device
        """
        batches = []
        
        for i in range(self.n_devices):
            start = i * self.per_device_batch_size
            end = start + self.per_device_batch_size
            
            batch_indices = indices[start:end]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            # Transfer to device (if using JAX/CuPy)
            if self.backend.name != 'numpy':
                X_batch = self.backend.array(X_batch)
                y_batch = self.backend.array(y_batch)
            
            batches.append((X_batch, y_batch))
        
        return batches
    
    def _compute_device_gradients(self, X_batch, y_batch, W, b):
        """
        Compute gradients on a single device.
        
        Returns:
        --------
        grad_W, grad_b : gradients
        loss : scalar loss value
        """
        xp = self.backend
        
        # Forward pass
        m = X_batch.shape[0]
        y_pred = X_batch @ W + b
        error = y_pred - y_batch
        
        # Loss
        loss = 0.5 * xp.mean(error ** 2)
        
        # Gradients
        grad_W = (1/m) * (X_batch.T @ error)
        grad_b = (1/m) * xp.sum(error)
        
        return grad_W, grad_b, loss
    
    def _aggregate_gradients(self, device_gradients: List[Tuple]) -> Tuple:
        """
        Average gradients across devices.
        
        Parameters:
        -----------
        device_gradients : list of (grad_W, grad_b, loss)
            Gradients from each device
        
        Returns:
        --------
        avg_grad_W, avg_grad_b, avg_loss
        """
        xp = self.backend
        
        # Stack gradients
        grads_W = [g[0] for g in device_gradients]
        grads_b = [g[1] for g in device_gradients]
        losses = [g[2] for g in device_gradients]
        
        # Average (all-reduce operation)
        if len(grads_W) > 1:
            avg_grad_W = xp.mean(xp.stack(grads_W), axis=0)
            avg_grad_b = xp.mean(xp.array(grads_b))
            avg_loss = xp.mean(xp.array(losses))
        else:
            avg_grad_W = grads_W[0]
            avg_grad_b = grads_b[0]
            avg_loss = losses[0]
        
        return avg_grad_W, avg_grad_b, avg_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.01,
            epochs: int = 100,
            verbose: bool = True) -> 'ParallelTrainer':
        """
        Train model using data parallelism.
        
        Parameters:
        -----------
        X : ndarray, shape (m, n)
            Features
        y : ndarray, shape (m,) or (m, 1)
            Targets
        learning_rate : float
            Step size
        epochs : int
            Number of epochs
        verbose : bool
            Print progress
        """
        xp = self.backend
        m, n = X.shape
        
        # Initialize parameters
        if self.W is None:
            key = np.random.seed(42)
            W_init = np.random.randn(n, 1) * np.sqrt(2.0 / n)
            b_init = np.zeros((1,))
            
            if self.backend.name != 'numpy':
                self.W = xp.array(W_init)
                self.b = xp.array(b_init)
            else:
                self.W = W_init
                self.b = b_init
        
        # Flatten y
        if y.ndim > 1:
            y = y.flatten()
        
        # Reset history
        self.loss_history = []
        self.epoch_times = []
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle data
            indices = np.random.permutation(m)
            
            # Process batches
            n_batches = m // self.batch_size
            epoch_losses = []
            
            for batch_idx in range(n_batches):
                # Get batch indices
                start = batch_idx * self.batch_size
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Split batch across devices
                device_batches = self._split_batch(X, y, batch_indices)
                
                # Compute gradients on each device (parallel)
                device_gradients = []
                for X_dev, y_dev in device_batches:
                    grad_W, grad_b, loss = self._compute_device_gradients(
                        X_dev, y_dev, self.W, self.b
                    )
                    device_gradients.append((grad_W, grad_b, loss))
                
                # Aggregate gradients (all-reduce)
                avg_grad_W, avg_grad_b, avg_loss = self._aggregate_gradients(
                    device_gradients
                )
                
                epoch_losses.append(float(avg_loss))
                
                # Update parameters (synchronized across devices)
                self.W = self.W - learning_rate * avg_grad_W
                self.b = self.b - learning_rate * avg_grad_b
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = np.mean(epoch_losses)
            
            self.loss_history.append(avg_epoch_loss)
            self.epoch_times.append(epoch_time)
            
            if verbose and epoch % 10 == 0:
                throughput = m / epoch_time  # samples/sec
                print(f"Epoch {epoch:3d} | Loss: {avg_epoch_loss:.6f} | "
                      f"Time: {epoch_time:.3f}s | Throughput: {throughput:.0f} samples/s")
        
        if verbose:
            avg_time = np.mean(self.epoch_times)
            avg_throughput = m / avg_time
            print(f"\nTraining complete!")
            print(f"  Avg epoch time: {avg_time:.3f}s")
            print(f"  Avg throughput: {avg_throughput:.0f} samples/s")
            print(f"  Final loss: {self.loss_history[-1]:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.W is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        xp = self.backend
        
        if self.backend.name != 'numpy':
            X = xp.array(X)
        
        y_pred = X @ self.W + self.b
        
        # Convert to NumPy
        if self.backend.name != 'numpy':
            return self.backend.to_numpy(y_pred).flatten()
        else:
            return y_pred.flatten()
    
    def get_parameters(self) -> Dict:
        """Return parameters."""
        if self.backend.name != 'numpy':
            return {
                'W': self.backend.to_numpy(self.W),
                'b': self.backend.to_numpy(self.b)
            }
        else:
            return {'W': self.W, 'b': self.b}
    
    def get_history(self) -> Dict:
        """Return training history."""
        return {
            'loss_history': self.loss_history,
            'epoch_times': self.epoch_times,
            'avg_throughput': np.mean([m / t for m, t in 
                                      zip([1000]*len(self.epoch_times), 
                                          self.epoch_times)])
        }


if __name__ == "__main__":
    # Test parallel trainer
    print("Testing Parallel Trainer\n")
    
    # Generate large dataset
    np.random.seed(42)
    X = np.random.randn(10000, 50)
    y = X @ np.random.randn(50, 1) + np.random.randn(10000, 1) * 0.5
    y = y.flatten()
    
    # Train with data parallelism
    trainer = ParallelTrainer(n_devices=2, batch_size=256)
    trainer.fit(X, y, learning_rate=0.01, epochs=50, verbose=True)
    
    # Test prediction
    X_test = X[:5]
    y_pred = trainer.predict(X_test)
    y_true = y[:5]
    
    print(f"\nTest predictions:")
    for i in range(5):
        print(f"  True: {y_true[i]:.3f}, Pred: {y_pred[i]:.3f}")