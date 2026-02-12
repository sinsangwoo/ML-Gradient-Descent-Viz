"""
Numerical Stability Analysis

This module analyzes numerical stability issues in gradient descent:
- Floating-point precision analysis (FP16, FP32, FP64)
- Catastrophic cancellation detection
- Gradient explosion/vanishing monitoring
- Numerical error accumulation tracking
- Condition number impact on precision

References:
- Higham, N.J. (2002). Accuracy and Stability of Numerical Algorithms
- Goldberg, D. (1991). What Every Computer Scientist Should Know About Floating-Point Arithmetic
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings


class NumericalStabilityAnalyzer:
    """
    Analyzes numerical stability of gradient descent implementation.
    
    Monitors:
    - Machine epsilon and roundoff errors
    - Gradient magnitude (explosion/vanishing)
    - Loss function precision
    - Parameter update precision
    """
    
    def __init__(self, dtype=np.float64):
        """
        Parameters:
        -----------
        dtype : numpy dtype
            Data type for numerical analysis (float16, float32, float64)
        """
        self.dtype = dtype
        self.eps = np.finfo(dtype).eps
        self.tiny = np.finfo(dtype).tiny
        self.max_value = np.finfo(dtype).max
        
        # History tracking
        self.gradient_norms = []
        self.loss_values = []
        self.parameter_changes = []
        self.condition_estimates = []
        
    def get_machine_epsilon_info(self) -> Dict:
        """
        Get information about machine epsilon and floating-point limits.
        
        Returns:
        --------
        info : dict
            Machine epsilon and precision information
        """
        return {
            'dtype': str(self.dtype),
            'machine_epsilon': self.eps,
            'smallest_positive': self.tiny,
            'largest_positive': self.max_value,
            'precision_bits': np.finfo(self.dtype).precision,
            'decimal_digits': int(np.floor(-np.log10(self.eps)))
        }
    
    def check_catastrophic_cancellation(self, a: float, b: float, threshold: float = 1e-8) -> bool:
        """
        Detect catastrophic cancellation in subtraction.
        
        Catastrophic cancellation occurs when subtracting two nearly equal numbers,
        leading to loss of significant digits.
        
        Example: (1.234567890 - 1.234567000) loses precision
        
        Parameters:
        -----------
        a, b : float
            Numbers being subtracted (a - b)
        threshold : float
            Relative difference threshold
            
        Returns:
        --------
        is_catastrophic : bool
            True if catastrophic cancellation detected
        """
        if abs(a) < self.tiny or abs(b) < self.tiny:
            return False
        
        relative_diff = abs(a - b) / max(abs(a), abs(b))
        return relative_diff < threshold
    
    def monitor_gradient(self, gradient: np.ndarray, step: int):
        """
        Monitor gradient for numerical issues.
        
        Checks for:
        - Gradient explosion (||∇|| >> 1)
        - Gradient vanishing (||∇|| << 1)
        - NaN or Inf values
        
        Parameters:
        -----------
        gradient : ndarray
            Gradient vector
        step : int
            Current iteration step
        """
        grad_norm = np.linalg.norm(gradient)
        self.gradient_norms.append(grad_norm)
        
        # Check for NaN/Inf
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            warnings.warn(f"Step {step}: Gradient contains NaN or Inf!")
            return
        
        # Check for explosion
        if grad_norm > 1e10:
            warnings.warn(f"Step {step}: Gradient explosion detected! ||∇|| = {grad_norm:.2e}")
        
        # Check for vanishing
        if grad_norm < 1e-10:
            warnings.warn(f"Step {step}: Gradient vanishing detected! ||∇|| = {grad_norm:.2e}")
    
    def monitor_loss(self, loss: float, step: int):
        """
        Monitor loss function for numerical precision issues.
        
        Parameters:
        -----------
        loss : float
            Current loss value
        step : int
            Current iteration step
        """
        self.loss_values.append(loss)
        
        # Check for NaN/Inf
        if np.isnan(loss) or np.isinf(loss):
            warnings.warn(f"Step {step}: Loss is NaN or Inf!")
            return
        
        # Check if loss is too close to machine epsilon
        if 0 < loss < self.eps:
            warnings.warn(f"Step {step}: Loss below machine epsilon! Loss = {loss:.2e}")
        
        # Check for loss increase (in later stages)
        if step > 10 and len(self.loss_values) >= 2:
            if self.loss_values[-1] > self.loss_values[-2]:
                warnings.warn(f"Step {step}: Loss increased! May indicate numerical instability.")
    
    def monitor_parameter_update(self, theta_old: np.ndarray, theta_new: np.ndarray, 
                                 step: int, learning_rate: float):
        """
        Monitor parameter updates for numerical precision.
        
        Parameters:
        -----------
        theta_old : ndarray
            Parameters before update
        theta_new : ndarray
            Parameters after update
        step : int
            Current iteration step
        learning_rate : float
            Learning rate used
        """
        delta = theta_new - theta_old
        delta_norm = np.linalg.norm(delta)
        theta_norm = np.linalg.norm(theta_old)
        
        self.parameter_changes.append(delta_norm)
        
        # Check if update is too small (stagnation)
        if theta_norm > 0:
            relative_change = delta_norm / theta_norm
            if relative_change < self.eps:
                warnings.warn(f"Step {step}: Parameter update below machine epsilon! "
                            f"Relative change = {relative_change:.2e}")
    
    def estimate_condition_number(self, X: np.ndarray) -> float:
        """
        Estimate condition number from data matrix.
        
        For linear regression, κ(X^T X) affects numerical stability.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
            
        Returns:
        --------
        kappa : float
            Estimated condition number
        """
        try:
            # For small matrices, compute exact condition number
            if X.shape[1] <= 10:
                H = (1/X.shape[0]) * (X.T @ X)
                eigvals = np.linalg.eigvalsh(H)
                kappa = eigvals.max() / (eigvals.min() + self.eps)
            else:
                # For larger matrices, use SVD
                _, s, _ = np.linalg.svd(X, full_matrices=False)
                kappa = s[0] / (s[-1] + self.eps)
            
            self.condition_estimates.append(kappa)
            return kappa
        except:
            warnings.warn("Failed to compute condition number")
            return np.inf
    
    def check_ill_conditioning(self, kappa: float) -> Dict[str, any]:
        """
        Check if problem is ill-conditioned based on condition number.
        
        Rules of thumb:
        - κ < 10: well-conditioned
        - 10 ≤ κ < 100: moderately conditioned
        - 100 ≤ κ < 1000: ill-conditioned
        - κ ≥ 1000: severely ill-conditioned
        
        Parameters:
        -----------
        kappa : float
            Condition number
            
        Returns:
        --------
        diagnosis : dict
            Conditioning diagnosis
        """
        if kappa < 10:
            status = "Well-conditioned"
            severity = "low"
        elif kappa < 100:
            status = "Moderately conditioned"
            severity = "medium"
        elif kappa < 1000:
            status = "Ill-conditioned"
            severity = "high"
        else:
            status = "Severely ill-conditioned"
            severity = "critical"
        
        # Estimate digits of precision lost
        digits_lost = int(np.floor(np.log10(kappa)))
        
        return {
            'condition_number': kappa,
            'status': status,
            'severity': severity,
            'estimated_digits_lost': digits_lost,
            'requires_preconditioning': kappa > 100
        }
    
    def compare_precision(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compare numerical precision across different floating-point types.
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target vector
            
        Returns:
        --------
        comparison : dict
            Precision comparison results
        """
        results = {}
        
        for dtype in [np.float16, np.float32, np.float64]:
            X_cast = X.astype(dtype)
            y_cast = y.astype(dtype)
            
            # Compute loss in different precisions
            W = 1.0
            b = 0.0
            y_pred = X_cast * W + b
            loss = np.mean((y_pred - y_cast)**2)
            
            results[str(dtype)] = {
                'loss': float(loss),
                'machine_epsilon': np.finfo(dtype).eps,
                'decimal_digits': int(np.floor(-np.log10(np.finfo(dtype).eps)))
            }
        
        return results
    
    def get_stability_report(self) -> Dict:
        """
        Generate comprehensive stability report.
        
        Returns:
        --------
        report : dict
            Complete stability analysis
        """
        report = {
            'machine_info': self.get_machine_epsilon_info(),
            'gradient_statistics': {
                'min_norm': min(self.gradient_norms) if self.gradient_norms else None,
                'max_norm': max(self.gradient_norms) if self.gradient_norms else None,
                'mean_norm': np.mean(self.gradient_norms) if self.gradient_norms else None,
                'std_norm': np.std(self.gradient_norms) if self.gradient_norms else None
            },
            'loss_statistics': {
                'min_loss': min(self.loss_values) if self.loss_values else None,
                'max_loss': max(self.loss_values) if self.loss_values else None,
                'final_loss': self.loss_values[-1] if self.loss_values else None
            },
            'parameter_change_statistics': {
                'min_change': min(self.parameter_changes) if self.parameter_changes else None,
                'max_change': max(self.parameter_changes) if self.parameter_changes else None,
                'mean_change': np.mean(self.parameter_changes) if self.parameter_changes else None
            }
        }
        
        return report
    
    def print_stability_report(self):
        """Print formatted stability report."""
        report = self.get_stability_report()
        
        print("="*70)
        print("NUMERICAL STABILITY ANALYSIS")
        print("="*70)
        
        print("\n[Machine Precision]")
        info = report['machine_info']
        print(f"  Data type:         {info['dtype']}")
        print(f"  Machine epsilon:   {info['machine_epsilon']:.2e}")
        print(f"  Decimal digits:    ~{info['decimal_digits']}")
        print(f"  Smallest positive: {info['smallest_positive']:.2e}")
        
        if self.gradient_norms:
            print("\n[Gradient Behavior]")
            gs = report['gradient_statistics']
            print(f"  Min ||∇||:  {gs['min_norm']:.6e}")
            print(f"  Max ||∇||:  {gs['max_norm']:.6e}")
            print(f"  Mean ||∇||: {gs['mean_norm']:.6e}")
            print(f"  Std ||∇||:  {gs['std_norm']:.6e}")
        
        if self.loss_values:
            print("\n[Loss Behavior]")
            ls = report['loss_statistics']
            print(f"  Min loss:   {ls['min_loss']:.6e}")
            print(f"  Max loss:   {ls['max_loss']:.6e}")
            print(f"  Final loss: {ls['final_loss']:.6e}")
        
        print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Testing NumericalStabilityAnalyzer\n")
    
    # Test different precisions
    for dtype in [np.float16, np.float32, np.float64]:
        analyzer = NumericalStabilityAnalyzer(dtype=dtype)
        print(f"\nPrecision: {dtype}")
        print("-" * 50)
        info = analyzer.get_machine_epsilon_info()
        print(f"  Machine ε: {info['machine_epsilon']:.2e}")
        print(f"  Decimal digits: ~{info['decimal_digits']}")
    
    # Test catastrophic cancellation
    print("\n" + "="*70)
    print("CATASTROPHIC CANCELLATION TEST")
    print("="*70)
    
    analyzer = NumericalStabilityAnalyzer(dtype=np.float64)
    
    test_pairs = [
        (1.234567890, 1.234567000),  # Close numbers
        (1000.0, 1.0),               # Different magnitudes
        (1.0, 1.0 + 1e-15)          # Very close
    ]
    
    for a, b in test_pairs:
        is_catastrophic = analyzer.check_catastrophic_cancellation(a, b)
        result = "⚠ DETECTED" if is_catastrophic else "✓ Safe"
        print(f"\n({a} - {b}): {result}")
        print(f"  Result: {a - b:.15f}")