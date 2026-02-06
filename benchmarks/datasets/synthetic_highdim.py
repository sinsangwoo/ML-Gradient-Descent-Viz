"""
Synthetic High-Dimensional Regression
======================================

Generates regression problems with d = 100 to 10,000 features.
Useful for testing scalability and memory efficiency.

Theoretical Properties:
- Controllable condition number
- Sparse or dense design matrices
- Known ground truth for validation
"""

import numpy as np
from typing import Tuple, Optional


def generate_highdim_regression(
    n_samples: int = 1000,
    n_features: int = 1000,
    n_informative: Optional[int] = None,
    condition_number: float = 10.0,
    noise_std: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate high-dimensional regression problem.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features (d = 100 to 10,000).
    n_informative : int, optional
        Number of informative features. If None, all features are informative.
    condition_number : float
        Target condition number κ = λ_max / λ_min.
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed.
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features + 1)
        Design matrix with bias term.
    y : np.ndarray, shape (n_samples,)
        Target values.
    true_params : np.ndarray, shape (n_features + 1,)
        Ground truth parameters.
    metadata : dict
        Dataset properties.
    """
    np.random.seed(seed)
    
    if n_informative is None:
        n_informative = n_features
    
    # Generate covariance matrix with specified condition number
    # Σ = Q Λ Q^T where Λ has eigenvalues from λ_min to λ_max
    Q = np.linalg.qr(np.random.randn(n_features, n_features))[0]
    
    lambda_max = condition_number
    lambda_min = 1.0
    eigenvalues = np.linspace(lambda_min, lambda_max, n_features)
    
    Sigma = Q @ np.diag(eigenvalues) @ Q.T
    
    # Generate features X ~ N(0, Σ)
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=Sigma,
        size=n_samples
    )
    
    # Generate true parameters (sparse)
    true_W = np.zeros(n_features)
    true_W[:n_informative] = np.random.randn(n_informative) * 2.0
    true_b = np.random.randn() * 5.0
    
    # Generate targets
    y = X @ true_W + true_b + np.random.randn(n_samples) * noise_std
    
    # Add bias column
    X = np.column_stack([np.ones(n_samples), X])
    true_params = np.concatenate([[true_b], true_W])
    
    # Compute actual condition number
    XtX = X.T @ X / n_samples
    eigs = np.linalg.eigvalsh(XtX)
    actual_kappa = eigs.max() / (eigs.min() + 1e-10)
    
    metadata = {
        'name': f'Synthetic-HighDim-d{n_features}',
        'n_samples': n_samples,
        'n_features': n_features + 1,  # Including bias
        'n_informative': n_informative,
        'target_condition_number': condition_number,
        'actual_condition_number': float(actual_kappa),
        'noise_std': noise_std,
        'max_eigenvalue': float(eigs.max()),
        'min_eigenvalue': float(eigs.min())
    }
    
    print(f"✓ High-dim dataset generated: {metadata}")
    
    return X, y, true_params, metadata


if __name__ == '__main__':
    # Test various dimensions
    for d in [100, 1000, 5000]:
        X, y, params, meta = generate_highdim_regression(
            n_samples=1000,
            n_features=d,
            condition_number=50.0
        )
        print(f"d={d}: shape={X.shape}, κ={meta['actual_condition_number']:.2f}")
