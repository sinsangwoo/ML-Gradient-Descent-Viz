"""
Extreme Condition Number Datasets
==================================

Generates regression problems with extreme ill-conditioning (κ → 10^6+).
Tests numerical stability and optimizer robustness.

Cases:
1. Near-singular matrices (κ ≈ 10^8)
2. Exponentially decaying eigenvalues
3. Random sparse matrices with few large eigenvalues
"""

import numpy as np
from typing import Tuple, Literal


def generate_extreme_condition_data(
    n_samples: int = 100,
    n_features: int = 50,
    condition_type: Literal['exponential', 'near_singular', 'sparse_extreme'] = 'exponential',
    target_kappa: float = 1e6,
    noise_std: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate extremely ill-conditioned regression problems.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    condition_type : str
        Type of conditioning:
        - 'exponential': λ_i = λ_max * exp(-i * log(κ) / d)
        - 'near_singular': Few very small eigenvalues
        - 'sparse_extreme': Random sparse with extreme values
    target_kappa : float
        Target condition number (10^6 to 10^12).
    noise_std : float
        Noise level.
    seed : int
        Random seed.
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features + 1)
    y : np.ndarray, shape (n_samples,)
    metadata : dict
    """
    np.random.seed(seed)
    
    # Generate orthogonal matrix
    Q = np.linalg.qr(np.random.randn(n_features, n_features))[0]
    
    # Generate eigenvalues based on condition type
    if condition_type == 'exponential':
        # Exponentially decaying: λ_i = λ_max * exp(-α * i)
        alpha = np.log(target_kappa) / (n_features - 1)
        eigenvalues = np.exp(-alpha * np.arange(n_features))
        eigenvalues = eigenvalues / eigenvalues[0]  # Normalize to [1, 1/κ]
        
    elif condition_type == 'near_singular':
        # Most eigenvalues ≈ 1, few very small
        eigenvalues = np.ones(n_features)
        n_small = max(1, n_features // 10)
        eigenvalues[-n_small:] = np.linspace(1.0, 1.0 / target_kappa, n_small)
        
    elif condition_type == 'sparse_extreme':
        # Random sparse: 90% near 1, 10% near 1/κ
        eigenvalues = np.ones(n_features)
        n_extreme = max(1, n_features // 10)
        extreme_indices = np.random.choice(n_features, n_extreme, replace=False)
        eigenvalues[extreme_indices] = np.random.uniform(1.0 / target_kappa, 1.0 / np.sqrt(target_kappa), n_extreme)
    
    else:
        raise ValueError(f"Unknown condition_type: {condition_type}")
    
    # Construct covariance matrix
    Sigma = Q @ np.diag(eigenvalues) @ Q.T
    
    # Generate data
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=Sigma,
        size=n_samples
    )
    
    # True parameters
    true_W = np.random.randn(n_features) * 2.0
    true_b = np.random.randn() * 5.0
    
    # Generate targets
    y = X @ true_W + true_b + np.random.randn(n_samples) * noise_std
    
    # Add bias
    X = np.column_stack([np.ones(n_samples), X])
    
    # Compute actual condition number
    XtX = X.T @ X / n_samples
    eigs = np.linalg.eigvalsh(XtX)
    actual_kappa = eigs.max() / (eigs.min() + 1e-10)
    
    metadata = {
        'name': f'Extreme-{condition_type}-κ{target_kappa:.0e}',
        'n_samples': n_samples,
        'n_features': n_features + 1,
        'condition_type': condition_type,
        'target_kappa': target_kappa,
        'actual_kappa': float(actual_kappa),
        'max_eigenvalue': float(eigs.max()),
        'min_eigenvalue': float(eigs.min()),
        'eigenvalue_ratio': f"{eigs.max() / eigs.min():.2e}"
    }
    
    print(f"✓ Extreme conditioning dataset: {metadata}")
    
    return X, y, metadata


if __name__ == '__main__':
    # Test extreme cases
    for kappa in [1e3, 1e6, 1e9]:
        X, y, meta = generate_extreme_condition_data(
            target_kappa=kappa,
            condition_type='exponential'
        )
        print(f"κ_target={kappa:.0e}, κ_actual={meta['actual_kappa']:.2e}")
