"""
Dataset Factory
===============

Create datasets from configuration.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.datasets import (
    load_mnist,
    load_california_housing,
    generate_highdim_regression,
    generate_extreme_condition_data
)
from data_generator import LinearDataGenerator


def create_dataset(dataset_name: str, params: Dict[str, Any]) -> Dict:
    """
    Create dataset from configuration.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name ('synthetic', 'mnist', 'california_housing', 
        'highdim', 'extreme_conditioning').
    params : dict
        Dataset-specific parameters.
    
    Returns
    -------
    data : dict
        Dictionary with keys:
        - X_train : np.ndarray
        - y_train : np.ndarray
        - X_test : np.ndarray
        - y_test : np.ndarray
        - metadata : dict
    """
    if dataset_name == 'synthetic':
        return _create_synthetic(params)
    elif dataset_name == 'mnist':
        return _create_mnist(params)
    elif dataset_name == 'california_housing':
        return _create_california_housing(params)
    elif dataset_name == 'highdim':
        return _create_highdim(params)
    elif dataset_name == 'extreme_conditioning':
        return _create_extreme_conditioning(params)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _create_synthetic(params: Dict) -> Dict:
    """Create synthetic linear regression dataset."""
    n_samples = params.get('n_samples', 1000)
    n_features = params.get('n_features', 100)
    noise_std = params.get('noise_std', 0.1)
    test_split = params.get('test_split', 0.2)
    
    # Generate data
    W_true = np.random.randn(n_features) * 2.0
    b_true = np.random.randn() * 5.0
    
    X = np.random.randn(n_samples, n_features)
    y = X @ W_true + b_true + np.random.randn(n_samples) * noise_std
    
    # Normalize if requested
    if params.get('normalize', True):
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std
    
    # Add bias if requested
    if params.get('add_bias', True):
        X = np.column_stack([np.ones(n_samples), X])
        W_true = np.concatenate([[b_true], W_true])
    
    # Train-test split
    n_train = int(n_samples * (1 - test_split))
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'metadata': {
            'n_features': X.shape[1],
            'n_train': n_train,
            'n_test': len(test_idx),
            'true_params': W_true
        }
    }


def _create_mnist(params: Dict) -> Dict:
    """Load MNIST dataset."""
    X_train, y_train, X_test, y_test, metadata = load_mnist(
        n_samples=params.get('n_samples', None),
        binary_classification=params.get('binary_classification', True),
        normalize=params.get('normalize', True),
        add_bias=params.get('add_bias', True)
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'metadata': metadata
    }


def _create_california_housing(params: Dict) -> Dict:
    """Load California Housing dataset."""
    X_train, y_train, X_test, y_test, metadata = load_california_housing(
        n_samples=params.get('n_samples', None),
        normalize=params.get('normalize', True),
        add_bias=params.get('add_bias', True),
        test_size=params.get('test_split', 0.2)
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'metadata': metadata
    }


def _create_highdim(params: Dict) -> Dict:
    """Generate high-dimensional synthetic dataset."""
    X, y, true_params, metadata = generate_highdim_regression(
        n_samples=params.get('n_samples', 1000),
        n_features=params.get('n_features', 1000),
        n_informative=params.get('n_informative', None),
        condition_number=params.get('condition_number', 10.0),
        noise_std=params.get('noise_std', 0.1)
    )
    
    # Train-test split
    test_split = params.get('test_split', 0.2)
    n_train = int(len(X) * (1 - test_split))
    
    return {
        'X_train': X[:n_train],
        'y_train': y[:n_train],
        'X_test': X[n_train:],
        'y_test': y[n_train:],
        'metadata': metadata
    }


def _create_extreme_conditioning(params: Dict) -> Dict:
    """Generate extreme condition number dataset."""
    X, y, metadata = generate_extreme_condition_data(
        n_samples=params.get('n_samples', 200),
        n_features=params.get('n_features', 50),
        condition_type=params.get('condition_type', 'exponential'),
        target_kappa=params.get('target_kappa', 1e6),
        noise_std=params.get('noise_std', 0.01)
    )
    
    # Train-test split
    test_split = params.get('test_split', 0.2)
    n_train = int(len(X) * (1 - test_split))
    
    return {
        'X_train': X[:n_train],
        'y_train': y[:n_train],
        'X_test': X[n_train:],
        'y_test': y[n_train:],
        'metadata': metadata
    }


if __name__ == '__main__':
    # Test dataset factory
    print("Testing dataset factory...\n")
    
    # Test synthetic
    print("Creating synthetic dataset...")
    data = create_dataset('synthetic', {
        'n_samples': 100,
        'n_features': 10,
        'normalize': True
    })
    print(f"  Train: {data['X_train'].shape}")
    print(f"  Test: {data['X_test'].shape}")
