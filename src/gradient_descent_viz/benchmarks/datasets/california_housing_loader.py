"""
California Housing Dataset Loader
==================================

Loads California Housing dataset for regression benchmarks.

Theoretical Properties:
- Input dimension: d = 8
- Samples: n ≈ 20,640
- Target: Median house value ($100k)
- Condition number: κ ≈ 50-100 (moderately conditioned)

Features:
- MedInc: Median income
- HouseAge: Median house age
- AveRooms: Average rooms per household
- AveBedrms: Average bedrooms per household  
- Population: Block population
- AveOccup: Average household occupancy
- Latitude, Longitude: Location
"""

import numpy as np
from typing import Tuple, Optional
import urllib.request
import os
import tarfile


def download_california_housing(data_dir: str = './data/housing') -> str:
    """Download California Housing dataset."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
    filepath = os.path.join(data_dir, 'housing.csv')
    
    if not os.path.exists(filepath):
        print("Downloading California Housing dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("✓ Dataset downloaded successfully")
    
    return filepath


def load_california_housing(
    n_samples: Optional[int] = None,
    normalize: bool = True,
    add_bias: bool = True,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load California Housing dataset.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to use. If None, use all ~20k.
    normalize : bool
        If True, standardize features.
    add_bias : bool
        If True, add bias column.
    test_size : float
        Fraction of data for testing.
    seed : int
        Random seed.
    
    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    metadata : dict
    """
    np.random.seed(seed)
    
    # Load data
    filepath = download_california_housing()
    
    # Read CSV (skip header)
    data = []
    with open(filepath, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            values = line.strip().split(',')
            # longitude,latitude,housing_median_age,total_rooms,total_bedrooms,
            # population,households,median_income,median_house_value,ocean_proximity
            try:
                row = [float(v) for v in values[:9]]  # Skip ocean_proximity
                data.append(row)
            except ValueError:
                continue
    
    data = np.array(data)
    
    # Remove rows with NaN
    data = data[~np.isnan(data).any(axis=1)]
    
    # Features: [longitude, latitude, age, rooms, bedrooms, pop, households, income]
    # Target: median_house_value (col 8)
    X = data[:, :8]
    y = data[:, 8] / 100000.0  # Scale to $100k units
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Train-test split
    n_train = int(len(X) * (1 - test_size))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Normalize
    if normalize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    
    # Add bias
    if add_bias:
        X_train = np.column_stack([np.ones(len(X_train)), X_train])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
    
    # Compute properties
    XtX = X_train.T @ X_train / len(X_train)
    eigenvalues = np.linalg.eigvalsh(XtX)
    condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)
    
    metadata = {
        'name': 'California Housing',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'condition_number': float(condition_number),
        'max_eigenvalue': float(eigenvalues.max()),
        'min_eigenvalue': float(eigenvalues.min()),
        'feature_names': ['bias', 'longitude', 'latitude', 'age', 'rooms', 
                         'bedrooms', 'population', 'households', 'income'] if add_bias 
                        else ['longitude', 'latitude', 'age', 'rooms', 
                             'bedrooms', 'population', 'households', 'income']
    }
    
    print(f"✓ California Housing loaded: {metadata}")
    
    return X_train, y_train, X_test, y_test, metadata


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, meta = load_california_housing()
    print(f"Training shape: {X_train.shape}")
    print(f"Condition number: {meta['condition_number']:.2f}")
