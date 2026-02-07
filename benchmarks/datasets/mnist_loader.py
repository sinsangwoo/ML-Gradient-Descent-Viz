"""
MNIST Dataset Loader
====================

Loads MNIST for linear regression benchmark.
Task: Predict digit class (0-9) from 784 pixel features.

Theoretical Properties:
- Input dimension: d = 784
- Training samples: n = 60,000
- Test samples: n = 10,000
- Condition number: κ ≈ 10^4 (ill-conditioned)
"""

import numpy as np
from typing import Tuple, Optional
import urllib.request
import gzip
import os


def download_mnist(data_dir: str = './data/mnist') -> None:
    """Download MNIST dataset from Yann LeCun's website."""
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    print("✓ MNIST dataset downloaded successfully")


def load_mnist_raw(data_dir: str = './data/mnist') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw MNIST data from binary files."""
    if not os.path.exists(data_dir):
        download_mnist(data_dir)
    
    def load_images(filename: str) -> np.ndarray:
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0
    
    def load_labels(filename: str) -> np.ndarray:
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)
    
    X_train = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    X_test = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    return X_train, y_train, X_test, y_test


def load_mnist(
    n_samples: Optional[int] = None,
    binary_classification: bool = False,
    normalize: bool = True,
    add_bias: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load MNIST for optimization benchmarking.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of training samples to use. If None, use all 60k.
    binary_classification : bool
        If True, convert to binary classification (0-4 vs 5-9).
    normalize : bool
        If True, standardize features to zero mean and unit variance.
    add_bias : bool
        If True, add bias column to features.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and test data.
    metadata : dict
        Dataset properties (condition number, dimensions, etc.).
    """
    np.random.seed(seed)
    
    # Load raw data
    X_train, y_train, X_test, y_test = load_mnist_raw()
    
    # Subsample if requested
    if n_samples is not None and n_samples < len(X_train):
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Binary classification
    if binary_classification:
        y_train = (y_train >= 5).astype(np.float32)
        y_test = (y_test >= 5).astype(np.float32)
    else:
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
    
    # Normalize features
    if normalize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    
    # Add bias term
    if add_bias:
        X_train = np.column_stack([np.ones(len(X_train)), X_train])
        X_test = np.column_stack([np.ones(len(X_test)), X_test])
    
    # Compute theoretical properties
    XtX = X_train.T @ X_train / len(X_train)
    eigenvalues = np.linalg.eigvalsh(XtX)
    condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)
    
    metadata = {
        'name': 'MNIST',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'task': 'binary_classification' if binary_classification else 'multiclass',
        'condition_number': float(condition_number),
        'max_eigenvalue': float(eigenvalues.max()),
        'min_eigenvalue': float(eigenvalues.min())
    }
    
    print(f"✓ MNIST loaded: {metadata}")
    
    return X_train, y_train, X_test, y_test, metadata


if __name__ == '__main__':
    # Test loading
    X_train, y_train, X_test, y_test, meta = load_mnist(
        n_samples=10000,
        binary_classification=True
    )
    
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Condition number: {meta['condition_number']:.2e}")
