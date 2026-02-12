"""
Optimizer Factory
=================

Create optimizers from configuration.
"""

import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimizers import (
    SGD,
    MomentumSGD,
    NesterovMomentum,
    AdaGrad,
    RMSProp,
    Adam,
    AdamW
)

OPTIMIZER_REGISTRY = {
    'sgd': SGD,
    'momentum': MomentumSGD,
    'nesterov': NesterovMomentum,
    'adagrad': AdaGrad,
    'rmsprop': RMSProp,
    'adam': Adam,
    'adamw': AdamW
}


def create_optimizer(optimizer_name: str, config: Dict[str, Any]):
    """
    Create optimizer from configuration.
    
    Parameters
    ----------
    optimizer_name : str
        Optimizer name (sgd, momentum, nesterov, adagrad, rmsprop, adam, adamw).
    config : dict
        Optimizer configuration.
    
    Returns
    -------
    optimizer : BaseOptimizer
        Initialized optimizer instance.
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
    
    # Filter out non-optimizer params
    exclude_keys = {'enabled', 'monitor_convergence', 'verbose'}
    optimizer_params = {
        k: v for k, v in config.items()
        if k not in exclude_keys
    }
    
    # Create optimizer
    return optimizer_cls(**optimizer_params)


if __name__ == '__main__':
    # Test optimizer factory
    print("Testing optimizer factory...\n")
    
    # Test Adam
    print("Creating Adam optimizer...")
    config = {
        'learning_rate': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'epochs': 100,
        'enabled': True
    }
    optimizer = create_optimizer('adam', config)
    print(f"  Created: {optimizer}")
    print(f"  LR: {optimizer.learning_rate}")
