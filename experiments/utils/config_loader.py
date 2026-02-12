"""
Configuration Loader
===================

Load and override YAML configurations with environment variable
substitution and nested parameter updates.
"""

import os
import yaml
import re
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file.
    
    Returns
    -------
    config : dict
        Parsed configuration.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve variable substitutions (${var} syntax)
    config = _resolve_variables(config)
    
    return config


def _resolve_variables(config: Dict, context: Dict = None) -> Dict:
    """
    Resolve variable substitutions in config.
    
    Supports:
    - ${var} - Reference to other config value
    - ${env:VAR} - Environment variable
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    context : dict, optional
        Context for variable resolution.
    
    Returns
    -------
    config : dict
        Config with variables resolved.
    """
    if context is None:
        context = config
    
    if isinstance(config, dict):
        return {k: _resolve_variables(v, context) for k, v in config.items()}
    
    elif isinstance(config, list):
        return [_resolve_variables(item, context) for item in config]
    
    elif isinstance(config, str):
        # Match ${...} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config)
        
        for match in matches:
            # Environment variable
            if match.startswith('env:'):
                var_name = match[4:]
                value = os.environ.get(var_name, '')
            # Config reference
            else:
                value = _get_nested_value(context, match)
            
            config = config.replace(f'${{{match}}}', str(value))
        
        return config
    
    else:
        return config


def _get_nested_value(d: Dict, path: str, default: Any = '') -> Any:
    """
    Get value from nested dict using dot notation.
    
    Parameters
    ----------
    d : dict
        Dictionary to search.
    path : str
        Dot-separated path (e.g., 'dataset.params.n_samples').
    default : any
        Default value if path not found.
    
    Returns
    -------
    value : any
        Value at path or default.
    """
    keys = path.split('.')
    value = d
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def override_config(config: Dict, overrides: Dict) -> Dict:
    """
    Override configuration values.
    
    Parameters
    ----------
    config : dict
        Base configuration.
    overrides : dict
        Overrides (supports dot notation for nested keys).
    
    Returns
    -------
    config : dict
        Updated configuration.
    
    Examples
    --------
    >>> config = {'dataset': {'params': {'n_samples': 1000}}}
    >>> overrides = {'dataset.params.n_samples': 500}
    >>> override_config(config, overrides)
    {'dataset': {'params': {'n_samples': 500}}}
    """
    import copy
    config = copy.deepcopy(config)
    
    for key, value in overrides.items():
        # Handle nested keys with dot notation
        if '.' in key:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        else:
            config[key] = value
    
    return config


if __name__ == '__main__':
    # Test config loader
    print("Testing config loader...")
    
    # Test variable resolution
    config = {
        'seed': 42,
        'dataset': {
            'params': {
                'n_samples': 1000,
                'seed': '${seed}'  # Reference to top-level seed
            }
        },
        'output': {
            'dir': '${env:HOME}/results'  # Environment variable
        }
    }
    
    resolved = _resolve_variables(config)
    print("\nResolved config:")
    print(yaml.dump(resolved, default_flow_style=False))
    
    # Test overrides
    overrides = {
        'dataset.params.n_samples': 500,
        'seed': 123
    }
    
    updated = override_config(resolved, overrides)
    print("\nWith overrides:")
    print(yaml.dump(updated, default_flow_style=False))
