#!/usr/bin/env python3
"""
Reproducible Experiment Runner
================================

Run fully reproducible optimization experiments from YAML configs.

Usage:
------
# Run default experiment
python run_experiment.py

# Run specific config
python run_experiment.py --config configs/mnist.yaml

# Override parameters
python run_experiment.py --config configs/default.yaml --seed 123 --dataset.params.n_samples=500

# Run with Hydra
python run_experiment.py dataset.name=mnist optimizers.adam.learning_rate=0.001
"""

import os
import sys
import yaml
import json
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reproducibility utilities
from experiments.utils.reproducibility import (
    set_random_seeds,
    get_system_info,
    save_experiment_metadata
)
from experiments.utils.config_loader import load_config, override_config
from experiments.utils.dataset_factory import create_dataset
from experiments.utils.optimizer_factory import create_optimizer
from experiments.utils.results_saver import ResultsSaver

# Import profiling
from performance import OptimizerProfiler


class ExperimentRunner:
    """
    Run reproducible experiments from configuration.
    """
    
    def __init__(self, config_path: str, overrides: Optional[Dict] = None):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file.
        overrides : dict, optional
            Configuration overrides.
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Apply overrides
        if overrides:
            self.config = override_config(self.config, overrides)
        
        # Setup logging
        self._setup_logging()
        
        # Setup reproducibility
        self._setup_reproducibility()
        
        # Create output directories
        self._setup_directories()
        
        # Initialize results saver
        self.results_saver = ResultsSaver(self.config)
        
        # Store experiment metadata
        self.metadata = {
            'config': self.config,
            'system_info': get_system_info(),
            'start_time': datetime.now().isoformat(),
            'config_path': config_path,
            'overrides': overrides or {}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Create logger
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(level)
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_config.get('save_to_file', True):
            log_dir = Path(log_config.get('log_dir', './logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{self.config['name']}_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging to {log_file}")
    
    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        repro_config = self.config.get('reproducibility', {})
        seed = self.config.get('seed', 42)
        
        # Set random seeds
        set_random_seeds(
            seed=seed,
            numpy_seed=repro_config.get('numpy_seed', seed),
            python_hash_seed=repro_config.get('python_hash_seed', seed),
            deterministic=repro_config.get('deterministic', True)
        )
        
        self.logger.info(f"Random seed set to {seed}")
        
        if repro_config.get('warn_on_nondeterministic', True):
            self.logger.warning(
                "Some operations may still be non-deterministic "
                "(e.g., GPU reductions, floating-point associativity)"
            )
    
    def _setup_directories(self):
        """Create output directories."""
        output_config = self.config.get('output', {})
        
        for dir_key in ['results_dir', 'plots_dir']:
            if dir_key in output_config:
                dir_path = Path(output_config[dir_key])
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
    
    def load_dataset(self):
        """Load dataset from configuration."""
        self.logger.info("Loading dataset...")
        
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        dataset_params = dataset_config.get('params', {})
        
        # Create dataset
        data = create_dataset(dataset_name, dataset_params)
        
        self.logger.info(
            f"Dataset loaded: {dataset_name} "
            f"(train: {data['X_train'].shape}, test: {data['X_test'].shape})"
        )
        
        # Store dataset info in metadata
        self.metadata['dataset'] = {
            'name': dataset_name,
            'train_shape': data['X_train'].shape,
            'test_shape': data['X_test'].shape,
            'metadata': data.get('metadata', {})
        }
        
        return data
    
    def run_optimizer(self, optimizer_name: str, optimizer_config: Dict, 
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Run single optimizer."""
        self.logger.info(f"Running optimizer: {optimizer_name}")
        
        # Create optimizer
        optimizer = create_optimizer(optimizer_name, optimizer_config)
        
        # Profile if enabled
        profiling_config = self.config.get('profiling', {})
        
        if profiling_config.get('enabled', False):
            profiler = OptimizerProfiler(
                enable_gpu=profiling_config.get('track_gpu', False)
            )
            profiler.start()
        
        # Train
        start_time = time.perf_counter()
        
        try:
            optimizer.fit(X_train, y_train)
            success = True
            error_msg = None
        except Exception as e:
            self.logger.error(f"{optimizer_name} failed: {e}")
            success = False
            error_msg = str(e)
        
        end_time = time.perf_counter()
        training_time = end_time - start_time
        
        # Stop profiler
        if profiling_config.get('enabled', False):
            profiler.stop()
            profile_stats = profiler.get_stats()
        else:
            profile_stats = {}
        
        # Evaluate
        if success:
            y_pred_test = optimizer.predict(X_test)
            test_mse = np.mean((y_test - y_pred_test) ** 2)
            
            history = optimizer.get_history()
            final_loss = history['losses'][-1] if history.get('losses') else None
            num_epochs = len(history.get('losses', []))
        else:
            test_mse = None
            final_loss = None
            num_epochs = 0
        
        # Compile results
        results = {
            'optimizer': optimizer_name,
            'config': optimizer_config,
            'success': success,
            'error': error_msg,
            'training_time': training_time,
            'test_mse': float(test_mse) if test_mse is not None else None,
            'final_train_loss': float(final_loss) if final_loss is not None else None,
            'num_epochs': num_epochs,
            'converged': final_loss < optimizer_config.get('tolerance', 1e-6) if final_loss else False,
            'profile': profile_stats
        }
        
        self.logger.info(
            f"{optimizer_name}: "
            f"Time={training_time:.2f}s, "
            f"Loss={final_loss:.2e if final_loss else 'N/A'}, "
            f"MSE={test_mse:.2e if test_mse else 'N/A'}"
        )
        
        return results
    
    def run(self) -> Dict:
        """Run complete experiment."""
        self.logger.info("="*60)
        self.logger.info(f"Starting experiment: {self.config['name']}")
        self.logger.info(f"Description: {self.config.get('description', 'N/A')}")
        self.logger.info("="*60)
        
        # Load dataset
        data = self.load_dataset()
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Run all enabled optimizers
        optimizer_configs = self.config.get('optimizers', {})
        results = {}
        
        for opt_name, opt_config in optimizer_configs.items():
            if not opt_config.get('enabled', True):
                self.logger.info(f"Skipping disabled optimizer: {opt_name}")
                continue
            
            result = self.run_optimizer(
                opt_name, opt_config,
                X_train, y_train,
                X_test, y_test
            )
            results[opt_name] = result
        
        # Add metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['results'] = results
        
        # Save results
        if self.config.get('output', {}).get('save_results', True):
            self.results_saver.save(self.metadata)
            self.logger.info("Results saved")
        
        self.logger.info("="*60)
        self.logger.info("Experiment complete!")
        self.logger.info("="*60)
        
        return self.metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run reproducible optimization experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    args, unknown = parser.parse_known_args()
    
    # Build overrides from command line
    overrides = {}
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.output_dir is not None:
        overrides['output'] = {'results_dir': args.output_dir}
    
    # Parse additional overrides (Hydra style)
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to parse as number, bool, or keep as string
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
            overrides[key] = value
    
    # Run experiment
    runner = ExperimentRunner(args.config, overrides)
    results = runner.run()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for opt_name, opt_result in results['results'].items():
        status = "✓" if opt_result['success'] else "✗"
        print(f"{status} {opt_name:12s}: "
              f"Time={opt_result['training_time']:6.2f}s, "
              f"MSE={opt_result['test_mse']:.2e if opt_result['test_mse'] else 'N/A'}")
    print("="*60)


if __name__ == '__main__':
    main()
