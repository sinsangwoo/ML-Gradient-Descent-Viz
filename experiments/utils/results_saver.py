"""
Results Saver
=============

Save experiment results in multiple formats.
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class ResultsSaver:
    """
    Save experimental results with metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize results saver.
        
        Parameters
        ----------
        config : dict
            Experiment configuration.
        """
        self.config = config
        self.output_config = config.get('output', {})
    
    def save(self, metadata: Dict[str, Any]) -> None:
        """
        Save results in configured format.
        
        Parameters
        ----------
        metadata : dict
            Complete experiment metadata.
        """
        output_format = self.output_config.get('format', 'json')
        results_dir = Path(self.output_config.get('results_dir', './results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.get('name', 'experiment')
        
        if output_format == 'json':
            self._save_json(metadata, results_dir, exp_name, timestamp)
        elif output_format == 'pickle':
            self._save_pickle(metadata, results_dir, exp_name, timestamp)
        elif output_format == 'csv':
            self._save_csv(metadata, results_dir, exp_name, timestamp)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _save_json(self, metadata: Dict, results_dir: Path, 
                   exp_name: str, timestamp: str) -> None:
        """Save as JSON."""
        filename = results_dir / f"{exp_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Results saved to: {filename}")
    
    def _save_pickle(self, metadata: Dict, results_dir: Path,
                     exp_name: str, timestamp: str) -> None:
        """Save as pickle."""
        filename = results_dir / f"{exp_name}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Results saved to: {filename}")
    
    def _save_csv(self, metadata: Dict, results_dir: Path,
                  exp_name: str, timestamp: str) -> None:
        """Save results as CSV (flattened)."""
        filename = results_dir / f"{exp_name}_{timestamp}.csv"
        
        # Extract results for CSV
        results = metadata.get('results', {})
        
        if not results:
            print("No results to save as CSV")
            return
        
        # Flatten results
        rows = []
        for opt_name, opt_result in results.items():
            row = {
                'optimizer': opt_name,
                'success': opt_result.get('success', False),
                'training_time': opt_result.get('training_time', None),
                'test_mse': opt_result.get('test_mse', None),
                'final_loss': opt_result.get('final_train_loss', None),
                'num_epochs': opt_result.get('num_epochs', 0),
                'converged': opt_result.get('converged', False)
            }
            rows.append(row)
        
        # Write CSV
        if rows:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Results saved to: {filename}")
