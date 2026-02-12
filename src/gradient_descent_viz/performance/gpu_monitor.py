"""
GPU Monitor
===========

Real-time GPU utilization monitoring during training.

Metrics:
- GPU memory usage
- GPU utilization %
- Memory bandwidth
- Temperature
- Power consumption
"""

import time
from typing import Dict, List, Optional
import numpy as np


class GPUMonitor:
    """
    Monitor GPU usage during training.
    
    Requires: nvidia-ml-py3 (pip install nvidia-ml-py3)
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.has_gpu = False
        self.measurements: List[Dict] = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.has_gpu = True
            
            # Get GPU name
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode('utf-8')
                
        except Exception as e:
            print(f"Warning: Could not initialize GPU monitoring: {e}")
            print("GPU metrics will not be available.")
    
    def record(self) -> Optional[Dict]:
        """Record current GPU metrics."""
        if not self.has_gpu:
            return None
        
        try:
            # Memory
            mem_info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used_mb = mem_info.used / 1024 / 1024
            memory_total_mb = mem_info.total / 1024 / 1024
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Utilization
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util_percent = util.gpu
            memory_util_percent = util.memory
            
            # Temperature
            try:
                temperature = self.nvml.nvmlDeviceGetTemperature(
                    self.handle, 
                    self.nvml.NVML_TEMPERATURE_GPU
                )
            except:
                temperature = None
            
            # Power
            try:
                power_mw = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
                power_w = power_mw / 1000.0
            except:
                power_w = None
            
            metrics = {
                'timestamp': time.time(),
                'memory_used_mb': memory_used_mb,
                'memory_total_mb': memory_total_mb,
                'memory_percent': memory_percent,
                'gpu_utilization': gpu_util_percent,
                'memory_utilization': memory_util_percent,
                'temperature_c': temperature,
                'power_w': power_w
            }
            
            self.measurements.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not record GPU metrics: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get summary statistics of GPU usage."""
        if not self.measurements:
            return {'error': 'No measurements recorded'}
        
        # Extract time series
        memory_used = [m['memory_used_mb'] for m in self.measurements]
        gpu_util = [m['gpu_utilization'] for m in self.measurements]
        mem_util = [m['memory_utilization'] for m in self.measurements]
        
        temps = [m['temperature_c'] for m in self.measurements if m['temperature_c'] is not None]
        powers = [m['power_w'] for m in self.measurements if m['power_w'] is not None]
        
        stats = {
            'gpu_name': self.gpu_name if self.has_gpu else 'N/A',
            'num_measurements': len(self.measurements),
            'memory': {
                'peak_mb': max(memory_used),
                'avg_mb': np.mean(memory_used),
                'min_mb': min(memory_used)
            },
            'utilization': {
                'gpu_avg_percent': np.mean(gpu_util),
                'gpu_max_percent': max(gpu_util),
                'memory_avg_percent': np.mean(mem_util),
                'memory_max_percent': max(mem_util)
            }
        }
        
        if temps:
            stats['temperature'] = {
                'avg_c': np.mean(temps),
                'max_c': max(temps),
                'min_c': min(temps)
            }
        
        if powers:
            stats['power'] = {
                'avg_w': np.mean(powers),
                'max_w': max(powers),
                'total_kwh': sum(powers) * len(powers) / 3600 / 1000  # Rough estimate
            }
        
        return stats
    
    def reset(self):
        """Reset measurements."""
        self.measurements = []


if __name__ == '__main__':
    # Test GPU monitoring
    monitor = GPUMonitor()
    
    if monitor.has_gpu:
        print(f"Monitoring GPU: {monitor.gpu_name}")
        print("Recording for 5 seconds...\n")
        
        for i in range(50):
            metrics = monitor.record()
            if metrics and i % 10 == 0:
                print(f"Memory: {metrics['memory_used_mb']:.1f} MB, "
                      f"GPU Util: {metrics['gpu_utilization']:.1f}%")
            time.sleep(0.1)
        
        print("\nGPU Usage Summary:")
        stats = monitor.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("No GPU available for monitoring.")
