"""
Device Manager - GPU Memory and Multi-Device Coordination

Manages:
- Device selection and allocation
- Memory monitoring
- Device-to-device communication
- Resource cleanup
"""

import numpy as np
from typing import List, Dict, Optional
from .backend import get_backend


class DeviceManager:
    """
    Manages GPU devices and memory.
    
    Example:
        >>> dm = DeviceManager()
        >>> dm.print_device_info()
        >>> best_device = dm.select_device(min_memory_gb=4)
    """
    
    def __init__(self):
        self.backend = get_backend()
        self.devices = self._enumerate_devices()
    
    def _enumerate_devices(self) -> List[Dict]:
        """
        Enumerate all available devices.
        
        Returns:
        --------
        devices : list of dict
            Device information
        """
        devices = []
        
        if self.backend.name == 'jax':
            import jax
            for i, device in enumerate(jax.devices()):
                devices.append({
                    'id': i,
                    'name': str(device),
                    'type': device.device_kind,
                    'backend': 'jax'
                })
        
        elif self.backend.name == 'cupy':
            import cupy as cp
            n_devices = cp.cuda.runtime.getDeviceCount()
            
            for i in range(n_devices):
                props = cp.cuda.runtime.getDeviceProperties(i)
                devices.append({
                    'id': i,
                    'name': props['name'].decode(),
                    'type': 'GPU',
                    'memory_total_gb': props['totalGlobalMem'] / 1e9,
                    'backend': 'cupy'
                })
        
        else:  # numpy
            import multiprocessing
            devices.append({
                'id': 0,
                'name': 'CPU',
                'type': 'CPU',
                'n_cores': multiprocessing.cpu_count(),
                'backend': 'numpy'
            })
        
        return devices
    
    def print_device_info(self):
        """Print detailed device information."""
        print(f"\n{'='*60}")
        print(f"Device Information ({self.backend.name.upper()} backend)")
        print(f"{'='*60}")
        
        for device in self.devices:
            print(f"\nDevice {device['id']}: {device['name']}")
            print(f"  Type: {device['type']}")
            
            if 'memory_total_gb' in device:
                print(f"  Memory: {device['memory_total_gb']:.2f} GB")
            
            if 'n_cores' in device:
                print(f"  Cores: {device['n_cores']}")
        
        print(f"\n{'='*60}\n")
    
    def get_memory_usage(self, device_id: int = 0) -> Dict:
        """
        Get memory usage for a device.
        
        Returns:
        --------
        memory_info : dict
            Contains 'used_gb', 'free_gb', 'total_gb'
        """
        if self.backend.name == 'cupy':
            import cupy as cp
            
            # Set device
            cp.cuda.Device(device_id).use()
            
            # Get memory pool
            mempool = cp.get_default_memory_pool()
            
            used = mempool.used_bytes() / 1e9
            total = mempool.total_bytes() / 1e9
            
            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            total_device = props['totalGlobalMem'] / 1e9
            
            return {
                'used_gb': used,
                'total_gb': total_device,
                'free_gb': total_device - used,
                'pool_total_gb': total
            }
        
        elif self.backend.name == 'jax':
            # JAX doesn't expose memory API directly
            return {
                'used_gb': None,
                'total_gb': None,
                'free_gb': None,
                'note': 'JAX memory tracking not available'
            }
        
        else:  # numpy
            import psutil
            mem = psutil.virtual_memory()
            
            return {
                'used_gb': (mem.total - mem.available) / 1e9,
                'total_gb': mem.total / 1e9,
                'free_gb': mem.available / 1e9
            }
    
    def select_device(self, min_memory_gb: float = 1.0) -> int:
        """
        Select best available device with sufficient memory.
        
        Parameters:
        -----------
        min_memory_gb : float
            Minimum free memory required (GB)
        
        Returns:
        --------
        device_id : int
            ID of selected device
        """
        if self.backend.name == 'numpy':
            return 0  # Only one CPU device
        
        # Check each device
        for device in self.devices:
            device_id = device['id']
            
            if self.backend.name == 'cupy':
                mem_info = self.get_memory_usage(device_id)
                if mem_info['free_gb'] >= min_memory_gb:
                    print(f"✓ Selected device {device_id}: {device['name']}")
                    print(f"  Free memory: {mem_info['free_gb']:.2f} GB")
                    return device_id
            else:
                # JAX: just return first device
                return device_id
        
        raise RuntimeError(
            f"No device with {min_memory_gb:.1f} GB free memory found"
        )
    
    def cleanup(self, device_id: int = None):
        """
        Free device memory.
        
        Parameters:
        -----------
        device_id : int, optional
            Device to clean (all if None)
        """
        if self.backend.name == 'cupy':
            import cupy as cp
            
            if device_id is not None:
                cp.cuda.Device(device_id).use()
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                print(f"✓ Cleaned device {device_id}")
            else:
                # Clean all devices
                for device in self.devices:
                    cp.cuda.Device(device['id']).use()
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                print(f"✓ Cleaned all {len(self.devices)} devices")
        
        else:
            print("Cleanup not needed for {self.backend.name} backend")


if __name__ == "__main__":
    # Test device manager
    print("Testing Device Manager\n")
    
    dm = DeviceManager()
    dm.print_device_info()
    
    # Check memory
    print("\nMemory Usage:")
    mem_info = dm.get_memory_usage(0)
    for key, value in mem_info.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Select device
    print("\nSelecting device...")
    device_id = dm.select_device(min_memory_gb=0.5)
    print(f"Selected device: {device_id}")