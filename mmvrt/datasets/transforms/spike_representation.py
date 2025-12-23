"""Spike representation transforms (TFP, TDM, etc.)."""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch.utils.data import get_worker_info

from mmvrt.registry import TRANSFORMS
from mmvrt.utils.spike_loader import voxelize_spikes_tfp


@TRANSFORMS.register_module()
class SpikeToTFP:
    """Convert spike matrix to TFP (Temporal Focal Plane) representation.
    
    Args:
        num_channels: Number of output channels after voxelization
        half_win_length: Half window length for TFP algorithm
        device: Device for TFP computation ('cpu', 'cuda:0', 'auto', etc.)
        devices: List of devices for multi-GPU mode (takes precedence over device)
    """
    
    def __init__(
        self,
        num_channels: int = 4,
        half_win_length: int = 20,
        device: Union[str, torch.device] = 'cpu',
        devices: Optional[List[Union[str, torch.device]]] = None,
    ):
        self.num_channels = num_channels
        self.half_win_length = half_win_length
        self.device = self._normalize_single_device(device)
        self.devices = self._normalize_device_pool(devices) if devices else None
        
        # Use device pool if provided, otherwise use single device
        if self.devices:
            self.device = None
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert spike matrix to TFP voxels.
        
        Args:
            results: Dictionary containing 'spike_matrix' (T, H, W) or list of matrices
        
        Returns:
            Dictionary with 'spike_voxel' added (num_channels, H, W) or list of voxels
        """
        if 'spike_matrix' not in results:
            # Create zeros if no spike matrix
            results['spike_voxel'] = np.zeros(
                (self.num_channels, 250, 400),
                dtype=np.float32
            )
            return results
        
        spike_matrix = results['spike_matrix']
        device = self._select_device()
        
        if isinstance(spike_matrix, list):
            # Process multiple spike matrices
            spike_voxels = []
            for matrix in spike_matrix:
                voxel = self._process_spike_matrix(matrix, device)
                spike_voxels.append(voxel)
            results['spike_voxel'] = spike_voxels
        else:
            # Process single spike matrix
            results['spike_voxel'] = self._process_spike_matrix(spike_matrix, device)
        
        return results
    
    def _process_spike_matrix(
        self, 
        spike_matrix: np.ndarray, 
        device: Union[str, torch.device]
    ) -> np.ndarray:
        """Process a single spike matrix to TFP voxels.
        
        Args:
            spike_matrix: Spike matrix (T, H, W) uint8
            device: Device for TFP computation
        
        Returns:
            Voxelized spikes (num_channels, H, W) float32 [0, 1]
        """
        if spike_matrix.ndim != 3:
            raise ValueError(
                f"spike_matrix must be 3D (T, H, W), got shape {spike_matrix.shape}"
            )
        
        T, H, W = spike_matrix.shape
        if T <= 2 * self.half_win_length:
            # Not enough time steps, return zeros
            return np.zeros((self.num_channels, H, W), dtype=np.float32)
        
        voxel = voxelize_spikes_tfp(
            spike_matrix,
            num_channels=self.num_channels,
            device=device,
            half_win_length=self.half_win_length,
        )
        
        return voxel
    
    def _select_device(self) -> str:
        """Select device for current worker/process."""
        if self.devices:
            # Multi-device mode: select based on worker/rank
            worker_info = get_worker_info()
            if worker_info is not None:
                idx = worker_info.id % len(self.devices)
                return self.devices[idx]
            
            # Fall back to LOCAL_RANK or RANK
            local_rank = os.environ.get('LOCAL_RANK')
            if local_rank is not None:
                idx = int(local_rank) % len(self.devices)
                return self.devices[idx]
            
            rank = os.environ.get('RANK')
            if rank is not None:
                idx = int(rank) % len(self.devices)
                return self.devices[idx]
            
            return self.devices[0]
        else:
            return self.device or 'cpu'
    
    def _normalize_single_device(self, device_value: Union[str, torch.device]) -> str:
        """Normalize a single device specification."""
        if device_value is None:
            return 'cpu'
        device_str = str(device_value).strip()
        if not device_str:
            return 'cpu'
        return self._sanitize_device_string(device_str)
    
    def _normalize_device_pool(
        self, 
        pool_value: List[Union[str, torch.device]]
    ) -> List[str]:
        """Normalize a list of devices for multi-GPU mode."""
        if not pool_value:
            return []
        if isinstance(pool_value, str):
            pool_value = [pool_value]
        
        normalized = []
        for dev in pool_value:
            sanitized = self._sanitize_device_string(str(dev).strip())
            if sanitized:
                normalized.append(sanitized)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_devices = []
        for dev in normalized:
            if dev not in seen:
                unique_devices.append(dev)
                seen.add(dev)
        
        return unique_devices
    
    def _sanitize_device_string(self, device_str: str) -> str:
        """Validate device string and gracefully fall back to a safe option."""
        if device_str.lower() in {'auto', 'auto_cuda', 'cuda:auto'}:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
                visible = torch.cuda.device_count()
                return f'cuda:{local_rank % visible}'
            return 'cpu'
        
        if device_str.lower().startswith('cuda'):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                return 'cpu'
            parts = device_str.split(':')
            if len(parts) == 1 or parts[1] == '':
                return 'cuda:0'
            try:
                dev_idx = int(parts[1])
            except ValueError:
                dev_idx = 0
            visible = torch.cuda.device_count()
            if dev_idx >= visible:
                dev_idx = dev_idx % max(1, visible)
            return f'cuda:{dev_idx}'
        
        return 'cpu'


@TRANSFORMS.register_module()
class TemporalBinning:
    """Temporal binning for spike data (K-binning strategy).
    
    This transform divides the time sequence into K bins and aggregates spikes.
    
    Args:
        num_bins: Number of temporal bins (K)
        binning_strategy: Binning strategy ('uniform', 'center_aligned', 'sliding')
    """
    
    def __init__(
        self,
        num_bins: int = 4,
        binning_strategy: str = 'uniform',
    ):
        self.num_bins = num_bins
        self.binning_strategy = binning_strategy
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal binning to spike data.
        
        Args:
            results: Dictionary containing 'spike_matrix' or 'spike_voxel'
        
        Returns:
            Dictionary with binned spike data
        """
        # This is a placeholder - actual implementation depends on requirements
        # For now, we'll keep the spike_voxel as-is
        return results

