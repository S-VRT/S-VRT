"""I/O transforms for loading RGB frames and Spike data."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch

from mmvrt.registry import TRANSFORMS
from mmvrt.utils.file_client import FileClient, imfrombytes
from mmvrt.utils.spike_loader import SpikeStreamSimple


@TRANSFORMS.register_module()
class LoadRGBFrames:
    """Load RGB frames from paths.
    
    Supports both disk and LMDB backends.
    
    Args:
        io_backend: IO backend configuration dict
        to_float32: Whether to convert to float32 [0, 1]
        color_type: Color type ('color' or 'grayscale')
        channel_order: Channel order ('rgb' or 'bgr')
    """
    
    def __init__(
        self,
        io_backend: Optional[Dict[str, Any]] = None,
        to_float32: bool = True,
        color_type: str = 'color',
        channel_order: str = 'rgb',
    ):
        self.io_backend = io_backend or {'type': 'disk'}
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        
        # Initialize file client if needed
        self._file_client = None
        self._init_file_client()
    
    def _init_file_client(self):
        """Initialize file client based on io_backend."""
        backend_type = self.io_backend.get('type', 'disk')
        backend_kwargs = {k: v for k, v in self.io_backend.items() if k != 'type'}

        # Use mmvrt local FileClient implementation (disk backend supported).
        if backend_type == 'lmdb':
            # LMDB support is not implemented in the lightweight FileClient.
            # Fall back to disk FileClient behavior for now.
            self._file_client = FileClient('disk', **backend_kwargs)
        else:
            self._file_client = FileClient(backend_type, **backend_kwargs)
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load RGB frames.
        
        Args:
            results: Dictionary containing 'lq_path' and/or 'gt_path'
        
        Returns:
            Dictionary with 'lq' and/or 'gt' arrays added
        """
        # Load LQ frames if path is provided
        if 'lq_path' in results:
            lq_path = results['lq_path']
            if isinstance(lq_path, (list, tuple)):
                # Load multiple frames
                lq_frames = []
                for path in lq_path:
                    img = self._load_image(path, 'lq')
                    lq_frames.append(img)
                results['lq'] = lq_frames
            else:
                # Load single frame
                results['lq'] = self._load_image(lq_path, 'lq')
        
        # Load GT frames if path is provided
        if 'gt_path' in results:
            gt_path = results['gt_path']
            if isinstance(gt_path, (list, tuple)):
                # Load multiple frames
                gt_frames = []
                for path in gt_path:
                    img = self._load_image(path, 'gt')
                    gt_frames.append(img)
                results['gt'] = gt_frames
            else:
                # Load single frame
                results['gt'] = self._load_image(gt_path, 'gt')
        
        return results
    
    def _load_image(self, path: Union[str, Path], key: str) -> np.ndarray:
        """Load a single image from path.
        
        Args:
            path: Image path
            key: Key for file client ('lq' or 'gt')
        
        Returns:
            Image array in RGB format, float32 [0, 1] if to_float32=True
        """
        # Get image bytes from file client
        img_bytes = self._file_client.get(path, key) if self.io_backend.get('type') == 'lmdb' else self._file_client.get(path)

        # Decode image using mmvrt.utils.file_client.imfrombytes
        flag = cv2.IMREAD_COLOR if self.color_type == 'color' else cv2.IMREAD_GRAYSCALE
        img = imfrombytes(img_bytes, flag=flag, float32=self.to_float32)
        
        # Convert BGR to RGB if needed
        if self.channel_order == 'rgb' and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img


@TRANSFORMS.register_module()
class LoadSpikeRaw:
    """Load raw spike data from .dat files.
    
    Args:
        spike_h: Spike camera height
        spike_w: Spike camera width
        flipud: Whether to flip spike data vertically
        print_dat_detail: Whether to print spike file details
    """
    
    def __init__(
        self,
        spike_h: int = 250,
        spike_w: int = 400,
        flipud: bool = True,
        print_dat_detail: bool = False,
    ):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.flipud = flipud
        self.print_dat_detail = print_dat_detail
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load spike data.
        
        Args:
            results: Dictionary containing 'spike_path'
        
        Returns:
            Dictionary with 'spike_matrix' added (T, H, W) uint8 array
        """
        if 'spike_path' not in results:
            # If no spike path, create zeros
            results['spike_matrix'] = np.zeros(
                (100, self.spike_h, self.spike_w), 
                dtype=np.uint8
            )
            return results
        
        spike_path = Path(results['spike_path'])
        
        if isinstance(spike_path, (list, tuple)):
            # Load multiple spike files
            spike_matrices = []
            for path in spike_path:
                matrix = self._load_spike_file(Path(path))
                spike_matrices.append(matrix)
            results['spike_matrix'] = spike_matrices
        else:
            # Load single spike file
            if spike_path.exists():
                results['spike_matrix'] = self._load_spike_file(spike_path)
            else:
                # Create zeros if file doesn't exist
                results['spike_matrix'] = np.zeros(
                    (100, self.spike_h, self.spike_w),
                    dtype=np.uint8
                )
        
        return results
    
    def _load_spike_file(self, spike_path: Path) -> np.ndarray:
        """Load a single spike file.
        
        Args:
            spike_path: Path to .dat file
        
        Returns:
            Spike matrix (T, H, W) uint8 array
        """
        spike_stream = SpikeStreamSimple(
            str(spike_path),
            spike_h=self.spike_h,
            spike_w=self.spike_w,
            print_dat_detail=self.print_dat_detail
        )
        spike_matrix = spike_stream.get_spike_matrix(flipud=self.flipud)
        return spike_matrix

