"""Formatting transforms for packing data into RestorationDataSample."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from mmvrt.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackRestorationInputs:
    """Pack pipeline output into RestorationDataSample.
    
    This transform:
    1. Concatenates RGB and spike voxels into input tensor
    2. Converts numpy arrays to tensors
    3. Stacks temporal frames
    4. Creates and returns a ``RestorationDataSample`` instance with:
       - ``inputs``: model input tensor or dict (for late-fusion)
       - ``gt``: ground truth tensor (optional)
       - ``pred``: None (filled later by model)
       - ``metainfo``: dictionary (clip_name, frame_idx, etc.)

    Note:
    - The pipeline's final transform should return a ``RestorationDataSample`` so
      downstream components (models, metrics, visualization hooks) operate using
      the unified DataSample contract. If you prefer a pipeline that returns plain
      dicts, the `RGBSpikeDataPreprocessor` also accepts dict inputs and converts
      them to a standardized dict for model consumption.
    
    Args:
        keys: Keys to pack from results dict
        meta_keys: Keys to include in metainfo
    """
    
    def __init__(
        self,
        keys: Optional[List[str]] = None,
        meta_keys: Optional[List[str]] = None,
    ):
        self.keys = keys or ['lq', 'gt']
        self.meta_keys = meta_keys or [
            'clip_name', 'frame_idx', 'total_num_frames', 
            'start_frame', 'key', 'gt_path'
        ]
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Pack results into a standardized dict consumed by the data_preprocessor.
        
        Args:
            results: Dictionary from pipeline transforms
        
        Returns:
            Dict with keys:
                - 'inputs': Tensor (T, C, H, W) or (B, T, C, H, W)
                - 'gt': Tensor or None
                - 'metainfo': dict
        """
        # Extract inputs (LQ + Spike concatenated)
        inputs = self._pack_inputs(results)

        # Extract GT
        gt = self._pack_gt(results)

        # Extract metainfo
        metainfo = self._pack_metainfo(results)

        return {'inputs': inputs, 'gt': gt, 'metainfo': metainfo}
    
    def _pack_inputs(self, results: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Pack LQ and spike voxels into a standardized structure.

        Returns:
            - If spike present: dict {'rgb': Tensor(T,3,H,W), 'spike': Tensor(T,S,H_s,W_s)}
            - Else: Tensor(T,3,H,W)
        """
        lq_frames = results.get('lq', None)
        spike_voxels = results.get('spike_voxel', None)
        
        if lq_frames is None:
            return None
        
        # Convert to list if single frame
        if not isinstance(lq_frames, list):
            lq_frames = [lq_frames]
        
        # Convert numpy to tensor if needed
        lq_tensors = []
        for frame in lq_frames:
            if isinstance(frame, np.ndarray):
                # Convert (H, W, C) to (C, H, W)
                if frame.ndim == 3 and frame.shape[2] in [1, 3, 4]:
                    frame = frame.transpose(2, 0, 1)
                frame = torch.from_numpy(frame).float()
            lq_tensors.append(frame)
        
        # Stack temporal frames: (T, C, H, W)
        lq_stacked = torch.stack(lq_tensors, dim=0)
        
        # If spike voxels are present, convert to tensor(s) and return separate
        # modalities. Spatial alignment and normalization are handled by the
        # `RGBSpikeDataPreprocessor`.
        if spike_voxels is not None:
            if not isinstance(spike_voxels, list):
                spike_voxels = [spike_voxels]

            spike_tensors = []
            for voxel in spike_voxels:
                if isinstance(voxel, np.ndarray):
                    # Voxel can be either (C, H, W) or (H, W, C)
                    if voxel.ndim == 3 and voxel.shape[0] <= voxel.shape[-1]:
                        spike_tensors.append(torch.from_numpy(voxel).float())
                    else:
                        spike_tensors.append(torch.from_numpy(voxel.transpose(2, 0, 1)).float())
                else:
                    spike_tensors.append(voxel)

            spike_stacked = torch.stack(spike_tensors, dim=0)
            return {'rgb': lq_stacked, 'spike': spike_stacked}

        return lq_stacked
    
    def _pack_gt(self, results: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Pack GT frames into tensor.
        
        Returns:
            Tensor of shape (T, 3, H, W)
        """
        gt_frames = results.get('gt', None)
        if gt_frames is None:
            return None
        
        # Convert to list if single frame
        if not isinstance(gt_frames, list):
            gt_frames = [gt_frames]
        
        # Convert numpy to tensor if needed
        gt_tensors = []
        for frame in gt_frames:
            if isinstance(frame, np.ndarray):
                # Convert (H, W, C) to (C, H, W)
                if frame.ndim == 3 and frame.shape[2] in [1, 3, 4]:
                    frame = frame.transpose(2, 0, 1)
                frame = torch.from_numpy(frame).float()
            gt_tensors.append(frame)
        
        # Stack temporal frames: (T, C, H, W)
        gt_stacked = torch.stack(gt_tensors, dim=0)
        
        return gt_stacked
    
    def _pack_metainfo(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Pack metainfo from results.
        
        Returns:
            Dictionary with metainfo
        """
        metainfo = {}
        for key in self.meta_keys:
            if key in results:
                metainfo[key] = results[key]
        return metainfo


@TRANSFORMS.register_module()
class TemporalSampling:
    """Temporal sampling for video frames.
    
    Samples a temporal window of frames around the current frame.
    This transform should be used before LoadRGBFrames/LoadSpikeRaw to
    generate neighbor frame paths.
    
    Args:
        num_frame: Number of frames in temporal window
        interval_list: List of temporal intervals to sample from
        random_reverse: Whether to randomly reverse temporal order
        filename_tmpl: Filename template (default: '08d')
        filename_ext: Filename extension (default: 'png')
    """
    
    def __init__(
        self,
        num_frame: int = 5,
        interval_list: Optional[List[int]] = None,
        random_reverse: bool = False,
        filename_tmpl: str = '08d',
        filename_ext: str = 'png',
    ):
        self.num_frame = num_frame
        self.interval_list = interval_list or [1]
        self.random_reverse = random_reverse
        self.filename_tmpl = filename_tmpl
        self.filename_ext = filename_ext
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sample temporal window.
        
        Args:
            results: Dictionary containing frame data and metadata
        
        Returns:
            Dictionary with neighbor frame paths added
        """
        import random
        from pathlib import Path
        
        # Get current frame index and clip metadata
        frame_idx = results.get('frame_idx', 0)
        total_num_frames = results.get('total_num_frames', 100)
        start_frame = results.get('start_frame', 0)
        clip_name = results.get('clip_name', '')
        
        # Get paths from results
        lq_path = results.get('lq_path', '')
        gt_path = results.get('gt_path', '')
        spike_path = results.get('spike_path', '')
        io_backend = results.get('io_backend', {'type': 'disk'})
        
        # Extract root paths and filename template
        filename_tmpl = results.get('filename_tmpl', self.filename_tmpl)
        filename_ext = results.get('filename_ext', self.filename_ext)
        
        # Select interval
        interval = random.choice(self.interval_list)
        
        # Calculate frame range
        endmost_start = start_frame + total_num_frames - self.num_frame * interval
        if frame_idx > endmost_start:
            frame_idx = random.randint(start_frame, endmost_start)
        
        # Generate neighbor list (absolute frame indices)
        neighbor_indices = list(range(frame_idx, frame_idx + self.num_frame * interval, interval))
        
        # Random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_indices.reverse()
        
        # Generate neighbor paths
        lq_paths = []
        gt_paths = []
        spike_paths = []
        
        # Extract root directories from original paths
        if io_backend.get('type') == 'lmdb':
            # LMDB: paths are just keys
            for neighbor_idx in neighbor_indices:
                neighbor_name = f'{neighbor_idx:{filename_tmpl}}'
                lq_paths.append(f'{clip_name}/{neighbor_name}')
                gt_paths.append(f'{clip_name}/{neighbor_name}')
                spike_paths.append(f'{clip_name}/spike/{neighbor_name}.dat')
        else:
            # Disk: extract root from original path
            if lq_path:
                lq_root = Path(lq_path).parent.parent
            else:
                lq_root = None
            
            if gt_path:
                gt_root = Path(gt_path).parent.parent
            else:
                gt_root = None
            
            if spike_path:
                spike_root = Path(spike_path).parent.parent.parent  # clip/spike/file.dat -> clip
            else:
                spike_root = None
            
            for neighbor_idx in neighbor_indices:
                neighbor_name = f'{neighbor_idx:{filename_tmpl}}'
                
                if lq_root:
                    lq_paths.append(str(lq_root / clip_name / f'{neighbor_name}.{filename_ext}'))
                else:
                    lq_paths.append('')
                
                if gt_root:
                    gt_paths.append(str(gt_root / clip_name / f'{neighbor_name}.{filename_ext}'))
                else:
                    gt_paths.append('')
                
                if spike_root:
                    spike_paths.append(str(spike_root / clip_name / 'spike' / f'{neighbor_name}.dat'))
                else:
                    spike_paths.append('')
        
        results['lq_path'] = lq_paths
        results['gt_path'] = gt_paths
        results['spike_path'] = spike_paths
        results['neighbor_list'] = neighbor_indices
        results['frame_idx'] = frame_idx
        
        return results

