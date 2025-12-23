"""GoPro RGB+Spike dataset implementation following MMDet-style separation."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import torch.utils.data as data

from mmvrt.datasets.datasets.base_video_dataset import BaseVideoDataset
from mmvrt.registry import DATASETS


@DATASETS.register_module()
class GoProRGBSpikeDataset(BaseVideoDataset):
    """GoPro dataset for RGB+Spike video restoration.
    
    This dataset only handles:
    1. Indexing (video clips, frame numbers, corresponding spike files, GT)
    2. Reading necessary raw data paths (delegating actual loading to transforms)
    3. Outputting unified field sample dicts (no augmentation, no TFP/TDM)
    
    All data processing (TFP/TDM, cropping, flipping, normalization) is handled
    by transforms in the pipeline.
    
    Args:
        dataroot_gt: Root path for ground truth frames
        dataroot_lq: Root path for low-quality input frames
        dataroot_spike: Root path for spike data (.dat files)
        meta_info_file: Path to meta information file
        num_frame: Number of frames in temporal window
        test_mode: Whether in test mode (affects train/val split)
        val_partition: Validation partition type ('REDS4' or 'official')
        filename_tmpl: Filename template for frames (default: '08d')
        filename_ext: Filename extension (default: 'png')
        io_backend: IO backend configuration dict
        **kwargs: Additional arguments passed to BaseVideoDataset
    """
    
    def __init__(
        self,
        dataroot_gt: str,
        dataroot_lq: str,
        dataroot_spike: str,
        meta_info_file: str,
        num_frame: int = 5,
        test_mode: bool = False,
        val_partition: str = 'REDS4',
        filename_tmpl: str = '08d',
        filename_ext: str = 'png',
        io_backend: Optional[Dict[str, Any]] = None,
        pipeline: Optional[list] = None,
        **kwargs
    ):
        self.dataroot_spike = Path(dataroot_spike)
        self.filename_tmpl = filename_tmpl
        self.filename_ext = filename_ext
        self.val_partition = val_partition
        self.io_backend = io_backend or {'type': 'disk'}
        
        # Store additional metadata for each key
        self.total_num_frames = []  # Total frames per clip
        self.start_frames = []  # Start frame index per clip
        
        # Create key to index mapping for quick lookup
        self._key_to_index = {}
        
        super().__init__(
            dataroot_gt=dataroot_gt,
            dataroot_lq=dataroot_lq,
            meta_info_file=meta_info_file,
            num_frame=num_frame,
            test_mode=test_mode,
            pipeline=pipeline,
            **kwargs
        )
        
        # Build key to index mapping after keys are created
        self._key_to_index = {key: idx for idx, key in enumerate(self.keys)}
    
    def _build_keys(self) -> List[str]:
        """Build list of keys from meta_info_file.
        
        Meta info file format (each line):
        clip_name frame_num (h,w,c) start_frame
        
        Example:
        720p_240fps_1 100 (720,1280,3) 0
        720p_240fps_3 100 (720,1280,3) 0
        """
        keys = []
        total_num_frames = []
        start_frames = []
        
        with open(self.meta_info_file, 'r') as fin:
            for line in fin:
                parts = line.strip().split(' ')
                if len(parts) < 4:
                    continue
                folder = parts[0]
                frame_num = int(parts[1])
                start_frame = int(parts[3])
                
                # Generate keys for all frames in this clip
                for i in range(start_frame, start_frame + frame_num):
                    key = f'{folder}/{i:{self.filename_tmpl}}'
                    keys.append(key)
                    total_num_frames.append(frame_num)
                    start_frames.append(start_frame)
        
        # Filter by validation partition
        val_partition = self._get_val_partition()
        
        filtered_keys = []
        filtered_total_frames = []
        filtered_start_frames = []
        
        for i, key in enumerate(keys):
            clip_name = key.split('/')[0]
            if self.test_mode:
                # In test mode, only include validation clips
                if clip_name in val_partition:
                    filtered_keys.append(key)
                    filtered_total_frames.append(total_num_frames[i])
                    filtered_start_frames.append(start_frames[i])
            else:
                # In train mode, exclude validation clips
                if clip_name not in val_partition:
                    filtered_keys.append(key)
                    filtered_total_frames.append(total_num_frames[i])
                    filtered_start_frames.append(start_frames[i])
        
        self.total_num_frames = filtered_total_frames
        self.start_frames = filtered_start_frames
        
        return filtered_keys
    
    def _get_val_partition(self) -> List[str]:
        """Get validation partition clip names."""
        if self.val_partition == 'REDS4':
            return ['000', '011', '015', '020']
        elif self.val_partition == 'official':
            return [f'{v:03d}' for v in range(240, 270)]
        else:
            # For GoPro or other datasets, return empty list if not specified
            return []
    
    def _load_sample(self, key: str) -> Dict[str, Any]:
        """Load raw sample data for a given key.
        
        Returns dictionary with paths and metadata (no actual data loading).
        Actual loading is done by Load transforms in the pipeline.
        
        Args:
            key: Sample key in format "clip_name/frame_number"
        
        Returns:
            Dictionary with:
            - lq_path: Path to LQ frame (single frame)
            - gt_path: Path to GT frame (single frame)
            - spike_path: Path to spike file (single frame)
            - clip_name: Clip name
            - frame_idx: Frame index
            - total_num_frames: Total frames in clip
            - start_frame: Start frame index of clip
        """
        clip_name, frame_name = key.split('/')
        frame_idx = int(frame_name)
        
        # Build paths for single frame
        if self.io_backend.get('type') == 'lmdb':
            lq_path = f'{clip_name}/{frame_name}'
            gt_path = f'{clip_name}/{frame_name}'
        else:
            lq_path = self.lq_root / clip_name / f'{frame_name}.{self.filename_ext}'
            gt_path = self.gt_root / clip_name / f'{frame_name}.{self.filename_ext}'
        
        spike_path = self.dataroot_spike / clip_name / 'spike' / f'{frame_name}.dat'
        
        # Get metadata for this key
        key_idx = self._key_to_index[key]
        total_num_frames = self.total_num_frames[key_idx]
        start_frame = self.start_frames[key_idx]
        
        return {
            'lq_path': str(lq_path),
            'gt_path': str(gt_path),
            'spike_path': str(spike_path),
            'clip_name': clip_name,
            'frame_idx': frame_idx,
            'total_num_frames': total_num_frames,
            'start_frame': start_frame,
            'io_backend': self.io_backend,
        }
    
    def get_neighbor_keys(self, key: str, neighbor_list: List[int]) -> Dict[str, List[str]]:
        """Get neighbor frame keys for temporal sampling.
        
        Args:
            key: Current frame key
            neighbor_list: List of frame indices relative to current frame
        
        Returns:
            Dictionary with 'lq_path', 'gt_path', 'spike_path' as lists
        """
        clip_name, frame_name = key.split('/')
        current_frame_idx = int(frame_name)
        
        lq_paths = []
        gt_paths = []
        spike_paths = []
        
        for neighbor_offset in neighbor_list:
            neighbor_idx = current_frame_idx + neighbor_offset
            neighbor_name = f'{neighbor_idx:{self.filename_tmpl}}'
            
            if self.io_backend.get('type') == 'lmdb':
                lq_paths.append(f'{clip_name}/{neighbor_name}')
                gt_paths.append(f'{clip_name}/{neighbor_name}')
            else:
                lq_paths.append(str(self.lq_root / clip_name / f'{neighbor_name}.{self.filename_ext}'))
                gt_paths.append(str(self.gt_root / clip_name / f'{neighbor_name}.{self.filename_ext}'))
            
            spike_paths.append(str(self.dataroot_spike / clip_name / 'spike' / f'{neighbor_name}.dat'))
        
        return {
            'lq_path': lq_paths,
            'gt_path': gt_paths,
            'spike_path': spike_paths,
        }

