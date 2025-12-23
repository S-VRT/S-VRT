"""Base video dataset for restoration tasks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch.utils.data as data

try:
    from mmengine.dataset import Compose
except Exception:  # pragma: no cover
    # Lightweight fallback so the dataset still works without mmengine
    class Compose:  # type: ignore
        def __init__(self, transforms):
            self.transforms = transforms or []

        def __call__(self, results):
            for t in self.transforms:
                results = t(results)
            return results


class BaseVideoDataset(data.Dataset, ABC):
    """Base class for video restoration datasets.
    
    This class defines the interface for video datasets that manage:
    1. Indexing (video clips, frame numbers, corresponding spike files, GT)
    2. Reading necessary raw data (or delegating to Load transforms)
    3. Outputting unified field sample dicts (no augmentation, no TFP/TDM)
    
    Subclasses should implement:
    - _load_sample(): Load raw data for a given key/index
    - _build_keys(): Build the list of keys from metadata
    """
    
    def __init__(
        self,
        dataroot_gt: str,
        dataroot_lq: str,
        meta_info_file: str,
        num_frame: int = 5,
        test_mode: bool = False,
        pipeline: Optional[List[Any]] = None,
        **kwargs
    ):
        """Initialize base video dataset.
        
        Args:
            dataroot_gt: Root path for ground truth frames
            dataroot_lq: Root path for low-quality input frames
            meta_info_file: Path to meta information file
            num_frame: Number of frames in temporal window
            test_mode: Whether in test mode (affects train/val split)
            **kwargs: Additional dataset-specific arguments
        """
        super().__init__()
        self.gt_root = Path(dataroot_gt)
        self.lq_root = Path(dataroot_lq)
        self.meta_info_file = meta_info_file
        self.num_frame = num_frame
        self.test_mode = test_mode

        self.pipeline_cfg = pipeline or []
        self.pipeline = Compose(self.pipeline_cfg)

        # Build keys from meta info
        self.keys = self._build_keys()
    
    @abstractmethod
    def _build_keys(self) -> List[str]:
        """Build list of keys from meta_info_file.
        
        Returns:
            List of keys in format "clip_name/frame_number"
        """
        pass
    
    @abstractmethod
    def _load_sample(self, key: str) -> Dict[str, Any]:
        """Load raw sample data for a given key.
        
        Args:
            key: Sample key in format "clip_name/frame_number"
        
        Returns:
            Dictionary with raw data (paths or arrays), e.g.:
            {
                'lq_path': Path to LQ frame,
                'gt_path': Path to GT frame,
                'spike_path': Path to spike file (optional),
                'clip_name': str,
                'frame_idx': int,
                ...
            }
        """
        pass
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by index.
        
        Returns raw sample dict (no augmentation, no transforms).
        Transforms will be applied in pipeline.
        """
        key = self.keys[index]
        sample = self._load_sample(key)
        sample['key'] = key
        # apply pipeline (Compose is a no-op when empty)
        sample = self.pipeline(sample)
        return sample
    
    def __len__(self) -> int:
        return len(self.keys)

