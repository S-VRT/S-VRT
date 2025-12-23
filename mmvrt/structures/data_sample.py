"""Data structures for restoration tasks following MMDet-style DataSample pattern."""

from typing import Any, Dict, Optional
import torch
from torch import Tensor


class RestorationDataSample:
    """Data structure for restoration task samples.
    
    This follows the MMDet DataSample pattern to provide a unified interface
    for components (restorer/metric/visualization) to work with restoration data.
    
    Attributes:
        inputs: Model input (may be RGB+Spike concatenated, or dict for late fusion)
        gt: Ground truth frames (B, T, C, H, W) or None
        pred: Prediction output (B, T, C, H, W) or None (filled after inference)
        metainfo: Dictionary containing video name, frame numbers, timestamps,
                  crop positions, scales, data source, etc.
    """
    
    def __init__(
        self,
        inputs: Optional[Tensor] = None,
        gt: Optional[Tensor] = None,
        pred: Optional[Tensor] = None,
        metainfo: Optional[Dict[str, Any]] = None,
    ):
        self.inputs = inputs
        self.gt = gt
        self.pred = pred
        self.metainfo = metainfo or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backward compatibility)."""
        result = {}
        if self.inputs is not None:
            result['lq'] = self.inputs  # Legacy key
            result['inputs'] = self.inputs
        if self.gt is not None:
            result['gt'] = self.gt
        if self.pred is not None:
            result['pred'] = self.pred
        result['metainfo'] = self.metainfo
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RestorationDataSample':
        """Create from dictionary format."""
        inputs = data.get('inputs') or data.get('lq') or data.get('L')
        gt = data.get('gt')
        pred = data.get('pred')
        metainfo = data.get('metainfo', {})
        return cls(inputs=inputs, gt=gt, pred=pred, metainfo=metainfo)
    
    def __repr__(self) -> str:
        parts = []
        if self.inputs is not None:
            parts.append(f"inputs={tuple(self.inputs.shape)}")
        if self.gt is not None:
            parts.append(f"gt={tuple(self.gt.shape)}")
        if self.pred is not None:
            parts.append(f"pred={tuple(self.pred.shape)}")
        if self.metainfo:
            parts.append(f"metainfo={list(self.metainfo.keys())}")
        return f"RestorationDataSample({', '.join(parts)})"


def pack_restoration_inputs(data: Dict[str, Any]) -> RestorationDataSample:
    """Pack dictionary data into RestorationDataSample.
    
    This is the standard transform that converts pipeline output dict
    into RestorationDataSample for use by restorer/metric/visualization.
    
    Args:
        data: Dictionary from pipeline transforms containing:
            - 'lq' or 'L' or 'inputs': input frames
            - 'gt': ground truth frames (optional)
            - 'metainfo': metadata dictionary (optional)
    
    Returns:
        RestorationDataSample instance
    """
    return RestorationDataSample.from_dict(data)

