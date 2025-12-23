"""Data preprocessor for RGB+Spike restoration tasks.

This handles normalization, modal concatenation, padding, and batch organization.
Following MMagic migration principles, normalization should be done here
rather than in transforms.
"""

from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

try:
    from mmengine.model import BaseDataPreprocessor
    _USE_MMENGINE = True
except ImportError:
    _USE_MMENGINE = False
    BaseDataPreprocessor = nn.Module

from mmvrt.registry import MODELS
from mmvrt.structures.data_sample import RestorationDataSample


@MODELS.register_module()
class RGBSpikeDataPreprocessor(BaseDataPreprocessor if _USE_MMENGINE else nn.Module):
    """Data preprocessor for RGB+Spike restoration.
    
    This handles:
    - RGB normalization (ImageNet-style or custom)
    - Spike normalization (optional)
    - Early-fusion: concatenate RGB and Spike along channel dimension
    - Late-fusion: keep RGB and Spike separate (for dual-stream backbones)
    - Batch padding and stacking
    
    Args:
        rgb_mean (list[float]): RGB mean for normalization. Default: ImageNet [0.485, 0.456, 0.406].
        rgb_std (list[float]): RGB std for normalization. Default: ImageNet [0.229, 0.224, 0.225].
        spike_mean (list[float], optional): Spike mean for normalization. If None, no normalization.
        spike_std (list[float], optional): Spike std for normalization. If None, no normalization.
        fusion_mode (str): Fusion mode. 'early' (concat) or 'late' (separate). Default: 'early'.
        bgr_to_rgb (bool): Whether to convert BGR to RGB. Default: False (assumes RGB input).
        stack_mode (str): How to stack temporal frames. 'stack' or 'list'. Default: 'stack'.
    """
    
    def __init__(
        self,
        rgb_mean: list = [0.485, 0.456, 0.406],
        rgb_std: list = [0.229, 0.224, 0.225],
        spike_mean: Optional[list] = None,
        spike_std: Optional[list] = None,
        fusion_mode: str = 'early',
        bgr_to_rgb: bool = False,
        stack_mode: str = 'stack',
        **kwargs
    ):
        if _USE_MMENGINE:
            super().__init__(**kwargs)
        else:
            super().__init__()
        
        # Register normalization parameters
        self.register_buffer('rgb_mean', torch.tensor(rgb_mean).view(1, 1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor(rgb_std).view(1, 1, 3, 1, 1))
        
        if spike_mean is not None and spike_std is not None:
            self.register_buffer('spike_mean', torch.tensor(spike_mean).view(1, 1, -1, 1, 1))
            self.register_buffer('spike_std', torch.tensor(spike_std).view(1, 1, -1, 1, 1))
            self.use_spike_norm = True
        else:
            self.use_spike_norm = False
        
        self.fusion_mode = fusion_mode
        self.bgr_to_rgb = bgr_to_rgb
        self.stack_mode = stack_mode
    
    def forward(
        self,
        data: Union[Dict[str, Tensor], Tensor, list],
        training: bool = False
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Forward preprocessing.
        
        Args:
            data: Input data. Can be:
                - Dict with 'inputs'/'lq'/'L' (input frames) and optionally 'gt' (ground truth)
                - Tensor: (B, T, C, H, W) or (T, C, H, W)
                - List of tensors
            training: Whether in training mode.
        
        Returns:
            Processed data. If fusion_mode='early', returns concatenated tensor.
            If fusion_mode='late', returns dict with 'rgb' and 'spike' keys.
        """
        # Accept RestorationDataSample, dicts produced by Pack transforms, or raw tensors.
        if isinstance(data, RestorationDataSample):
            inputs = data.inputs
            gt = data.gt
            metainfo = getattr(data, 'metainfo', None)
        elif isinstance(data, dict):
            # Pipeline may return {'inputs': tensor_or_dict, 'gt': ..., 'metainfo': ...}
            potential_inputs = data.get('inputs')
            if isinstance(potential_inputs, RestorationDataSample):
                inputs = potential_inputs.inputs
                gt = potential_inputs.gt
                metainfo = getattr(potential_inputs, 'metainfo', None)
            else:
                inputs = potential_inputs or data.get('lq') or data.get('L')
                gt = data.get('gt') or data.get('H')
                metainfo = data.get('metainfo', None)
        elif isinstance(data, Tensor):
            inputs = data
            gt = None
            metainfo = None
        elif isinstance(data, list):
            # Assume list of (T, C, H, W) tensors
            inputs = torch.stack(data, dim=0)  # (T, C, H, W)
            gt = None
            metainfo = None
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if inputs is None:
            raise ValueError("No inputs found in data passed to RGBSpikeDataPreprocessor")

        # If inputs is a plain tensor -> convert to (B, T, C, H, W)
        if torch.is_tensor(inputs):
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(0)  # (1, T, C, H, W)
            if inputs.dim() == 5:
                pass
            else:
                raise AssertionError(f"Expected 4D (T,C,H,W) or 5D (B,T,C,H,W) tensor, got {inputs.dim()}D")

            B, T, C, H, W = inputs.shape

            # Input is single-modal concatenated (RGB [+ spike])
            # Split RGB and spike channels by convention if possible.
            if C >= 3:
                rgb = inputs[:, :, :3, :, :]
                spike = inputs[:, :, 3:, :, :] if C > 3 else None
            else:
                raise ValueError("Tensor inputs must have at least 3 channels for RGB")

        else:
            # Inputs may be a dict {'rgb': Tensor(T,3,H,W), 'spike': Tensor(T,S,Hs,Ws)}
            if isinstance(inputs, dict):
                rgb = inputs.get('rgb', None)
                spike = inputs.get('spike', None)
                # Ensure tensors and add batch dim if needed
                if isinstance(rgb, np.ndarray):
                    rgb = torch.from_numpy(rgb).float()
                if torch.is_tensor(rgb) and rgb.dim() == 4:
                    rgb = rgb.unsqueeze(0)  # (1, T, 3, H, W)
                if spike is not None:
                    if isinstance(spike, np.ndarray):
                        spike = torch.from_numpy(spike).float()
                    if torch.is_tensor(spike) and spike.dim() == 4:
                        spike = spike.unsqueeze(0)  # (1, T, S, Hs, Ws)
                B, T, _, H, W = rgb.shape
        
        # BGR to RGB conversion (if needed)
        if self.bgr_to_rgb and rgb is not None and rgb.size(2) >= 3:
            rgb = rgb[:, :, [2, 1, 0] + list(range(3, rgb.size(2))), :, :]

        # Normalize RGB
        rgb = (rgb - self.rgb_mean) / self.rgb_std

        # Normalize Spike (if configured)
        if spike is not None and self.use_spike_norm:
            spike = (spike - self.spike_mean) / self.spike_std

        # Spatial alignment: if spike spatial resolution differs, resize spike to rgb
        if spike is not None:
            # spike: (B, T, S, Hs, Ws), rgb: (B, T, 3, H, W)
            _, _, S, Hs, Ws = spike.shape
            if Hs != H or Ws != W:
                # reshape to (B*T, S, Hs, Ws) -> interpolate -> reshape back
                spike_bt = spike.view(B * T, S, Hs, Ws)
                spike_bt = F.interpolate(spike_bt, size=(H, W), mode='bilinear', align_corners=False)
                spike = spike_bt.view(B, T, S, H, W)

        # Fusion
        if self.fusion_mode == 'early':
            if spike is not None:
                # concatenate along channel dimension -> (B, T, C_total, H, W)
                inputs_processed = torch.cat([rgb, spike], dim=2)
            else:
                inputs_processed = rgb
        elif self.fusion_mode == 'late':
            inputs_processed = {'rgb': rgb, 'spike': spike}
        else:
            raise ValueError(f"Invalid fusion_mode: {self.fusion_mode}. Must be 'early' or 'late'")
        
        # Process GT if provided
        if gt is not None:
            if gt.dim() == 4:
                gt = gt.unsqueeze(0)
            if self.bgr_to_rgb and gt.size(2) >= 3:
                gt = gt[:, :, [2, 1, 0] + list(range(3, gt.size(2))), :, :]
            # GT typically doesn't need normalization (it's the target)
            # But we can apply same BGR->RGB conversion
        
        # Return standardized dict consumed by models/evaluator/hooks.
        result: Dict[str, Any] = {}
        result['inputs'] = inputs_processed
        result['gt'] = gt
        result['metainfo'] = metainfo or {}
        return result

