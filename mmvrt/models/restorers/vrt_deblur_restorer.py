"""VRT deblur restorer following MMDet-style restorer pattern.

This is the glue layer that combines backbone, head, loss, and data_preprocessor.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch import Tensor

from mmvrt.models.restorers.base_restorer import BaseRestorer
from mmvrt.registry import MODELS


@MODELS.register_module()
class VRTDeblurRestorer(BaseRestorer):
    """VRT deblur restorer.
    
    This is the top-level model for video deblurring tasks using VRT architecture.
    It combines backbone, optional head, loss, and data_preprocessor.
    
    Args:
        backbone (dict): Backbone config (e.g., VRTBackbone).
        head (dict, optional): Head config for final reconstruction. If None,
            backbone output is used directly.
        data_preprocessor (dict, optional): Data preprocessor config.
        loss (dict, optional): Loss module config.
        init_cfg (dict, optional): Initialization config.
    """
    
    def __init__(
        self,
        backbone: Dict[str, Any],
        head: Optional[Dict[str, Any]] = None,
        data_preprocessor: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ):
        # Build components via registry (MMEngine-compatible)
        # NOTE: we clone dicts to avoid mutating caller config when Runner.from_cfg reuses them
        backbone_cfg = dict(backbone)
        head_cfg = dict(head) if head else None
        loss_cfg = dict(loss) if loss else None
        preprocessor_cfg = dict(data_preprocessor) if data_preprocessor else None

        backbone_module = MODELS.build(backbone_cfg)
        # If head_cfg is provided, inject sensible defaults from backbone (embed/pa/upscale)
        if head_cfg and isinstance(head_cfg, dict):
            # derive feature channels from backbone
            feat_ch = getattr(backbone_module, 'feat_channels', getattr(backbone_module, 'in_chans', None))
            # If backbone is non-pa, linear_fuse expects feat_ch * temporal_length
            if getattr(backbone_module, 'pa_frames', None) in (None, False):
                temporal_len = None
                img_size_attr = getattr(backbone_module, 'img_size', None)
                if isinstance(img_size_attr, (list, tuple)) and len(img_size_attr) > 0:
                    temporal_len = int(img_size_attr[0])
                if temporal_len:
                    head_cfg.setdefault('in_channels', feat_ch * temporal_len)
                else:
                    head_cfg.setdefault('in_channels', feat_ch)
            else:
                head_cfg.setdefault('in_channels', feat_ch)
            # propagate pa_frames and upscale when available
            if getattr(backbone_module, 'pa_frames', None) is not None:
                head_cfg.setdefault('pa_frames', bool(getattr(backbone_module, 'pa_frames')))
            if getattr(backbone_module, 'upscale', None) is not None:
                head_cfg.setdefault('upscale', getattr(backbone_module, 'upscale'))
        head_module = MODELS.build(head_cfg) if head_cfg else None
        loss_module = MODELS.build(loss_cfg) if loss_cfg else None
        preprocessor_module = MODELS.build(preprocessor_cfg) if preprocessor_cfg else None

        super().__init__(
            backbone=backbone_module,
            head=head_module,
            data_preprocessor=preprocessor_module,
            loss_module=loss_module,
            init_cfg=init_cfg,
        )

