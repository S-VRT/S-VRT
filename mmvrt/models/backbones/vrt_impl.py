"""Simplified, self-contained VRT implementation built from migrated layers.

This VRT is a migration-friendly implementation that uses components from
`mmvrt.models.layers.attention` and `mmvrt.models.motion.spynet`. It aims to be
feature-compatible for configuration-driven construction while remaining
readable and modular.
"""
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmvrt.models.layers.attention import Stage, RTMSA, TMSAG, Mlp_GEGLU
from mmvrt.models.motion.spynet import SpyNet, flow_warp


class VRT(nn.Module):
    """Migration-friendly VRT backbone (simplified).

    Notes:
    - This implementation intentionally focuses on modularity and using the
      migrated layer components rather than reproducing every legacy detail.
    - It preserves the standard constructor signature enough for configs to
      remain compatible.
    """

    def __init__(self,
                 upscale: int = 1,
                 in_chans: int = 3,
                 out_chans: int = 3,
                 img_size: List[int] = [6, 64, 64],
                 window_size: List[int] = [6, 8, 8],
                 depths: List[int] = [8, 8, 8],
                 embed_dims: List[int] = [120, 120, 120],
                 num_heads: List[int] = [6, 6, 6],
                 pa_frames: int = 2,
                 spynet_path: Optional[str] = None,
                 **kwargs):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames

        # Lightweight front conv to map input channels -> embed dim
        self.conv_first = nn.Conv3d(in_chans, embed_dims[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # A small stack of stages built from migrated Stage wrapper
        self.stages = nn.ModuleList()
        for i, d in enumerate(depths):
            dim = embed_dims[i] if i < len(embed_dims) else embed_dims[-1]
            heads = num_heads[i] if i < len(num_heads) else num_heads[-1]
            input_res = (img_size[0], img_size[1], img_size[2])
            self.stages.append(Stage(in_dim=dim, dim=dim, input_resolution=input_res,
                                     depth=d, num_heads=heads, window_size=window_size))

        # Norm and simple reconstruction head
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.conv_after_body = nn.Linear(embed_dims[-1], embed_dims[0])
        if self.upscale == 1:
            self.conv_last = nn.Conv3d(embed_dims[0], out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        else:
            self.conv_last = nn.Conv3d(embed_dims[0], out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # SpyNet for flow estimation (used by some stages)
        self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])

    def forward_features(self, x):
        # x: (N, embed, D, H, W)
        # apply each Stage sequentially
        out = x
        flows_backward = []
        flows_forward = []
        for st in self.stages:
            out = st(out, flows_backward, flows_forward)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward expects x shape (N, D, C, H, W)."""
        if x.dim() != 5:
            raise ValueError("Expected input tensor of shape (N, D, C, H, W)")
        x_in = x.clone()
        # conv_first expects (N, C, D, H, W)
        x = x.transpose(1, 2)  # (N, C, D, H, W) -> (N, C, D, H, W) (noop but keep semantics)
        x = self.conv_first(x)
        features = self.forward_features(x)
        x = features + self.conv_after_body(features.transpose(1, 4)).transpose(1, 4)
        x = self.conv_last(x).transpose(1, 2)  # (N, C_out, D, H, W) -> (N, D, C_out, H, W)
        # residual to RGB channels if same spatial dims
        x = x + x_in[:, :, :self.out_chans, :, :]
        return x


