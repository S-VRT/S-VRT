from typing import Dict

import torch
from torch import nn


class ConcatFusionOperator(nn.Module):
    def __init__(
        self,
        rgb_chans: int,
        spike_chans: int,
        out_chans: int,
        operator_params: Dict,
    ):
        super().__init__()
        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.operator_params = operator_params
        self.proj = nn.Conv2d(rgb_chans + spike_chans, out_chans, kernel_size=1)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() == 5:
            bsz, steps, _, height, width = rgb_feat.shape
            if spike_feat.shape[:2] != (bsz, steps):
                raise ValueError('rgb and spike must share batch and time dimensions')
            rgb_feat = rgb_feat.reshape(bsz * steps, self.rgb_chans, height, width)
            spike_feat = spike_feat.reshape(bsz * steps, self.spike_chans, height, width)
            fused = torch.cat([rgb_feat, spike_feat], dim=1)
            out = self.proj(fused)
            return out.reshape(bsz, steps, self.out_chans, height, width)
        if rgb_feat.dim() == 4:
            fused = torch.cat([rgb_feat, spike_feat], dim=1)
            return self.proj(fused)
        raise ValueError('Expected rgb and spike features with 4 or 5 dimensions')


__all__ = ['ConcatFusionOperator']
