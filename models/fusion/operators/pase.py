from typing import Dict

import torch
from torch import nn

from models.spk_encoder import PixelAdaptiveSpikeEncoder


class PaseFusionOperator(nn.Module):
    frame_contract = "expanded"

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
        kernel_size = int(operator_params.get('kernel_size', 3))
        hidden_chans = int(operator_params.get('hidden_chans', 32))
        normalize_kernel = bool(operator_params.get('normalize_kernel', True))
        self.pase = PixelAdaptiveSpikeEncoder(
            in_chans=spike_chans,
            out_chans=out_chans,
            kernel_size=kernel_size,
            hidden_chans=hidden_chans,
            normalize_kernel=normalize_kernel,
        )
        self.rgb_proj = nn.Conv2d(rgb_chans, out_chans, kernel_size=1)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != spike_feat.dim():
            raise ValueError('rgb and spike must have the same number of dimensions')
        if rgb_feat.dim() == 5:
            bsz, steps, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_steps, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, time, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            rgb_flat = rgb_feat.reshape(bsz * steps, rgb_chans, height, width)
            spike_flat = spike_feat.reshape(bsz * steps, spike_chans, height, width)
            fused = self.rgb_proj(rgb_flat) + self.pase(spike_flat)
            return fused.reshape(bsz, steps, self.out_chans, height, width)
        if rgb_feat.dim() == 4:
            bsz, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, height, width) != (spike_bsz, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            return self.rgb_proj(rgb_feat) + self.pase(spike_feat)
        raise ValueError('Expected rgb and spike features with 4 or 5 dimensions')


__all__ = ['PaseFusionOperator']
