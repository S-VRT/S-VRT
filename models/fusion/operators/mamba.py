from typing import Dict

import torch
from torch import nn


class MambaFusionOperator(nn.Module):
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
        self.rgb_proj = nn.Conv2d(rgb_chans, out_chans, kernel_size=1)
        self.spike_proj = nn.Conv2d(spike_chans, out_chans, kernel_size=1)
        try:
            from mamba_ssm import Mamba  # type: ignore
        except (ImportError, ModuleNotFoundError):
            self.mamba = None
            return
        d_state = int(operator_params.get('d_state', 16))
        d_conv = int(operator_params.get('d_conv', 4))
        expand = int(operator_params.get('expand', 2))
        self.mamba = Mamba(d_model=out_chans, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            raise RuntimeError("mamba_ssm is required for mamba fusion operator.")
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
            fused = self.rgb_proj(rgb_flat) + self.spike_proj(spike_flat)
            fused = fused.reshape(bsz, steps, self.out_chans, height, width)
            seq = fused.permute(0, 3, 4, 1, 2).reshape(bsz * height * width, steps, self.out_chans)
            seq = self.mamba(seq)
            out = seq.reshape(bsz, height, width, steps, self.out_chans).permute(0, 3, 4, 1, 2)
            return out
        if rgb_feat.dim() == 4:
            bsz, rgb_chans, height, width = rgb_feat.shape
            spike_bsz, spike_chans, spike_height, spike_width = spike_feat.shape
            if (bsz, height, width) != (spike_bsz, spike_height, spike_width):
                raise ValueError('rgb and spike must share batch, height, and width dimensions')
            if rgb_chans != self.rgb_chans:
                raise ValueError(f'Expected rgb channels={self.rgb_chans}, got {rgb_chans}')
            if spike_chans != self.spike_chans:
                raise ValueError(f'Expected spike channels={self.spike_chans}, got {spike_chans}')
            fused = self.rgb_proj(rgb_feat) + self.spike_proj(spike_feat)
            seq = fused.permute(0, 2, 3, 1).reshape(bsz * height * width, 1, self.out_chans)
            seq = self.mamba(seq)
            out = seq.reshape(bsz, height, width, self.out_chans).permute(0, 3, 1, 2)
            return out
        raise ValueError('Expected rgb and spike features with 4 or 5 dimensions')


__all__ = ['MambaFusionOperator']
