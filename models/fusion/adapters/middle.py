from typing import Any, Optional

import torch
from torch import nn

from ..base import validate_mode


class MiddleFusionAdapter(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.operator = operator
        self.mode = validate_mode(mode)
        self.inject_stages = set(inject_stages or [])
        self.kwargs = kwargs

    def forward(self, stage_idx: int, x: torch.Tensor, spike_ctx: torch.Tensor) -> torch.Tensor:
        if stage_idx not in self.inject_stages:
            return x
        if x.dim() != 5:
            raise ValueError("x must be 5D tensor [B, C, D, H, W]")
        if spike_ctx.dim() != 5:
            raise ValueError("spike_ctx must be 5D tensor [B, C, D, H, W]")
        bsz, chans, steps, height, width = x.shape
        spk_bsz, spk_chans, spk_steps, spk_height, spk_width = spike_ctx.shape
        if (bsz, steps, height, width) != (spk_bsz, spk_steps, spk_height, spk_width):
            raise ValueError("x and spike_ctx must share batch, time, height, and width dimensions")

        rgb_feat = x.permute(0, 2, 1, 3, 4)
        spk_feat = spike_ctx.permute(0, 2, 1, 3, 4)
        fused = self.operator(rgb_feat, spk_feat).permute(0, 2, 1, 3, 4)
        if self.mode == "replace":
            return fused
        return x + fused


__all__ = ["MiddleFusionAdapter"]
