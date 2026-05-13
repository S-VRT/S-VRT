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
        self._rgb_chans = getattr(operator, "rgb_chans", None)
        self._spike_chans = getattr(operator, "spike_chans", None)
        self._out_chans = getattr(operator, "out_chans", None)
        if self._rgb_chans is None or self._spike_chans is None or self._out_chans is None:
            raise ValueError("Middle fusion operator must expose rgb_chans, spike_chans, and out_chans.")

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
        if chans != self._rgb_chans:
            raise ValueError(
                f"Middle fusion expected x channels={self._rgb_chans}, got {chans}."
            )
        if spk_chans != self._spike_chans:
            raise ValueError(
                f"Middle fusion expected spike_ctx channels={self._spike_chans}, got {spk_chans}."
            )
        if self._out_chans != chans:
            raise ValueError(
                f"Middle fusion operator out_chans={self._out_chans} must match stage channels={chans}."
            )

        rgb_feat = x.permute(0, 2, 1, 3, 4)
        spk_feat = spike_ctx.permute(0, 2, 1, 3, 4)
        fused = self.operator(rgb_feat, spk_feat).permute(0, 2, 1, 3, 4)
        if self.mode == "replace":
            return fused
        return x + fused


__all__ = ["MiddleFusionAdapter"]
