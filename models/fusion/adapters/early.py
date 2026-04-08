from typing import Any, Optional

import torch
from torch import nn


class EarlyFusionAdapter(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages if inject_stages is not None else []
        self.kwargs = kwargs

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        # rgb: [B, N, 3, H, W], spike: [B, N, T, H, W] or [B, N, C, H, W]
        if rgb.dim() != 5:
            raise ValueError("rgb must be 5D tensor [B, N, C, H, W]")
        if spike.dim() != 5:
            raise ValueError("spike must be 5D tensor [B, N, T, H, W] or [B, N, C, H, W]")
        bsz, steps, rgb_chans, height, width = rgb.shape
        time_dim = spike.size(2)
        rgb_rep = rgb.unsqueeze(2).expand(bsz, steps, time_dim, rgb_chans, height, width)
        rgb_rep = rgb_rep.reshape(bsz, steps * time_dim, rgb_chans, height, width)
        spk = spike.reshape(bsz, steps * time_dim, 1, height, width)
        if spk.size(2) != 1:
            spk = spk.mean(dim=2, keepdim=True)
        return self.operator(rgb_rep, spk)


__all__ = ["EarlyFusionAdapter"]
