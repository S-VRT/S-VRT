from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpikeUpsample(nn.Module):
    def __init__(self, spike_chans: int):
        super().__init__()
        self.spike_chans = spike_chans
        self.refine = nn.Sequential(
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spike_chans, spike_chans, kernel_size=3, padding=1),
        )

    def forward(self, spike: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if spike.dim() != 4:
            raise ValueError("spike must be 4D tensor [B_flat, S, H, W]")

        _, spike_chans, _, _ = spike.shape
        if spike_chans != self.spike_chans:
            raise ValueError(f"Expected spike channels={self.spike_chans}, got {spike_chans}")

        upsampled = F.interpolate(
            spike,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(upsampled)


class EarlyFusionAdapter(nn.Module):
    def __init__(
        self,
        operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        spike_chans: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages if inject_stages is not None else []
        self.spike_chans = spike_chans
        self.spike_upsample = SpikeUpsample(spike_chans) if spike_chans is not None else None
        self.expects_structured_early = bool(
            getattr(operator, "expects_structured_early", False)
        )
        self.kwargs = kwargs

    def forward(self, rgb: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        if rgb.dim() != 5:
            raise ValueError("rgb must be 5D tensor [B, N, C, H, W]")
        if spike.dim() != 5:
            raise ValueError("spike must be 5D tensor [B, N, S, H, W]")

        bsz, steps, rgb_chans, height, width = rgb.shape
        spike_bsz, spike_steps, spike_steps_per_frame, spike_height, spike_width = spike.shape

        if (bsz, steps) != (spike_bsz, spike_steps):
            raise ValueError("rgb and spike must share batch size and steps")

        if (spike_height, spike_width) != (height, width):
            if self.spike_upsample is None:
                raise ValueError(
                    "Cannot upsample spike features to match rgb spatial dimensions without spike_chans."
                )
            spike_flat = spike.reshape(bsz * steps, spike_steps_per_frame, spike_height, spike_width)
            spike_flat = self.spike_upsample(spike_flat, target_h=height, target_w=width)
            spike = spike_flat.reshape(bsz, steps, spike_steps_per_frame, height, width)

        if self.expects_structured_early:
            return self.operator(rgb, spike)

        rgb_rep = rgb.unsqueeze(2).expand(
            bsz, steps, spike_steps_per_frame, rgb_chans, height, width
        )
        rgb_rep = rgb_rep.reshape(bsz, steps * spike_steps_per_frame, rgb_chans, height, width)
        spk = spike.reshape(bsz, steps * spike_steps_per_frame, 1, height, width)
        return self.operator(rgb_rep, spk)


__all__ = ["SpikeUpsample", "EarlyFusionAdapter"]
