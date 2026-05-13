from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexRestorationReducer(nn.Module):
    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, ns, chans, height, width = x.shape
        if spike_bins <= 0 or ns % spike_bins != 0:
            raise ValueError(f"Invalid spike_bins={spike_bins} for x shape {tuple(x.shape)}")
        frames = ns // spike_bins
        groups = x.reshape(bsz, frames, spike_bins, chans, height, width)
        index = max(0, min(self.index, spike_bins - 1))
        return groups[:, :, index, :, :, :]


class SelectorRestorationReducer(nn.Module):
    def __init__(self, selector_hidden: int = 8):
        super().__init__()
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=1, out_channels=selector_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=selector_hidden, out_channels=1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, ns, chans, height, width = x.shape
        if spike_bins <= 0 or ns % spike_bins != 0:
            raise ValueError(f"Invalid spike_bins={spike_bins} for x shape {tuple(x.shape)}")
        frames = ns // spike_bins
        groups = x.reshape(bsz, frames, spike_bins, chans, height, width)
        score_in = groups.mean(dim=3, keepdim=True).reshape(bsz * frames, 1, spike_bins, height, width)
        logits = self.selector(score_in).reshape(bsz * frames, spike_bins)
        weights = F.softmax(logits, dim=1).view(bsz, frames, spike_bins, 1, 1, 1)
        return (groups * weights).sum(dim=2)


class ResidualSelectorRestorationReducer(SelectorRestorationReducer):
    def forward(self, x: torch.Tensor, spike_bins: int, base_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if base_rgb is None:
            raise ValueError("base_rgb is required for residual_selector reducer.")
        residual = super().forward(x=x, spike_bins=spike_bins, base_rgb=None)
        if residual.shape != base_rgb.shape:
            raise ValueError(
                f"Residual reducer expects base_rgb shape {tuple(residual.shape)}, got {tuple(base_rgb.shape)}"
            )
        return residual


def build_restoration_reducer(cfg: Optional[Dict]) -> nn.Module:
    cfg = cfg or {}
    reducer_type = str(cfg.get("type", "index")).lower()
    if reducer_type == "index":
        return IndexRestorationReducer(index=int(cfg.get("index", 0)))
    if reducer_type == "selector":
        return SelectorRestorationReducer(selector_hidden=int(cfg.get("selector_hidden", 8)))
    if reducer_type == "residual_selector":
        return ResidualSelectorRestorationReducer(selector_hidden=int(cfg.get("selector_hidden", 8)))
    raise ValueError(f"Unsupported restoration reducer type: {reducer_type}")


__all__ = [
    "IndexRestorationReducer",
    "SelectorRestorationReducer",
    "ResidualSelectorRestorationReducer",
    "build_restoration_reducer",
]
