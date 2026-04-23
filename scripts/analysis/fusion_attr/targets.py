from __future__ import annotations

import torch

from .io import AnalysisSample


def build_box_mask(sample: AnalysisSample, height: int, width: int, device: torch.device) -> torch.Tensor:
    if sample.mask_type != "box":
        raise ValueError(f"Unsupported mask type: {sample.mask_type}")
    x1, y1, x2, y2 = sample.xyxy
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Mask box is empty after clipping")
    mask = torch.zeros(1, 1, height, width, device=device)
    mask[:, :, y1:y2, x1:x2] = 1.0
    return mask


def masked_charbonnier_target(
    output: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    while mask.ndim < output.ndim:
        mask = mask.unsqueeze(1)
    diff = (output - gt) * mask
    active_elements = (mask > 0).float().expand_as(output).sum().clamp_min(1.0)
    loss = torch.sqrt(diff * diff + eps).sum() / active_elements
    return -loss
