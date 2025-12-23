from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmvrt.registry import MODELS


@MODELS.register_module()
class CharbonnierLoss(nn.Module):
    """Standard Charbonnier (L1-like) loss used in many restoration models."""

    def __init__(self, eps: float = 1e-3, weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def forward(self, outputs, batch):
        pred = outputs["pred"]
        gt = batch["gt"]
        loss = torch.sqrt((pred - gt).pow(2) + self.eps).mean()
        return loss * self.weight


@MODELS.register_module()
class SSIMLoss(nn.Module):
    """SSIM loss wrapper built on the frame-wise SSIM helper."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, outputs, batch):
        pred = outputs["pred"]
        gt = batch["gt"]
        return (1 - ssim(pred, gt)) * self.weight


# Simple SSIM for 3D data treated frame-wise
def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = pred.shape
    pred = pred.reshape(b * t, c, h, w)
    gt = gt.reshape(b * t, c, h, w)
    return _ssim_2d(pred, gt).mean()


def _ssim_2d(x: torch.Tensor, y: torch.Tensor, window_size: int = 7, c1=0.01**2, c2=0.03**2):
    mu_x = F.avg_pool2d(x, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, 1, window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, 1, window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, 1, window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, window_size // 2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean([1, 2, 3])


