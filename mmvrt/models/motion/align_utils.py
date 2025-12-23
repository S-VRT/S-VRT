"""Alignment utilities for image/feature warping (mirrors legacy get_aligned_image_2frames)."""
from typing import Tuple
import torch
from mmvrt.models.motion.flow_ops import flow_warp


def get_aligned_image_2frames(x: torch.Tensor, flows_backward: torch.Tensor, flows_forward: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel image warping for 2-frame alignment.

    Args:
        x: Tensor of shape (B, N, C, H, W)
        flows_backward: Tensor of shape (B, N-1, 2, H, W)
        flows_forward: Tensor of shape (B, N-1, 2, H, W)

    Returns:
        x_backward, x_forward: each shape (B, N, C', H, W) where C' depends on alignment mode.
    """
    n = x.size(1)
    # backward
    # legacy pads used 4 repeats for nearest4 behavior; keep same shape contract
    x_backward = [torch.zeros_like(x[:, -1, ...]).repeat(1, 4, 1, 1)]
    for i in range(n - 1, 0, -1):
        x_i = x[:, i, ...]
        flow = flows_backward[:, i - 1, ...]
        x_i_warped = flow_warp(x_i, flow.permute(0, 2, 3, 1), interp_mode='nearest4')
        x_backward.insert(0, x_i_warped)

    # forward
    x_forward = [torch.zeros_like(x[:, 0, ...]).repeat(1, 4, 1, 1)]
    for i in range(0, n - 1):
        x_i = x[:, i, ...]
        flow = flows_forward[:, i, ...]
        x_forward.append(flow_warp(x_i, flow.permute(0, 2, 3, 1), interp_mode='nearest4'))

    x_backward = torch.stack(x_backward, 1)
    x_forward = torch.stack(x_forward, 1)
    return [x_backward, x_forward]


