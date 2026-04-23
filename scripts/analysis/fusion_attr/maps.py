from __future__ import annotations

import numpy as np
import torch


def reduce_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim == 3:
        return torch.linalg.vector_norm(data, dim=0)
    if data.ndim == 2:
        return data
    raise ValueError(f"Expected 2D, 3D, 4D, or 5D tensor, got {tuple(tensor.shape)}")


def normalize_map(values: torch.Tensor, low: float = 1.0, high: float = 99.0) -> torch.Tensor:
    arr = values.detach().float().cpu().numpy()
    lo = float(np.percentile(arr, low))
    hi = float(np.percentile(arr, high))
    if hi <= lo:
        return torch.zeros_like(values, dtype=torch.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return torch.from_numpy(out).to(dtype=torch.float32, device=values.device)


def compute_fusion_delta(fusion_output: torch.Tensor, rgb_reference: torch.Tensor) -> torch.Tensor:
    if fusion_output.shape != rgb_reference.shape:
        raise ValueError(
            f"fusion_output and rgb_reference shapes differ: {tuple(fusion_output.shape)} vs {tuple(rgb_reference.shape)}"
        )
    return reduce_to_2d(fusion_output - rgb_reference)


def compute_error_map(output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if output.shape != gt.shape:
        raise ValueError(f"output and gt shapes differ: {tuple(output.shape)} vs {tuple(gt.shape)}")
    data = (output.detach().float() - gt.detach().float()).abs()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected image tensor with channels, got {tuple(output.shape)}")
    return data.mean(dim=0)
