from __future__ import annotations

import torch


def _flatten_spatial_feature(feature: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    data = feature.detach().float()
    if data.ndim == 5:
        data = data[0, data.shape[1] // 2]
    elif data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected CHW feature tensor, got {tuple(feature.shape)}")
    channels, height, width = data.shape
    matrix = data.reshape(channels, height * width).transpose(0, 1)
    return matrix, height, width


def pca_variance_ratio(feature: torch.Tensor) -> torch.Tensor:
    matrix, _, _ = _flatten_spatial_feature(feature)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, _ = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    return energy / total


def pca_feature_heatmap(feature: torch.Tensor, component: int = 0) -> torch.Tensor:
    matrix, height, width = _flatten_spatial_feature(feature)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    if component < 0 or component >= vh.shape[0]:
        raise ValueError(f"component out of range: {component}")
    projection = centered @ vh[component].unsqueeze(1)
    return projection.reshape(height, width)
