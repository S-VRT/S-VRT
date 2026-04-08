from typing import Protocol

import torch


class FusionOperator(Protocol):
    def __call__(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        ...


def validate_mode(mode: str) -> str:
    normalized = str(mode).lower().strip()
    if normalized not in ('replace', 'residual'):
        raise ValueError(f"Unsupported fusion mode: {normalized}")
    return normalized


__all__ = ['FusionOperator', 'validate_mode']
