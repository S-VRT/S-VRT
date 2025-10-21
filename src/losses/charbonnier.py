from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    L_charb(x, y) = mean( sqrt( (x - y)^2 + delta^2 ) ) over all elements.
    Default reduction is mean over batch and all dims.
    """

    def __init__(self, delta: float = 1e-3) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = input - target
        loss = torch.sqrt(diff * diff + (self.delta * self.delta))
        return loss.mean()



