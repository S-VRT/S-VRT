from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
except Exception:  # pragma: no cover
    models = None  # type: ignore


class _VGGFeature(nn.Module):
    def __init__(self, layers: List[str]) -> None:
        super().__init__()
        assert models is not None, "torchvision is required for VGGPerceptualLoss"
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        vgg.features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Map layer names to indices in vgg.features
        name_to_idx = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
            "relu5_3": 29,
        }
        max_idx = max(name_to_idx[l] for l in layers)
        self.features = nn.Sequential(*list(vgg.features.children())[: max_idx + 1])
        self.target_indices = [name_to_idx[l] for l in layers]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.target_indices:
                feats.append(x)
        return feats


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using selected VGG layers.
    Input expects images normalized to ImageNet stats in [0,1] range.
    """

    def __init__(self, layers: List[str] | None = None) -> None:
        super().__init__()
        if layers is None:
            layers = ["relu3_3"]
        self.vgg = _VGGFeature(layers)
        self.layers = layers

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Merge batch and time if 5D
        if input.dim() == 5 and input.shape[2] >= 1:
            b, t = input.shape[0], input.shape[1]
            input = input.reshape(b * t, *input.shape[2:])
            target = target.reshape(b * t, *target.shape[2:])

        x = self._normalize(input)
        y = self._normalize(target)
        fx = self.vgg(x)
        fy = self.vgg(y)
        loss = 0.0
        for a, b in zip(fx, fy):
            loss = loss + F.l1_loss(a, b)
        return loss / max(len(fx), 1)



