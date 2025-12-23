from typing import Any, Optional
import torch
import torch.nn as nn

from mmvrt.registry import MODELS

try:
    from .vrt_backbone import VRTBackbone
except Exception:
    # Fallback import path for environments where relative imports differ
    from mmvrt.models.backbones.vrt_backbone import VRTBackbone


@MODELS.register_module()
class SGPVRTBackbone(nn.Module):
    """SGP variant wrapper for the standard VRT backbone.

    This small wrapper builds the existing `VRTBackbone` with `use_sgp=True`.
    It allows selecting the SGP variant from configs with `type='SGPVRTBackbone'`.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Force use_sgp flag for this variant while forwarding other args.
        kwargs = dict(kwargs)
        kwargs.setdefault('use_sgp', True)
        super().__init__()
        self.inner = VRTBackbone(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)

    def init_weights(self, pretrained: Optional[str] = None, strict: bool = True):
        if hasattr(self.inner, 'init_weights'):
            self.inner.init_weights(pretrained=pretrained, strict=strict)


