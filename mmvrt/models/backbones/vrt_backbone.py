"""Lightweight VRT backbone wrapper for mmvrt.

This simplified backbone ensures the package can be imported reliably for
smoke tests. It prefers a native assembly (`network_vrt.VRT`) if available,
otherwise falls back to a minimal placeholder that raises at forward time.
"""

from typing import Any, Dict, Optional
import torch.nn as nn

from mmvrt.registry import MODELS

from mmvrt.models.backbones.vrt_network import VRT as VRTNet


@MODELS.register_module()
class VRTBackbone(nn.Module):
    """Wrapper that instantiates the canonical migrated VRT network.

    This implementation intentionally removes legacy fallbacks and requires the
    migrated `VRT` implementation to be available under
    `mmvrt.models.backbones.vrt_network.VRT`.
    """

    def __init__(self,
                 net_cfg: Optional[Dict[str, Any]] = None,
                 init_cfg: Optional[Dict] = None,
                 **kwargs: Any):
        super().__init__()
        cfg: Dict[str, Any] = {}
        if net_cfg and isinstance(net_cfg, dict):
            cfg.update(net_cfg)
        cfg.update(kwargs or {})

        # Instantiate the canonical migrated VRT implementation.
        try:
            self.net = VRTNet(**cfg)
        except Exception as e:
            raise ImportError("Failed to instantiate migrated VRT from "
                              "`mmvrt.models.backbones.vrt_network.VRT`.") from e

    def forward(self, x):
        return self.net(x)

