from typing import Any, Optional

from torch import nn

from .early import EarlyFusionAdapter
from .middle import MiddleFusionAdapter


class HybridFusionAdapter(nn.Module):
    def __init__(
        self,
        early_operator: nn.Module,
        middle_operator: nn.Module,
        mode: str = "replace",
        inject_stages: Optional[list] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.early_adapter = EarlyFusionAdapter(
            operator=early_operator,
            mode=mode,
            inject_stages=inject_stages,
            **kwargs,
        )
        self.middle_adapter = MiddleFusionAdapter(
            operator=middle_operator,
            mode=mode,
            inject_stages=inject_stages,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        if "stage_idx" in kwargs or (args and isinstance(args[0], int)):
            return self.middle_adapter(*args, **kwargs)
        return self.early_adapter(*args, **kwargs)

    def early(self, rgb, spike):
        return self.early_adapter(rgb=rgb, spike=spike)

    def middle(self, stage_idx, x, spike_ctx):
        return self.middle_adapter(stage_idx=stage_idx, x=x, spike_ctx=spike_ctx)


__all__ = ["HybridFusionAdapter"]
