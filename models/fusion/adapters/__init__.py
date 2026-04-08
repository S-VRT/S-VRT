from typing import Any

from ..base import validate_mode


class IdentityFusionAdapter:
    def __init__(self, placement: str = 'replace', **kwargs: Any):
        validate_mode(placement)
        self.placement = placement
        self.kwargs = kwargs


def build_adapter(name: str, placement: str = 'replace', **kwargs: Any):
    if name == 'identity':
        return IdentityFusionAdapter(placement=placement, **kwargs)
    raise ValueError(f"Unknown fusion adapter: {name}")


__all__ = ['build_adapter', 'IdentityFusionAdapter']
