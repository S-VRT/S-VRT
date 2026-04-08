from typing import Any

from ..base import validate_mode


class IdentityFusionAdapter:
    def __init__(
        self,
        placement: str,
        operator: Any,
        mode: str,
        inject_stages: list,
        **kwargs: Any,
    ):
        self.placement = placement
        self.operator = operator
        self.mode = mode
        self.inject_stages = inject_stages
        self.kwargs = kwargs


def build_adapter(
    placement: str,
    operator: Any,
    mode: str,
    inject_stages: list,
    **kwargs: Any,
):
    known_placements = {'early', 'middle', 'hybrid'}
    if placement not in known_placements:
        raise ValueError(f"Unknown fusion placement: {placement}")
    canonical_mode = validate_mode(mode)
    return IdentityFusionAdapter(
        placement=placement,
        operator=operator,
        mode=canonical_mode,
        inject_stages=inject_stages,
        **kwargs,
    )


__all__ = ['build_adapter', 'IdentityFusionAdapter']
