from typing import Any

from ..base import validate_mode
from .early import EarlyFusionAdapter
from .hybrid import HybridFusionAdapter
from .middle import MiddleFusionAdapter


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
    if placement == 'early':
        return EarlyFusionAdapter(
            operator=operator,
            mode=canonical_mode,
            inject_stages=inject_stages,
            **kwargs,
        )
    if placement == 'middle':
        return MiddleFusionAdapter(
            operator=operator,
            mode=canonical_mode,
            inject_stages=inject_stages,
            **kwargs,
        )
    early_operator = kwargs.pop('early_operator', None)
    middle_operator = kwargs.pop('middle_operator', None)
    if early_operator is None or middle_operator is None:
        raise ValueError("Hybrid fusion requires both early_operator and middle_operator.")
    return HybridFusionAdapter(
        early_operator=early_operator,
        middle_operator=middle_operator,
        mode=canonical_mode,
        inject_stages=inject_stages,
        **kwargs,
    )


__all__ = [
    'build_adapter',
    'IdentityFusionAdapter',
    'EarlyFusionAdapter',
    'MiddleFusionAdapter',
    'HybridFusionAdapter',
]
