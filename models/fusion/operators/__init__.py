from typing import Any

from ..base import validate_mode


class ConcatFusionOperator:
    def __init__(self, placement: str = 'replace', **kwargs: Any):
        validate_mode(placement)
        self.placement = placement
        self.kwargs = kwargs


def build_operator(name: str, placement: str = 'replace', **kwargs: Any):
    if name == 'concat':
        return ConcatFusionOperator(placement=placement, **kwargs)
    raise ValueError(f"Unknown fusion operator: {name}")


__all__ = ['build_operator', 'ConcatFusionOperator']
