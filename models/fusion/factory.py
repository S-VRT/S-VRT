from typing import Any

from .adapters import build_adapter
from .operators import build_operator


def create_fusion_operator(name: str, placement: str = 'replace', **kwargs: Any):
    return build_operator(name, placement=placement, **kwargs)


def create_fusion_adapter(name: str, placement: str = 'replace', **kwargs: Any):
    return build_adapter(name, placement=placement, **kwargs)


__all__ = ['create_fusion_operator', 'create_fusion_adapter']
