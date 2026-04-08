from typing import Any, Dict

from .adapters import build_adapter
from .operators import build_operator


def create_fusion_operator(
    operator_name: str,
    rgb_chans: int,
    spike_chans: int,
    out_chans: int,
    operator_params: Dict[str, Any],
):
    return build_operator(
        operator_name=operator_name,
        rgb_chans=rgb_chans,
        spike_chans=spike_chans,
        out_chans=out_chans,
        operator_params=operator_params,
    )


def create_fusion_adapter(
    placement: str,
    operator: Any,
    mode: str,
    inject_stages: list,
    **kwargs: Any,
):
    return build_adapter(
        placement=placement,
        operator=operator,
        mode=mode,
        inject_stages=inject_stages,
        **kwargs,
    )


__all__ = ['create_fusion_operator', 'create_fusion_adapter']
