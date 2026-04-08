from typing import Any

from .concat import ConcatFusionOperator
from .gated import GatedFusionOperator


def build_operator(
    operator_name: str,
    rgb_chans: int,
    spike_chans: int,
    out_chans: int,
    operator_params: dict,
):
    if operator_name == 'concat':
        return ConcatFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if operator_name == 'gated':
        return GatedFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    raise ValueError(f"Unknown fusion operator: {operator_name}")


__all__ = ['build_operator', 'ConcatFusionOperator', 'GatedFusionOperator']
