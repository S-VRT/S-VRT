from typing import Any

from .attention import AttentionFusionOperator
from .concat import ConcatFusionOperator
from .gated import GatedFusionOperator
from .mamba import MambaFusionOperator
from .pase import PaseFusionOperator
from .pase_residual import PaseResidualFusionOperator


def build_operator(
    operator_name: str,
    rgb_chans: int,
    spike_chans: int,
    out_chans: int,
    operator_params: dict,
):
    normalized_name = str(operator_name).lower().strip()
    if normalized_name == 'concat':
        return ConcatFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if normalized_name == 'gated':
        return GatedFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if normalized_name == 'pase':
        return PaseFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if normalized_name == 'mamba':
        return MambaFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if normalized_name == 'attention':
        return AttentionFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    if normalized_name == 'pase_residual':
        return PaseResidualFusionOperator(
            rgb_chans=rgb_chans,
            spike_chans=spike_chans,
            out_chans=out_chans,
            operator_params=operator_params,
        )
    raise ValueError(f"Unknown fusion operator: {operator_name}")


__all__ = [
    'build_operator',
    'AttentionFusionOperator',
    'ConcatFusionOperator',
    'GatedFusionOperator',
    'MambaFusionOperator',
    'PaseFusionOperator',
    'PaseResidualFusionOperator',
]
