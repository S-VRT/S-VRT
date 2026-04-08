from typing import Any

class ConcatFusionOperator:
    def __init__(
        self,
        rgb_chans: int,
        spike_chans: int,
        out_chans: int,
        operator_params: dict,
    ):
        self.rgb_chans = rgb_chans
        self.spike_chans = spike_chans
        self.out_chans = out_chans
        self.operator_params = operator_params

    def __call__(self, rgb_feat, spike_feat):
        return rgb_feat


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
    raise ValueError(f"Unknown fusion operator: {operator_name}")


__all__ = ['build_operator', 'ConcatFusionOperator']
