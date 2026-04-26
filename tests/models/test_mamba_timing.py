import pytest
import torch


class _TimerStub:
    def __init__(self):
        self.names = []

    def timer(self, name):
        self.names.append(name)

        class _Ctx:
            def __enter__(_self):
                return None

            def __exit__(_self, *_exc):
                return False

        return _Ctx()


def test_mamba_operator_records_forward_stage_timings(monkeypatch):
    from models.fusion.operators import mamba as mamba_module
    from models.fusion.operators.mamba import MambaFusionOperator

    class _FakeBlock(torch.nn.Module):
        def __init__(self, model_dim, d_state, d_conv, expand):
            super().__init__()

        def forward(self, tokens):
            return tokens

    monkeypatch.setattr(mamba_module, "_MambaBlock", _FakeBlock)

    operator = MambaFusionOperator(
        rgb_chans=3,
        spike_chans=1,
        out_chans=3,
        operator_params={"token_dim": 4, "token_stride": 2, "num_layers": 1},
    )
    timer = _TimerStub()
    operator.set_timer(timer)

    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)
    operator(rgb, spike)

    assert timer.names == [
        "mamba_rgb_encoder",
        "mamba_spike_encoder",
        "mamba_token_pack",
        "mamba_mixer",
        "mamba_writeback",
        "mamba_upsample",
    ]
