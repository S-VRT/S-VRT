import pytest
import torch


class _TimerStub:
    def __init__(self, record_ranges=False):
        self.names = []
        self.range_names = []
        self.record_ranges = record_ranges

    def timer(self, name):
        self.names.append(name)

        class _Ctx:
            def __enter__(_self):
                return None

            def __exit__(_self, *_exc):
                return False

        return _Ctx()

    def profile_range(self, name):
        if self.record_ranges:
            self.range_names.append(name)

        class _Ctx:
            def __enter__(_self):
                return None

            def __exit__(_self, *_exc):
                return False

        return _Ctx()


def test_mamba_operator_uses_timer_only_when_attached(monkeypatch):
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
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)
    operator(rgb, spike)

    timer = _TimerStub()
    operator.set_timer(timer)
    operator(rgb, spike)

    assert timer.names == [
        "mamba_rgb_encoder",
        "mamba_spike_encoder",
        "mamba_token_pack",
        "mamba_mixer",
        "mamba_writeback",
        "mamba_upsample",
    ]


def test_mamba_operator_profile_ranges_are_config_gated(monkeypatch):
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

    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    timer = _TimerStub(record_ranges=False)
    operator.set_timer(timer)
    operator(rgb, spike)
    assert timer.range_names == []

    timer = _TimerStub(record_ranges=True)
    operator.set_timer(timer)
    operator(rgb, spike)

    assert timer.range_names == [
        "mamba_rgb_encoder",
        "mamba_spike_encoder",
        "mamba_token_pack",
        "mamba_mixer",
        "mamba_writeback",
        "mamba_upsample",
    ]


def test_mamba_operator_defaults_to_fp32_mixer_policy(monkeypatch):
    from models.fusion.operators import mamba as mamba_module
    from models.fusion.operators.mamba import MambaFusionOperator

    class _FakeBlock(torch.nn.Module):
        def __init__(self, model_dim, d_state, d_conv, expand):
            super().__init__()

        def forward(self, tokens):
            assert tokens.dtype == torch.float32
            return tokens

    monkeypatch.setattr(mamba_module, "_MambaBlock", _FakeBlock)

    operator = MambaFusionOperator(3, 1, 3, {"token_dim": 4, "token_stride": 2, "num_layers": 1})

    assert operator.mamba_amp_policy == "fp32"


def test_mamba_operator_disables_diagnostics_by_default(monkeypatch):
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

    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)
    operator(rgb, spike)

    assert operator.enable_diagnostics is False
    assert operator.diagnostics() == {"warmup_stage": "full"}


def test_mamba_operator_rejects_unknown_amp_policy():
    from models.fusion.operators.mamba import MambaFusionOperator

    try:
        MambaFusionOperator(3, 1, 3, {"mamba_amp_policy": "bad"})
    except ValueError as exc:
        assert "mamba_amp_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad mamba_amp_policy")
