import pytest
import torch
from torch import nn

from models.fusion.adapters.early import EarlyFusionAdapter, SpikeUpsample
from models.fusion.factory import create_fusion_adapter, create_fusion_operator


class RecordingOperator(nn.Module):
    frame_contract = "expanded"

    def __init__(self):
        super().__init__()
        self.last_rgb = None
        self.last_spike = None

    def forward(self, rgb, spike):
        self.last_rgb = rgb
        self.last_spike = spike
        return rgb


class StructuredRecordingOperator(nn.Module):
    frame_contract = "collapsed"
    expects_structured_early = True

    def __init__(self):
        super().__init__()
        self.last_rgb = None
        self.last_spike = None

    def forward(self, rgb, spike):
        self.last_rgb = rgb
        self.last_spike = spike
        return rgb


def test_concat_operator_shape():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


@pytest.mark.parametrize(
    ("operator_name", "expected_frame_contract"),
    [
        ("gated", "expanded"),
        ("concat", "expanded"),
        ("pase", "expanded"),
        ("mamba", "collapsed"),
    ],
)
def test_fusion_operator_frame_contract_metadata(operator_name, expected_frame_contract):
    op = create_fusion_operator(operator_name, 3, 1, 3, {})
    assert getattr(op, "frame_contract", None) == expected_frame_contract


def test_gated_operator_shape():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_pase_operator_shape():
    op = create_fusion_operator("pase", 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_mamba_operator_shape_or_missing_dep():
    op = create_fusion_operator("mamba", 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    try:
        out = op(rgb_feat, spike_feat)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
    else:
        assert out.shape == (2, 5, 3, 16, 16)


def test_early_adapter_returns_main_and_exec_for_expanded_operator():
    class ExpandedStub(torch.nn.Module):
        frame_contract = "expanded"

        def forward(self, rgb_rep, spk):
            return rgb_rep

    adapter = EarlyFusionAdapter(operator=ExpandedStub(), mode="replace", inject_stages=[], spike_chans=4)
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb, spike)

    assert set(result.keys()) == {"fused_main", "backbone_view", "aux_view", "meta"}
    assert result["meta"]["frame_contract"] == "expanded"
    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 8, 3, 8, 8)
    assert result["aux_view"].shape == (1, 8, 3, 8, 8)
    assert result["meta"]["main_from_exec_rule"] == "center_subframe"


def test_early_adapter_returns_main_and_exec_for_collapsed_operator():
    class CollapsedStub(torch.nn.Module):
        frame_contract = "collapsed"

        def forward(self, rgb, spike):
            return rgb

    adapter = EarlyFusionAdapter(operator=CollapsedStub(), mode="replace", inject_stages=[], spike_chans=4)
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)

    result = adapter(rgb, spike)

    assert result["meta"]["frame_contract"] == "collapsed"
    assert result["fused_main"].shape == (1, 2, 3, 8, 8)
    assert result["backbone_view"].shape == (1, 2, 3, 8, 8)
    assert result["aux_view"] is None
    assert result["meta"]["main_from_exec_rule"] is None


def test_early_adapter_expands_time():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 48, 3, 12, 12)


def test_spike_upsample_shape():
    upsample = SpikeUpsample(spike_chans=8)
    spike = torch.randn(10, 8, 6, 6)
    out = upsample(spike, target_h=12, target_w=12)
    assert out.shape == (10, 8, 12, 12)


def test_spike_upsample_preserves_channels():
    upsample = SpikeUpsample(spike_chans=8)
    spike = torch.randn(4, 8, 5, 7)
    out = upsample(spike, target_h=9, target_w=11)
    assert out.shape[1] == spike.shape[1]


def test_spike_upsample_gradient_flows():
    upsample = SpikeUpsample(spike_chans=8)
    spike = torch.randn(3, 8, 4, 4, requires_grad=True)
    out = upsample(spike, target_h=10, target_w=10)
    out.sum().backward()

    grads = [param.grad for param in upsample.refine.parameters()]
    assert grads
    assert all(grad is not None for grad in grads)


def test_early_adapter_spatial_mismatch():
    op = RecordingOperator()
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 48, 3, 12, 12)
    assert op.last_rgb.shape == (2, 48, 3, 12, 12)
    assert op.last_spike.shape == (2, 48, 1, 12, 12)


def test_early_adapter_spatial_mismatch_no_spike_chans_raises():
    op = RecordingOperator()
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    with pytest.raises(ValueError, match=r"(?i)((unable|cannot) upsample|(unable|cannot) to upsample)"):
        adapter(rgb=rgb, spike=spike)


def test_early_adapter_receives_spike_chans_via_factory():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = create_fusion_adapter(
        placement="early",
        operator=op,
        mode="replace",
        inject_stages=[],
        spike_chans=8,
    )
    assert isinstance(adapter, EarlyFusionAdapter)
    assert adapter.spike_chans == 8

    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 48, 3, 12, 12)


def test_early_adapter_gated_spatial_mismatch():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 48, 3, 12, 12)


def test_early_adapter_output_is_3_channels():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape[2] == 3
    assert result["backbone_view"].shape[2] == 3


@pytest.mark.parametrize(
    ("operator_name", "expected_main_steps", "expected_exec_steps"),
    [
        ("gated", 6, 48),
        ("pase", 6, 48),
        ("mamba", 6, 6),
    ],
)
def test_early_adapter_supports_all_early_operators(operator_name, expected_main_steps, expected_exec_steps):
    op = create_fusion_operator(operator_name, 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    try:
        result = adapter(rgb=rgb, spike=spike)
    except RuntimeError as exc:
        if operator_name == "mamba":
            assert "mamba_ssm is required" in str(exc)
            return
        raise
    assert result["fused_main"].shape == (2, expected_main_steps, 3, 12, 12)
    assert result["backbone_view"].shape == (2, expected_exec_steps, 3, 12, 12)


def test_gated_operator_passthrough_at_init():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    rgb = torch.ones(1, 1, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 1, 1, 8, 8)
    with torch.no_grad():
        out = op(rgb, spike)
    assert torch.allclose(out, rgb, atol=1e-6), (
        f"Expected near-passthrough at init, max diff={(out - rgb).abs().max():.4f}"
    )


def test_gated_operator_no_rgb_proj():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    assert not hasattr(op, "rgb_proj"), "rgb_proj should be removed from GatedFusionOperator"


def test_gated_operator_has_correction():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    assert hasattr(op, "correction"), "GatedFusionOperator must have 'correction' attribute"


def test_mamba_operator_structured_early_shape_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "model_dim": 48,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 3,
        },
    )
    rgb = torch.randn(2, 5, 3, 12, 12)
    spike = torch.randn(2, 5, 8, 12, 12)
    try:
        out = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return
    assert out.shape == (2, 5, 3, 12, 12)


def test_mamba_operator_small_output_non_degenerate_init_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "token_dim": 24,
            "token_stride": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 1,
            "alpha_init": 0.05,
            "gate_bias_init": -2.0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.ones(1, 2, 3, 8, 8) * 0.5
    spike = torch.zeros(1, 2, 6, 8, 8)

    try:
        out = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return

    max_diff = (out - rgb).abs().max().item()
    diagnostics = op.diagnostics()
    assert 1e-7 < max_diff < 1e-2
    assert 1e-7 < diagnostics["effective_update_norm"] < 1e-2

    loss = (out - rgb).abs().mean()
    loss.backward()

    delta_grad = op.fusion_writeback_head["delta"].weight.grad
    gate_grad = op.fusion_writeback_head["gate"].weight.grad
    assert delta_grad is not None and delta_grad.abs().sum().item() > 0.0
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0.0


def test_early_adapter_mamba_keeps_frame_structure():
    op = StructuredRecordingOperator()
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 6, 3, 12, 12)
    assert op.last_rgb.shape == (2, 6, 3, 12, 12)
    assert op.last_spike.shape == (2, 6, 8, 12, 12)


def test_early_adapter_mamba_upsamples_without_flattening():
    op = StructuredRecordingOperator()
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    result = adapter(rgb=rgb, spike=spike)
    assert result["fused_main"].shape == (2, 6, 3, 12, 12)
    assert result["backbone_view"].shape == (2, 6, 3, 12, 12)
    assert op.last_spike.shape == (2, 6, 8, 12, 12)


def test_early_wrapper_inferrs_contract_from_operator_without_new_config():
    op = create_fusion_operator("mamba", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, mode="replace", inject_stages=[], spike_chans=4)
    assert getattr(adapter.operator, "frame_contract", None) == "collapsed"


def test_mamba_operator_exposes_scalar_diagnostics_or_missing_dep():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {
            "token_dim": 24,
            "token_stride": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_layers": 1,
            "alpha_init": 0.05,
            "gate_bias_init": -2.0,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 12, 12)
    spike = torch.randn(1, 2, 6, 12, 12)

    try:
        _ = op(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
        return

    diagnostics = op.diagnostics()
    assert diagnostics["warmup_stage"] == "full"
    assert set(diagnostics) >= {
        "token_norm",
        "mamba_norm",
        "delta_norm",
        "gate_mean",
        "effective_update_norm",
        "warmup_stage",
    }
    assert all(isinstance(diagnostics[key], float) for key in diagnostics if key != "warmup_stage")


def test_mamba_operator_writeback_only_stage_freezes_token_mixers():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1},
    )

    op.set_warmup_stage("writeback_only")

    assert all(not p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(not p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(not p.requires_grad for p in op.mamba_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True


def test_mamba_operator_token_mixer_stage_unfreezes_temporal_stack():
    op = create_fusion_operator(
        "mamba",
        3,
        1,
        3,
        {"token_dim": 16, "token_stride": 2, "num_layers": 1},
    )

    op.set_warmup_stage("token_mixer")

    assert all(p.requires_grad for p in op.rgb_context_encoder.parameters())
    assert all(p.requires_grad for p in op.spike_token_encoder.parameters())
    assert all(p.requires_grad for p in op.mamba_token_mixer.parameters())
    assert all(p.requires_grad for p in op.fusion_writeback_head.parameters())
    assert op.alpha.requires_grad is True
