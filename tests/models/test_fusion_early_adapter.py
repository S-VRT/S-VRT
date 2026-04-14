import pytest
import torch
from torch import nn

from models.fusion.adapters.early import EarlyFusionAdapter, SpikeUpsample
from models.fusion.factory import create_fusion_adapter, create_fusion_operator


class RecordingOperator(nn.Module):
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


def test_early_adapter_expands_time():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)


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
    assert all(grad.abs().sum() > 0 for grad in grads)


def test_early_adapter_spatial_mismatch():
    op = RecordingOperator()
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)
    assert op.last_rgb.shape == (2, 48, 3, 12, 12)
    assert op.last_spike.shape == (2, 48, 1, 12, 12)


def test_early_adapter_spatial_mismatch_no_spike_chans_raises():
    op = RecordingOperator()
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    with pytest.raises(ValueError, match=r"(?i)(unable|cannot) upsample|(?i)(unable|cannot) to upsample"):
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
    assert adapter.spike_upsample is not None

    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)


def test_early_adapter_gated_spatial_mismatch():
    op = create_fusion_operator("gated", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)


def test_early_adapter_output_is_3_channels():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op, spike_chans=8)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 6, 6)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape[2] == 3


@pytest.mark.parametrize("operator_name", ["gated", "pase", "mamba"])
def test_early_adapter_supports_all_early_operators(operator_name):
    op = create_fusion_operator(operator_name, 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    try:
        out = adapter(rgb=rgb, spike=spike)
    except RuntimeError as exc:
        if operator_name == "mamba":
            assert "mamba_ssm is required" in str(exc)
            return
        raise
    assert out.shape == (2, 48, 3, 12, 12)
