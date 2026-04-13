import torch
import pytest

from models.fusion.factory import create_fusion_operator
from models.fusion.adapters.early import EarlyFusionAdapter


def test_concat_operator_shape():
    op = create_fusion_operator('concat', 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_gated_operator_shape():
    op = create_fusion_operator('gated', 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_pase_operator_shape():
    op = create_fusion_operator('pase', 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_mamba_operator_shape_or_missing_dep():
    op = create_fusion_operator('mamba', 3, 1, 3, {})
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 1, 16, 16)
    try:
        out = op(rgb_feat, spike_feat)
    except RuntimeError as exc:
        assert 'mamba_ssm is required' in str(exc)
    else:
        assert out.shape == (2, 5, 3, 16, 16)


def test_early_adapter_expands_time():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)


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
