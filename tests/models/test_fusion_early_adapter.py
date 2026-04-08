import torch

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


def test_early_adapter_expands_time():
    op = create_fusion_operator("concat", 3, 1, 3, {})
    adapter = EarlyFusionAdapter(operator=op)
    rgb = torch.randn(2, 6, 3, 12, 12)
    spike = torch.randn(2, 6, 8, 12, 12)
    out = adapter(rgb=rgb, spike=spike)
    assert out.shape == (2, 48, 3, 12, 12)
