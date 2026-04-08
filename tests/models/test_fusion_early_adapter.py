import torch

from models.fusion.factory import create_fusion_operator


def test_concat_operator_shape():
    op = create_fusion_operator(
        operator_name='concat',
        rgb_chans=3,
        spike_chans=8,
        out_chans=3,
        operator_params={},
    )
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 8, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)


def test_gated_operator_shape():
    op = create_fusion_operator(
        operator_name='gated',
        rgb_chans=3,
        spike_chans=8,
        out_chans=3,
        operator_params={},
    )
    rgb_feat = torch.randn(2, 5, 3, 16, 16)
    spike_feat = torch.randn(2, 5, 8, 16, 16)
    out = op(rgb_feat, spike_feat)
    assert out.shape == (2, 5, 3, 16, 16)
