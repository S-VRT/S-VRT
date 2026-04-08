import pytest
import torch

from models.fusion.factory import create_fusion_operator, create_fusion_adapter


def test_create_fusion_operator_concat():
    op = create_fusion_operator(
        operator_name='concat',
        rgb_chans=3,
        spike_chans=8,
        out_chans=3,
        operator_params={},
    )
    assert callable(op)
    assert op is not None
    rgb_feat = torch.randn(2, 3, 8, 8)
    spike_feat = torch.randn(2, 8, 8, 8)
    out = op(rgb_feat, spike_feat)
    assert isinstance(out, torch.Tensor)
    assert out.shape == rgb_feat.shape


def test_create_fusion_operator_unknown():
    with pytest.raises(ValueError, match='Unknown fusion operator'):
        create_fusion_operator(
            operator_name='unknown',
            rgb_chans=3,
            spike_chans=8,
            out_chans=3,
            operator_params={},
        )


def test_pase_operator_constructs():
    op = create_fusion_operator(
        operator_name='pase',
        rgb_chans=3,
        spike_chans=8,
        out_chans=3,
        operator_params={},
    )
    assert op is not None


def test_mamba_operator_missing_dep_raises_runtime():
    try:
        import mamba_ssm  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        op = create_fusion_operator(
            operator_name='mamba',
            rgb_chans=3,
            spike_chans=8,
            out_chans=3,
            operator_params={},
        )
        with pytest.raises(RuntimeError, match='mamba_ssm is required'):
            op(torch.randn(1, 2, 3, 8, 8), torch.randn(1, 2, 8, 8, 8))
    else:
        pytest.skip('mamba_ssm is available; missing-dependency guard is not applicable.')


def test_create_fusion_adapter_unknown_placement():
    with pytest.raises(ValueError, match='Unknown fusion placement'):
        create_fusion_adapter(
            placement='unknown',
            operator=None,
            mode='replace',
            inject_stages=[],
        )
