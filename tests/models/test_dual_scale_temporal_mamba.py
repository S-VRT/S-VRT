import pytest
import torch

from models.fusion.factory import create_fusion_operator


def test_factory_builds_dual_scale_temporal_mamba():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={"token_dim": 8, "patch_stride": 4, "local_layers": 1, "global_layers": 1},
    )

    assert operator.__class__.__name__ == "DualScaleTemporalMambaFusionOperator"
    assert operator.frame_contract == "collapsed"
    assert operator.expects_structured_early is True
    assert operator.spike_chans == 21


def test_dual_scale_temporal_mamba_rejects_non_rgb3():
    with pytest.raises(ValueError, match="rgb_chans=3"):
        create_fusion_operator(
            "dual_scale_temporal_mamba",
            rgb_chans=4,
            spike_chans=21,
            out_chans=3,
            operator_params={},
        )


def test_dual_scale_temporal_mamba_rejects_non_positive_spike_chans():
    with pytest.raises(ValueError, match="spike_chans>0"):
        create_fusion_operator(
            "dual_scale_temporal_mamba",
            rgb_chans=3,
            spike_chans=0,
            out_chans=3,
            operator_params={},
        )


def test_dual_scale_temporal_mamba_shape_contract_or_missing_dep():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={"token_dim": 8, "patch_stride": 4, "local_layers": 1, "global_layers": 1},
    )
    rgb = torch.randn(1, 3, 3, 16, 16)
    spike = torch.randn(1, 3, 21, 16, 16)

    try:
        out = operator(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
    else:
        assert out.shape == rgb.shape


def test_dual_scale_temporal_mamba_exposes_diagnostics_or_missing_dep():
    operator = create_fusion_operator(
        "dual_scale_temporal_mamba",
        rgb_chans=3,
        spike_chans=21,
        out_chans=3,
        operator_params={
            "token_dim": 8,
            "patch_stride": 4,
            "local_layers": 1,
            "global_layers": 1,
            "enable_diagnostics": True,
        },
    )
    rgb = torch.randn(1, 2, 3, 16, 16)
    spike = torch.randn(1, 2, 21, 16, 16)

    try:
        _ = operator(rgb, spike)
    except RuntimeError as exc:
        assert "mamba_ssm is required" in str(exc)
    else:
        diagnostics = operator.diagnostics()
        assert "local_norm" in diagnostics
        assert "global_norm" in diagnostics
        assert "summary_gate_mean" in diagnostics
        assert "effective_update_norm" in diagnostics
