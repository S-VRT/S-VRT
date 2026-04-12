import pytest
import torch
import torch.nn as nn

from models.architectures.vrt.vrt import VRT
from models.fusion.adapters.hybrid import HybridFusionAdapter
from models.fusion.adapters.middle import MiddleFusionAdapter


def test_vrt_builds_with_fusion_config():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 4,
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    assert model is not None
    assert hasattr(model, "fusion_enabled")
    assert model.fusion_enabled is True
    assert hasattr(model, "fusion_operator")
    assert model.fusion_operator is not None
    assert hasattr(model, "fusion_adapter")
    assert model.fusion_adapter is not None
    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 1, 8, 8)
    fused = model.fusion_adapter(rgb=rgb, spike=spike)
    assert fused.shape == (1, 2, 4, 8, 8)


def test_vrt_forward_triggers_early_fusion(monkeypatch):
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "hybrid",
                "operator": "concat",
                "out_chans": 4,
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    called = {"adapter": False}
    original_adapter = model.fusion_adapter

    def _adapter_wrapper(*args, **kwargs):
        called["adapter"] = True
        return original_adapter(*args, **kwargs)

    model.fusion_adapter = _adapter_wrapper

    dummy_flows = [
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 4, 4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.zeros(1, 1, 2, 1, 1),
    ]

    def _fake_get_flows(_x):
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, chans, height, width = _x.shape
        return [
            torch.zeros(bsz, steps, model.in_chans * 4, height, width),
            torch.zeros(bsz, steps, model.in_chans * 4, height, width),
        ]

    def _fake_forward_features(_x, _fb, _ff, fusion_hook=None, spike_ctx=None):
        return torch.zeros_like(_x)

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", _fake_forward_features)

    x = torch.randn(1, 2, 4, 8, 8)
    out = model(x)

    assert called["adapter"] is True
    assert out.shape == (1, 2, 3, 8, 8)


def test_vrt_builds_with_middle_fusion_adapter():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "middle",
                "operator": "concat",
                "out_chans": 16,
                "inject_stages": [1],
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    assert isinstance(model.fusion_adapter, MiddleFusionAdapter)


def test_vrt_builds_with_hybrid_fusion_adapter():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "hybrid",
                "operator": "concat",
                "out_chans": 4,
                "middle": {"out_chans": 16},
                "inject_stages": [1],
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    assert isinstance(model.fusion_adapter, HybridFusionAdapter)
    assert hasattr(model.fusion_adapter, "early_adapter")
    assert hasattr(model.fusion_adapter, "middle_adapter")
    assert model.fusion_adapter.early_adapter.operator is not model.fusion_adapter.middle_adapter.operator
    assert model.fusion_adapter.early_adapter.operator.rgb_chans == 3
    assert model.fusion_adapter.middle_adapter.operator.rgb_chans == 16
    assert model.fusion_adapter.middle_adapter.operator.spike_chans == 1


def test_vrt_forward_features_passes_fusion_hook_for_middle():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "middle",
                "operator": "concat",
                "out_chans": 16,
                "inject_stages": [],
                "operator_params": {},
            }
        }
    }

    model = VRT(
        upscale=1,
        in_chans=4,
        out_chans=3,
        img_size=[2, 8, 8],
        window_size=[2, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    )

    class DummyStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.called = None

        def forward(self, x, flows_backward, flows_forward, fusion_hook=None, stage_idx=None, spike_ctx=None):
            self.called = {
                "fusion_hook": fusion_hook,
                "stage_idx": stage_idx,
                "spike_ctx": spike_ctx,
            }
            return x

    dummy_stages = [DummyStage() for _ in range(7)]
    model.stage1, model.stage2, model.stage3, model.stage4, model.stage5, model.stage6, model.stage7 = dummy_stages
    model.stage8 = nn.ModuleList([nn.Identity()])

    x = torch.randn(1, 16, 2, 8, 8)
    spike_ctx = torch.randn_like(x)
    dummy_flows = [
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 4, 4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.zeros(1, 1, 2, 1, 1),
    ]

    model.forward_features(
        x,
        dummy_flows,
        dummy_flows,
        fusion_hook=model.fusion_adapter,
        spike_ctx=spike_ctx,
    )

    assert dummy_stages[0].called is not None
    assert dummy_stages[0].called["fusion_hook"] is model.fusion_adapter
    assert dummy_stages[0].called["stage_idx"] == 1
    assert dummy_stages[0].called["spike_ctx"] is spike_ctx


def test_vrt_middle_out_chans_mismatch_raises():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "middle",
                "operator": "concat",
                "out_chans": 8,
                "inject_stages": [1],
                "operator_params": {},
            }
        }
    }

    with pytest.raises(ValueError, match="out_chans"):
        VRT(
            upscale=1,
            in_chans=4,
            out_chans=3,
            img_size=[2, 8, 8],
            window_size=[2, 4, 4],
            depths=[1] * 8,
            indep_reconsts=[],
            embed_dims=[16] * 8,
            num_heads=[1] * 8,
            pa_frames=2,
            use_flash_attn=False,
            optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
            opt=opt,
        )


def test_vrt_middle_inject_stages_mixed_dims_raises():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "middle",
                "operator": "concat",
                "out_chans": 16,
                "inject_stages": [1, 2],
                "operator_params": {},
            }
        }
    }

    with pytest.raises(ValueError, match="multiple feature dims"):
        VRT(
            upscale=1,
            in_chans=4,
            out_chans=3,
            img_size=[2, 8, 8],
            window_size=[2, 4, 4],
            depths=[1] * 8,
            indep_reconsts=[],
            embed_dims=[16, 32, 16, 16, 16, 16, 16, 16],
            num_heads=[1] * 8,
            pa_frames=2,
            use_flash_attn=False,
            optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
            opt=opt,
        )


def test_full_t_rejects_non_spikecv_tfp():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 11,
                "early": {"expand_to_full_t": True},
                "operator_params": {},
            }
        },
        "datasets": {
            "train": {
                "spike_reconstruction": {
                    "type": "middle_tfp",
                }
            }
        },
    }

    with pytest.raises(ValueError, match="full-T early fusion requires spikecv_tfp"):
        VRT(
            upscale=1,
            in_chans=11,
            out_chans=3,
            img_size=[2, 8, 8],
            window_size=[2, 4, 4],
            depths=[1] * 8,
            indep_reconsts=[],
            embed_dims=[16] * 8,
            num_heads=[1] * 8,
            pa_frames=2,
            use_flash_attn=False,
            optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
            opt=opt,
        )


def test_full_t_hybrid_rejects_non_spikecv_tfp_from_test_dataset():
    opt = {
        "netG": {
            "fusion": {
                "enable": True,
                "placement": "hybrid",
                "operator": "concat",
                "out_chans": 4,
                "middle": {"out_chans": 16},
                "inject_stages": [1],
                "operator_params": {},
            }
        },
        "datasets": {
            "test": {
                "spike_reconstruction": {
                    "type": "middle_tfp",
                }
            }
        },
    }

    with pytest.raises(ValueError, match="full-T early fusion requires spikecv_tfp"):
        VRT(
            upscale=1,
            in_chans=4,
            out_chans=3,
            img_size=[2, 8, 8],
            window_size=[2, 4, 4],
            depths=[1] * 8,
            indep_reconsts=[],
            embed_dims=[16] * 8,
            num_heads=[1] * 8,
            pa_frames=2,
            use_flash_attn=False,
            optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
            opt=opt,
        )
