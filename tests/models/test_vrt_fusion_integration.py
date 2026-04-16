import pytest
import torch
import torch.nn as nn

from models.architectures.vrt.vrt import VRT
from models.fusion.adapters.hybrid import HybridFusionAdapter
from models.fusion.adapters.middle import MiddleFusionAdapter
from models.fusion.reducers import build_restoration_reducer


def test_vrt_builds_with_fusion_config():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
    original_forward = model.fusion_adapter.forward

    def _adapter_wrapper(*args, **kwargs):
        called["adapter"] = True
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.fusion_adapter, "forward", _adapter_wrapper)

    dummy_flows = [
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 4, 4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.zeros(1, 1, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, chans, height, width = _x.shape
        return [
            torch.zeros(bsz, steps, model.backbone_in_chans * 4, height, width),
            torch.zeros(bsz, steps, model.backbone_in_chans * 4, height, width),
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


def test_vrt_get_aligned_image_uses_backbone_in_chans_for_early_fusion():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 11,
            },
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            }
        }
    }

    model = VRT(
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

    x = torch.randn(1, 2, 3, 8, 8)
    flows = torch.zeros(1, 1, 2, 8, 8)
    x_backward, x_forward = model.get_aligned_image_2frames(x, flows, flows)

    assert model.backbone_in_chans == 3
    assert x_backward.shape == (1, 2, 12, 8, 8)
    assert x_forward.shape == (1, 2, 12, 8, 8)


def test_vrt_builds_with_middle_fusion_adapter():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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


def test_vrt_builds_with_early_fusion_out_chans_3():
    """Early fusion with out_chans=3 and in_chans=11 should build successfully."""
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 11,
            },
            "output_mode": "restoration",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }

    model = VRT(
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

    assert model.fusion_enabled is True
    assert model.output_mode == "restoration"
    assert model.spike_bins == 8
    assert model.conv_first.weight.shape[1] == 27


def test_index_reducer_selects_configured_index():
    reducer = build_restoration_reducer({"type": "index", "index": 1})
    x = torch.arange(6.0).reshape(1, 6, 1, 1, 1).expand(1, 6, 3, 2, 2)
    out = reducer(x=x, spike_bins=3, base_rgb=None)
    assert out.shape == (1, 2, 3, 2, 2)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[0, 1, 0, 0, 0].item() == 4.0


def test_index_reducer_uses_configured_position():
    reducer = build_restoration_reducer({"type": "index", "index": 2})
    x = torch.arange(12.0).reshape(1, 12, 1, 1, 1).expand(1, 12, 3, 8, 8)
    selected = reducer(x=x, spike_bins=4, base_rgb=None)
    assert selected.shape == (1, 3, 3, 8, 8)
    assert selected[0, 0, 0, 0, 0].item() == 2.0
    assert selected[0, 1, 0, 0, 0].item() == 6.0
    assert selected[0, 2, 0, 0, 0].item() == 10.0


def test_selector_reducer_restores_n_frames():
    reducer = build_restoration_reducer({"type": "selector", "selector_hidden": 8})
    x = torch.randn(2, 12, 3, 8, 8)
    out = reducer(x=x, spike_bins=4, base_rgb=None)
    assert out.shape == (2, 3, 3, 8, 8)


def test_residual_selector_reducer_uses_base_rgb_shape():
    reducer = build_restoration_reducer({"type": "residual_selector", "selector_hidden": 8})
    x = torch.randn(2, 12, 3, 8, 8)
    base = torch.randn(2, 3, 3, 8, 8)
    out = reducer(x=x, spike_bins=4, base_rgb=base)
    assert out.shape == base.shape


def test_vrt_builds_with_selector_reducer():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 11,
            },
            "output_mode": "restoration",
            "restoration_reducer": {"type": "selector", "selector_hidden": 8},
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }

    model = VRT(
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

    assert model.restoration_reducer is not None


def test_vrt_builds_with_interpolation_mode():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 11,
            },
            "output_mode": "interpolation",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
        }
    }

    model = VRT(
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

    assert model.output_mode == "interpolation"
    assert model.restoration_reducer is None


def test_vrt_forward_features_passes_fusion_hook_for_middle():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 11,
            },
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
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 4,
            },
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
