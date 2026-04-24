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
    assert fused["fused_main"].shape == (1, 2, 4, 8, 8)
    assert fused["backbone_view"].shape == (1, 2, 4, 8, 8)
    assert fused["meta"]["frame_contract"] == "expanded"


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


def test_vrt_stores_explicit_fusion_main_exec_aux_and_meta_after_forward():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
            "output_mode": "restoration",
            "restoration_reducer": {"type": "index", "index": 2},
        }
    }
    model = VRT(
        upscale=1,
        in_chans=7,
        img_size=[6, 16, 16],
        window_size=[6, 8, 8],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    ).eval()

    # [B, N, 3+4, H, W] — 3 rgb + 4 spike bins
    x = torch.randn(1, 6, 7, 16, 16)
    with torch.no_grad():
        _ = model(x)

    assert hasattr(model, '_last_fusion_main'), "_last_fusion_main not set after forward"
    assert hasattr(model, '_last_fusion_exec'), "_last_fusion_exec not set after forward"
    assert hasattr(model, '_last_fusion_aux'), "_last_fusion_aux not set after forward"
    assert hasattr(model, '_last_fusion_meta'), "_last_fusion_meta not set after forward"
    assert hasattr(model, '_last_spike_bins'), "_last_spike_bins not set after forward"
    assert model._last_spike_bins == 4
    assert model._last_fusion_main.shape == (1, 6, 3, 16, 16)
    assert model._last_fusion_exec.shape == (1, 24, 3, 16, 16)
    assert model._last_fusion_aux.shape == (1, 24, 3, 16, 16)
    assert model._last_fusion_meta["frame_contract"] == "expanded"


def test_vrt_expanded_operator_keeps_main_n_and_exec_ns():
    opt = {
        "netG": {
            "input": {
                "strategy": "fusion",
                "mode": "dual",
                "raw_ingress_chans": 7,
            },
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "gated",
                "out_chans": 3,
                "operator_params": {},
            },
            "output_mode": "restoration",
            "restoration_reducer": {"type": "index", "index": 2},
        }
    }
    model = VRT(
        upscale=1,
        in_chans=7,
        img_size=[6, 16, 16],
        window_size=[6, 8, 8],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt=opt,
    ).eval()

    x = torch.randn(1, 6, 7, 16, 16)
    with torch.no_grad():
        _ = model(x)

    assert model._last_fusion_main.shape == (1, 6, 3, 16, 16)
    assert model._last_fusion_exec.shape == (1, 24, 3, 16, 16)
    assert model._last_fusion_meta["frame_contract"] == "expanded"


def test_vrt_builds_with_structured_early_mamba_config():
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
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 11},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 48, "d_state": 32, "d_conv": 4, "expand": 2, "num_layers": 3},
                    "early": {"expand_to_full_t": False},
                },
            }
        },
    )
    assert model.fusion_operator is not None
    assert getattr(model.fusion_operator, "expects_structured_early", False) is True
    assert getattr(model.fusion_operator, "frame_contract", None) == "collapsed"


def test_vrt_collapsed_operator_keeps_main_and_exec_equal(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[6, 8, 8],
        window_size=[6, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 8, "num_layers": 1},
                    "early": {"expand_to_full_t": False},
                },
                "output_mode": "restoration",
            }
        },
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)

    dummy_flows = [
        torch.zeros(1, 5, 2, 8, 8),
        torch.zeros(1, 5, 2, 4, 4),
        torch.zeros(1, 5, 2, 2, 2),
        torch.zeros(1, 5, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, _, height, width = _x.shape
        chans = model.backbone_in_chans * 4
        return [
            torch.zeros(bsz, steps, chans, height, width),
            torch.zeros(bsz, steps, chans, height, width),
        ]

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 7, 8, 8)
    with torch.no_grad():
        _ = model(x)

    assert model._last_fusion_main.shape == (1, 6, 3, 8, 8)
    assert model._last_fusion_exec.shape == (1, 6, 3, 8, 8)
    assert model._last_fusion_meta["frame_contract"] == "collapsed"


def test_vrt_structured_early_mamba_collapses_subframe_flow_spike(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[6, 8, 8],
        window_size=[6, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "mamba",
                    "out_chans": 3,
                    "operator_params": {"model_dim": 8, "num_layers": 1},
                    "early": {"expand_to_full_t": False},
                },
                "output_mode": "restoration",
            }
        },
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)

    captured = {}
    dummy_flows = [
        torch.zeros(1, 5, 2, 8, 8),
        torch.zeros(1, 5, 2, 4, 4),
        torch.zeros(1, 5, 2, 2, 2),
        torch.zeros(1, 5, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        captured["x_shape"] = tuple(_x.shape)
        captured["flow_spike"] = flow_spike.detach().clone()
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, _, height, width = _x.shape
        chans = model.backbone_in_chans * 4
        return [
            torch.zeros(bsz, steps, chans, height, width),
            torch.zeros(bsz, steps, chans, height, width),
        ]

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 7, 8, 8)
    flow_spike = torch.arange(1 * 24 * 25 * 8 * 8, dtype=torch.float32).reshape(1, 24, 25, 8, 8)

    with torch.no_grad():
        out = model(x, flow_spike=flow_spike)

    expected_flow = flow_spike.reshape(1, 6, 4, 25, 8, 8).mean(dim=2)
    assert captured["x_shape"] == (1, 6, 3, 8, 8)
    assert tuple(captured["flow_spike"].shape) == (1, 6, 25, 8, 8)
    assert torch.equal(captured["flow_spike"], expected_flow)
    assert model._last_fusion_main.shape == (1, 6, 3, 8, 8)
    assert model._last_fusion_exec.shape == (1, 6, 3, 8, 8)
    assert model._last_fusion_meta["frame_contract"] == "collapsed"
    assert out.shape == (1, 6, 3, 8, 8)


def test_vrt_flow_alignment_uses_execution_steps_not_main_steps(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=7,
        out_chans=3,
        img_size=[6, 8, 8],
        window_size=[6, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "scflow", "checkpoint": None, "params": {}},
        opt={
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 7},
                "fusion": {
                    "placement": "early",
                    "operator": "gated",
                    "out_chans": 3,
                    "operator_params": {},
                },
                "output_mode": "restoration",
                "restoration_reducer": {"type": "index", "index": 2},
            }
        },
    )

    captured = {}
    dummy_flows = [
        torch.zeros(1, 23, 2, 8, 8),
        torch.zeros(1, 23, 2, 4, 4),
        torch.zeros(1, 23, 2, 2, 2),
        torch.zeros(1, 23, 2, 1, 1),
    ]

    def _fake_get_flows(_x, flow_spike=None):
        captured["x_shape"] = tuple(_x.shape)
        captured["flow_shape"] = None if flow_spike is None else tuple(flow_spike.shape)
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, _, height, width = _x.shape
        chans = model.backbone_in_chans * 4
        return [
            torch.zeros(bsz, steps, chans, height, width),
            torch.zeros(bsz, steps, chans, height, width),
        ]

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 7, 8, 8)
    flow_spike = torch.randn(1, 24, 25, 8, 8)

    with torch.no_grad():
        _ = model(x, flow_spike=flow_spike)

    assert captured["x_shape"][1] == 24
    assert captured["flow_shape"][1] == 24


def test_vrt_rejects_mamba_with_full_t_early_expansion():
    with pytest.raises(ValueError, match="mamba.*expand_to_full_t"):
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
            opt={
                "netG": {
                    "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 11},
                    "fusion": {
                        "placement": "early",
                        "operator": "mamba",
                        "out_chans": 3,
                        "early": {"expand_to_full_t": True},
                    },
                }
            },
        )


def test_model_vrt_optimize_parameters_switches_mamba_warmup_stage_and_phase2_unfreezes(monkeypatch):
    from collections import OrderedDict
    from contextlib import nullcontext
    from models.model_vrt import ModelVRT

    class _WarmupAwareOperator:
        def __init__(self):
            self.last_stage = None

        def set_warmup_stage(self, stage):
            self.last_stage = stage

    class _BareNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion_enabled = True
            self.fusion_cfg = {"placement": "early"}
            self.fusion_adapter = type("Adapter", (), {"operator": _WarmupAwareOperator()})()
            self.spynet_conv = nn.Linear(4, 4)
            self.pa_deform_conv = nn.Linear(4, 4)
            self.backbone_base = nn.Linear(4, 4)
            self.lora_A = nn.Parameter(torch.zeros(1))
            self.lora_B = nn.Parameter(torch.zeros(1))

    model = ModelVRT.__new__(ModelVRT)
    model.opt = {"train": {"checkpoint_save": 100}, "dist": False}
    model.opt_train = {
        "fusion_warmup": {"head_only_iters": 2},
        "G_optimizer_clipgrad": None,
        "G_regularizer_orthstep": None,
        "G_regularizer_clipstep": None,
        "E_decay": 0,
        "phase2_lora_mode": True,
        "use_lora": True,
    }
    model.fix_iter = 10
    model.fix_keys = ["spynet", "pa_deform"]
    model.fix_unflagged = False
    model.timer = type("TimerStub", (), {"current_timings": {}, "timer": staticmethod(lambda *_args, **_kwargs: nullcontext())})()
    model.log_dict = OrderedDict()
    model.grad_scaler = type(
        "ScalerStub",
        (),
        {
            "is_enabled": staticmethod(lambda: False),
            "step": staticmethod(lambda _optimizer: None),
            "update": staticmethod(lambda: None),
            "scale": staticmethod(lambda value: value),
        },
    )()
    model.G_optimizer = type("OptimizerStub", (), {"zero_grad": staticmethod(lambda: None), "step": staticmethod(lambda: None)})()
    model.fusion_debug = type(
        "DebugStub",
        (),
        {
            "should_dump_phase1_last": staticmethod(lambda *args, **kwargs: False),
            "arm": staticmethod(lambda: None),
            "disarm": staticmethod(lambda: None),
        },
    )()

    bare = _BareNet()
    bare.spynet_conv.weight.requires_grad_(False)
    bare.pa_deform_conv.weight.requires_grad_(False)
    bare.lora_A.requires_grad_(False)
    bare.lora_B.requires_grad_(False)
    bare.backbone_base.weight.requires_grad_(False)
    model.netG = bare
    model.get_bare_model = lambda net: net
    model._phase1_fusion_forward = lambda: None
    model.netG_forward = lambda: setattr(model, "E", torch.zeros(1, 1, 3, 4, 4, requires_grad=True))
    model._compute_fusion_aux_loss = lambda is_phase1, current_step=None: torch.tensor(0.0, requires_grad=True)
    model.G_lossfn_weight = 0.0
    model.G_lossfn = lambda pred, target: pred.sum() * 0.0
    model.H = torch.zeros(1, 1, 3, 4, 4)

    model.optimize_parameters(current_step=0)
    assert bare.fusion_adapter.operator.last_stage == "writeback_only"

    model.optimize_parameters(current_step=3)
    assert bare.fusion_adapter.operator.last_stage == "token_mixer"

    model.optimize_parameters(current_step=10)
    assert bare.fusion_adapter.operator.last_stage == "full"
    assert bare.spynet_conv.weight.requires_grad is True
    assert bare.pa_deform_conv.weight.requires_grad is True
    assert bare.lora_A.requires_grad is True
    assert bare.lora_B.requires_grad is True
    assert bare.backbone_base.weight.requires_grad is False


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
