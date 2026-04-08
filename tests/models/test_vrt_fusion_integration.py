import torch

from models.architectures.vrt.vrt import VRT


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

    def _fake_forward_features(_x, _fb, _ff):
        return torch.zeros_like(_x)

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", _fake_forward_features)

    x = torch.randn(1, 2, 4, 8, 8)
    out = model(x)

    assert called["adapter"] is True
    assert out.shape == (1, 2, 3, 8, 8)
