import pytest
import torch

from models.architectures.vrt.vrt import VRT


def _build_vrt(opt, in_chans=4):
    return VRT(
        upscale=1,
        in_chans=in_chans,
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


def _patch_lightweight_forward(model, monkeypatch):
    dummy_flows = [
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 4, 4),
        torch.zeros(1, 1, 2, 2, 2),
        torch.zeros(1, 1, 2, 1, 1),
    ]

    def _fake_get_flows(_x):
        return dummy_flows, dummy_flows

    def _fake_aligned(_x, _fb, _ff):
        bsz, steps, _chans, height, width = _x.shape
        return [
            torch.zeros(bsz, steps, model.in_chans * 4, height, width),
            torch.zeros(bsz, steps, model.in_chans * 4, height, width),
        ]

    def _fake_forward_features(_x, _fb, _ff, fusion_hook=None, spike_ctx=None):
        return torch.zeros_like(_x)

    monkeypatch.setattr(model, "get_flows", _fake_get_flows)
    monkeypatch.setattr(model, "get_aligned_image_2frames", _fake_aligned)
    monkeypatch.setattr(model, "forward_features", _fake_forward_features)


def test_vrt_forward_logs_concat_path_by_default(caplog, monkeypatch):
    model = _build_vrt(opt={"netG": {}}, in_chans=4)
    _patch_lightweight_forward(model, monkeypatch)
    caplog.set_level("INFO")
    x = torch.randn(1, 2, 4, 8, 8)
    _ = model(x)
    assert "input_path=concat_path" in caplog.text


def test_vrt_forward_logs_dual_path_when_input_mode_dual(caplog, monkeypatch):
    model = _build_vrt(opt={"netG": {"input_mode": "dual"}}, in_chans=4)
    _patch_lightweight_forward(model, monkeypatch)
    caplog.set_level("INFO")
    x = torch.randn(1, 2, 4, 8, 8)
    _ = model(x)
    assert "input_path=dual_path" in caplog.text


def test_vrt_forward_logs_dual_fallback_marker(caplog, monkeypatch):
    model = _build_vrt(opt={"netG": {"input_mode": "dual"}}, in_chans=4)
    _patch_lightweight_forward(model, monkeypatch)
    caplog.set_level("INFO")
    model.set_input_path_marker("dual_fallback_to_concat_path")
    x = torch.randn(1, 2, 4, 8, 8)
    _ = model(x)
    assert "input_path=dual_fallback_to_concat_path" in caplog.text


def test_vrt_dual_with_fusion_requires_spike_channels():
    opt = {
        "netG": {
            "input_mode": "dual",
            "fusion": {
                "enable": True,
                "placement": "middle",
                "operator": "concat",
                "out_chans": 16,
                "inject_stages": [1],
                "operator_params": {},
            },
        }
    }
    with pytest.raises(ValueError, match="in_chans=3"):
        _build_vrt(opt=opt, in_chans=3)


def test_vrt_dual_early_out_chans_mismatch_includes_dims():
    opt = {
        "netG": {
            "input_mode": "dual",
            "fusion": {
                "enable": True,
                "placement": "early",
                "operator": "concat",
                "out_chans": 5,
                "operator_params": {},
            },
        }
    }
    with pytest.raises(ValueError, match=r"input_mode=dual.*out_chans \(5\).*in_chans \(4\)"):
        _build_vrt(opt=opt, in_chans=4)


def test_vrt_dual_hybrid_out_chans_mismatch_includes_dims():
    opt = {
        "netG": {
            "input_mode": "dual",
            "fusion": {
                "enable": True,
                "placement": "hybrid",
                "operator": "concat",
                "out_chans": 5,
                "middle": {"out_chans": 16},
                "inject_stages": [1],
                "operator_params": {},
            },
        }
    }
    with pytest.raises(ValueError, match=r"input_mode=dual.*out_chans \(5\).*in_chans \(4\)"):
        _build_vrt(opt=opt, in_chans=4)


@pytest.mark.parametrize(
    "opt,in_chans",
    [
        ({"netG": {"input_mode": "concat", "fusion": {"enable": False}}}, 4),
        (
            {
                "netG": {
                    "input_mode": "dual",
                    "fusion": {
                        "enable": True,
                        "placement": "early",
                        "operator": "concat",
                        "out_chans": 4,
                        "early": {"expand_to_full_t": False},
                        "operator_params": {},
                    },
                }
            },
            4,
        ),
        (
            {
                "netG": {
                    "input_mode": "dual",
                    "fusion": {
                        "enable": True,
                        "placement": "middle",
                        "operator": "concat",
                        "out_chans": 16,
                        "inject_stages": [1],
                        "operator_params": {},
                    },
                }
            },
            4,
        ),
        (
            {
                "netG": {
                    "input_mode": "dual",
                    "fusion": {
                        "enable": True,
                        "placement": "hybrid",
                        "operator": "concat",
                        "out_chans": 4,
                        "middle": {"out_chans": 16},
                        "inject_stages": [1],
                        "operator_params": {},
                    },
                }
            },
            4,
        ),
        (
            {
                "netG": {
                    "input_mode": "dual",
                    "fusion": {
                        "enable": True,
                        "placement": "early",
                        "operator": "concat",
                        "out_chans": 11,
                        "early": {"expand_to_full_t": True},
                        "operator_params": {},
                    }
                },
                "datasets": {"train": {"spike_reconstruction": {"type": "spikecv_tfp"}}},
            },
            11,
        ),
    ],
)
def test_vrt_acceptance_matrix_constructs(opt, in_chans):
    model = _build_vrt(opt=opt, in_chans=in_chans)
    assert model is not None
