import pytest
import torch

from models.model_plain import ModelPlain
from models.model_vrt import ModelVRT


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyTimer:
    def timer(self, _name):
        return _NullCtx()


class _MarkerNet:
    def __init__(self):
        self.last_marker = None
        self.training = True

    def set_input_path_marker(self, marker):
        self.last_marker = marker


def _build_stub_model(input_mode, in_chans=7):
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"netG": {"input_mode": input_mode, "in_chans": in_chans}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    return model


def test_build_model_input_concat_uses_legacy_l():
    model = _build_stub_model("concat", in_chans=7)
    l = torch.randn(1, 2, 7, 8, 8)
    out = model._build_model_input_tensor({"L": l})
    assert out is l


def test_build_model_input_uses_canonical_input_mode_dual():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {"netG": {"input": {"mode": "dual"}, "in_chans": 7}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    out = model._build_model_input_tensor(
        {"L_rgb": torch.randn(1, 2, 3, 8, 8), "L_spike": torch.randn(1, 2, 4, 8, 8)}
    )
    assert out.shape == (1, 2, 7, 8, 8)


def test_concat_strategy_accepts_concat_mode():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {"input": {"strategy": "concat", "mode": "concat", "raw_ingress_chans": 7}}
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    out = model._build_model_input_tensor({"L": torch.randn(1, 2, 7, 8, 8)})
    assert out.shape == (1, 2, 7, 8, 8)


def test_fusion_strategy_rejects_concat_mode():
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {"input": {"strategy": "fusion", "mode": "concat", "raw_ingress_chans": 11}}
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    with pytest.raises(ValueError, match="strategy=fusion"):
        model._resolve_input_mode()


def test_build_model_input_concat_missing_l_raises():
    model = _build_stub_model("concat", in_chans=7)
    with pytest.raises(KeyError, match="input_mode=concat"):
        model._build_model_input_tensor({})


def test_build_model_input_dual_prefers_l_rgb_l_spike():
    model = _build_stub_model("dual", in_chans=7)
    l_rgb = torch.randn(1, 2, 3, 8, 8)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    legacy_l = torch.randn(1, 2, 7, 8, 8)
    out = model._build_model_input_tensor({"L_rgb": l_rgb, "L_spike": l_spike, "L": legacy_l})
    assert torch.allclose(out, torch.cat([l_rgb, l_spike], dim=2))


def test_build_model_input_dual_fallbacks_to_legacy_l():
    model = _build_stub_model("dual", in_chans=7)
    l = torch.randn(1, 2, 7, 8, 8)
    out = model._build_model_input_tensor({"L": l})
    assert out is l


def test_build_model_input_dual_missing_keys_raises():
    model = _build_stub_model("dual", in_chans=7)
    with pytest.raises(KeyError, match="input_mode=dual"):
        model._build_model_input_tensor({})


def test_build_model_input_dual_partial_payload_raises():
    model = _build_stub_model("dual", in_chans=7)
    with pytest.raises(KeyError, match="partial dual payload"):
        model._build_model_input_tensor({"L_rgb": torch.randn(1, 2, 3, 8, 8)})


def test_build_model_input_dual_shape_mismatch_raises():
    model = _build_stub_model("dual", in_chans=7)
    l_rgb = torch.randn(1, 2, 3, 8, 8)
    l_spike = torch.randn(1, 3, 4, 8, 8)
    with pytest.raises(ValueError, match="requires matching"):
        model._build_model_input_tensor({"L_rgb": l_rgb, "L_spike": l_spike})


def test_build_model_input_dual_resizes_spike_before_concat():
    model = _build_stub_model("dual", in_chans=7)
    l_rgb = torch.randn(1, 2, 3, 16, 16)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    out = model._build_model_input_tensor({"L_rgb": l_rgb, "L_spike": l_spike})
    assert out.shape == (1, 2, 7, 16, 16)


def test_validate_dual_input_allows_spatial_mismatch():
    model = _build_stub_model("dual", in_chans=7)
    l_rgb = torch.randn(1, 2, 3, 16, 16)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    model._validate_dual_input_tensors(l_rgb, l_spike)


def test_build_model_input_dual_invalid_rgb_channels_raises():
    model = _build_stub_model("dual", in_chans=7)
    l_rgb = torch.randn(1, 2, 2, 8, 8)
    l_spike = torch.randn(1, 2, 5, 8, 8)
    with pytest.raises(ValueError, match="expects L_rgb channels=3"):
        model._build_model_input_tensor({"L_rgb": l_rgb, "L_spike": l_spike})


def test_build_model_input_invalid_mode_raises():
    model = _build_stub_model("bad_mode", in_chans=7)
    with pytest.raises(ValueError, match="input_mode"):
        model._build_model_input_tensor({"L": torch.randn(1, 2, 7, 8, 8)})


def test_feed_data_enforces_channel_assert_after_dual_concat():
    model = _build_stub_model("dual", in_chans=8)
    model.netG = _MarkerNet()
    l_rgb = torch.randn(1, 2, 3, 8, 8)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    with pytest.raises(ValueError, match="Channel Mismatch"):
        model.feed_data({"L_rgb": l_rgb, "L_spike": l_spike, "H": torch.randn(1, 2, 3, 8, 8)})


def test_model_vrt_feed_data_uses_dual_builder_contract():
    model = ModelVRT.__new__(ModelVRT)
    model.opt = {"netG": {"input_mode": "dual", "in_chans": 7}}
    model.device = "cpu"
    model.timer = _DummyTimer()
    l_rgb = torch.randn(1, 2, 3, 8, 8)
    l_spike = torch.randn(1, 2, 4, 8, 8)
    model.feed_data({"L_rgb": l_rgb, "L_spike": l_spike, "H": torch.randn(1, 2, 3, 8, 8)})
    assert model.L.shape == (1, 2, 7, 8, 8)


def test_model_plain_marks_dual_fallback_path_on_net():
    model = _build_stub_model("dual", in_chans=7)
    model.netG = _MarkerNet()
    model._build_model_input_tensor({"L": torch.randn(1, 2, 7, 8, 8)})
    assert model.netG.last_marker == "dual_fallback_to_concat_path"


def test_model_plain_marks_dual_path_when_dual_payload_present():
    model = _build_stub_model("dual", in_chans=7)
    model.netG = _MarkerNet()
    model._build_model_input_tensor(
        {"L_rgb": torch.randn(1, 2, 3, 8, 8), "L_spike": torch.randn(1, 2, 4, 8, 8)}
    )
    assert model.netG.last_marker == "dual_path"


def test_feed_data_dual_fallback_still_enforces_channel_assert():
    model = _build_stub_model("dual", in_chans=8)
    model.netG = _MarkerNet()
    with pytest.raises(ValueError, match="Channel Mismatch"):
        model.feed_data({"L": torch.randn(1, 2, 7, 8, 8), "H": torch.randn(1, 2, 3, 8, 8)})

