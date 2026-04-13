from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from data.spike_recc.encoding25 import build_output_dir, validate_encoding25_tensor
from models.architectures.vrt.vrt import VRT
from models.model_plain import ModelPlain
from models.optical_flow.scflow.wrapper import SCFlowWrapper


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyTimer:
    def timer(self, _name):
        return _NullCtx()




class _DummyNet:
    training = True

class _DummySpikeFlow:
    input_type = "spike"

    def __call__(self, x1, x2):
        bsz, _c, h, w = x1.shape
        return [torch.zeros(bsz, 2, h // (2 ** i), w // (2 ** i)) for i in range(4)]


def _build_stub_plain_model(flow_module="scflow"):
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {
            "in_chans": 7,
            "optical_flow": {"module": flow_module},
        }
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    model.netG = _DummyNet()
    return model


def test_validate_encoding25_tensor_rejects_non_25_channels():
    bad = np.zeros((11, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 25"):
        validate_encoding25_tensor(bad)


def test_build_output_dir_uses_dataset_local_convention(tmp_path):
    clip_dir = tmp_path / "GOPR0001"
    out = build_output_dir(clip_dir, dt=10)
    assert out.name == "encoding25_dt10"


def test_scflow_wrapper_rejects_non_25_channels():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="channels=25"):
        wrapper._validate_spike_pair(torch.randn(1, 11, 8, 8), torch.randn(1, 25, 8, 8))


def test_model_plain_requires_l_flow_spike_for_scflow():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 6, 7, 16, 16),
        "H": torch.randn(1, 6, 3, 16, 16),
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data)


def test_vrt_scflow_branch_requires_flow_spike():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    with pytest.raises(ValueError, match="flow_spike"):
        vrt.get_flow_2frames(torch.randn(1, 3, 7, 8, 8), flow_spike=None)


def test_vrt_scflow_branch_accepts_25ch_flow_spike():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, flows_forward = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    assert len(flows_backward) == 4
    assert len(flows_forward) == 4


def test_dataset_rejects_non_encoding25_representation_for_scflow():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    with pytest.raises(ValueError, match="representation='encoding25'"):
        ds._parse_spike_flow_config(
            {"spike_flow": {"representation": "tfp", "dt": 10, "root": "auto"}},
            optical_flow_module="scflow",
        )


def test_dataset_missing_encoding25_artifact_reports_path(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    with pytest.raises(ValueError, match="Missing encoding25 artifact"):
        ds._load_encoded_flow_spike("clip_a", 1)
