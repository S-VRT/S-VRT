"""Layer 2: SCFlow functional integration tests using synthetic data."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from data.spike_recc.encoding25 import (
    build_centered_window,
    compute_center_index,
    validate_encoding25_tensor,
)
from models.architectures.vrt.vrt import VRT
from models.model_plain import ModelPlain
from models.optical_flow.scflow.wrapper import SCFlowWrapper


# ---------------------------------------------------------------------------
# Shared stubs (defined here; do NOT import from contract test file)
# ---------------------------------------------------------------------------

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _DummyTimer:
    def timer(self, _name): return _NullCtx()


class _DummySpikeFlow:
    input_type = "spike"

    def __init__(self):
        self.calls = []

    def __call__(self, x1, x2):
        self.calls.append((x1, x2))
        bsz, _c, h, w = x1.shape
        return [torch.zeros(bsz, 2, h // (2**i), w // (2**i)) for i in range(4)]


class _MockSCFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_dt = None

    def forward(self, spk1, spk2, flow_init, dt=10):
        self.received_dt = dt
        b, _, h, w = spk1.shape
        flows = [torch.zeros(b, 2, h // (2**i), w // (2**i)) for i in range(4)]
        return flows, {}


class _RecordingNet:
    training = True
    _input_path_marker = None

    def set_input_path_marker(self, m):
        self._input_path_marker = m

    def __call__(self, x, **kwargs):
        self.last_call_kwargs = kwargs
        return torch.zeros_like(x[:, :, :3, :, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stub_plain_model_with_net(flow_module, net):
    model = ModelPlain.__new__(ModelPlain)
    model.opt = {
        "netG": {
            "in_chans": 7,
            "input_mode": "concat",
            "optical_flow": {"module": flow_module},
        }
    }
    model.device = "cpu"
    model.timer = _DummyTimer()
    model.netG = net
    model.L_flow_spike = None
    return model


# ---------------------------------------------------------------------------
# Group 1 — encoding25 data round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_centered_window_extracts_correct_slice():
    spike_matrix = np.arange(100 * 8 * 8).reshape(100, 8, 8).astype(np.float32)
    center = 50
    result = build_centered_window(spike_matrix, center=center)
    expected = spike_matrix[38:63]
    assert np.array_equal(result, expected)


@pytest.mark.integration
def test_compute_center_and_build_window_pipeline():
    spike_matrix = np.zeros((200, 8, 8), dtype=np.float32)
    center = compute_center_index(frame_index=3, clip_start_frame=0, dt=10, center_offset=40)
    assert center == 70
    result = build_centered_window(spike_matrix, center=center)
    expected = spike_matrix[58:83]
    assert np.array_equal(result, expected)


@pytest.mark.integration
def test_encoding25_npy_roundtrip(tmp_path):
    arr = np.random.rand(25, 8, 8).astype(np.float32)
    out_dir = tmp_path / "clip" / "encoding25_dt10"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "000001.npy", arr)
    loaded = np.load(out_dir / "000001.npy").astype(np.float32)
    validate_encoding25_tensor(loaded)  # must not raise
    assert np.allclose(arr, loaded)


# ---------------------------------------------------------------------------
# Group 2 — Dataset _load_encoded_flow_spike actual loading
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_encoded_flow_spike_returns_correct_shape_and_dtype(tmp_path):
    artifact_dir = tmp_path / "clip_a" / "encoding25_dt10"
    artifact_dir.mkdir(parents=True)
    arr = np.zeros((25, 8, 8), dtype=np.float32)
    np.save(artifact_dir / "000001.npy", arr)

    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (25, 8, 8)
    assert result.dtype == np.float32


@pytest.mark.integration
def test_load_encoded_flow_spike_explicit_root_overrides_spike_root(tmp_path):
    explicit_root = tmp_path / "explicit"
    artifact_dir = explicit_root / "clip_a" / "encoding25_dt10"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "000001.npy", np.zeros((25, 8, 8), dtype=np.float32))

    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path / "other"
    ds.spike_flow_root = str(explicit_root)
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (25, 8, 8)


# ---------------------------------------------------------------------------
# Group 3 — SCFlowWrapper forward output shapes
# ---------------------------------------------------------------------------

def _make_scflow_wrapper(mock_model):
    """Build a SCFlowWrapper bypassing __init__ but properly initializing nn.Module."""
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    nn.Module.__init__(wrapper)
    wrapper.input_type = "spike"
    wrapper.device = torch.device("cpu")
    wrapper.dt = 10
    wrapper.model = mock_model
    return wrapper


@pytest.mark.integration
def test_scflow_wrapper_forward_returns_4_scales():
    wrapper = _make_scflow_wrapper(_MockSCFlowModel())
    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    result = wrapper.forward(spk1, spk2)
    assert len(result) == 4


@pytest.mark.integration
def test_scflow_wrapper_forward_output_shapes():
    wrapper = _make_scflow_wrapper(_MockSCFlowModel())
    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    result = wrapper.forward(spk1, spk2)
    expected_shapes = [(1, 2, 16, 16), (1, 2, 8, 8), (1, 2, 4, 4), (1, 2, 2, 2)]
    for flow, expected in zip(result, expected_shapes):
        assert tuple(flow.shape) == expected


@pytest.mark.integration
def test_scflow_wrapper_forward_passes_dt_to_model():
    mock_model = _MockSCFlowModel()
    wrapper = _make_scflow_wrapper(mock_model)
    spk1 = torch.randn(1, 25, 16, 16)
    spk2 = torch.randn(1, 25, 16, 16)
    wrapper.forward(spk1, spk2)
    assert mock_model.received_dt == 10


# ---------------------------------------------------------------------------
# Group 4 — ModelPlain netG_forward routing
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_netg_forward_passes_flow_spike_for_scflow():
    net = _RecordingNet()
    model = _build_stub_plain_model_with_net("scflow", net)
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 25, 8, 8),
    }
    model.feed_data(data, need_H=False)
    model.netG_forward()
    assert "flow_spike" in net.last_call_kwargs
    assert tuple(net.last_call_kwargs["flow_spike"].shape) == (1, 4, 25, 8, 8)


@pytest.mark.integration
def test_netg_forward_omits_flow_spike_for_non_scflow():
    net = _RecordingNet()
    model = _build_stub_plain_model_with_net("spynet", net)
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
    }
    model.feed_data(data, need_H=False)
    model.netG_forward()
    assert "flow_spike" not in net.last_call_kwargs


# ---------------------------------------------------------------------------
# Group 5 — VRT get_flow_2frames complete output
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_vrt_get_flow_2frames_backward_forward_count():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, flows_forward = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    assert len(flows_backward) == 4
    assert len(flows_forward) == 4


@pytest.mark.integration
def test_vrt_get_flow_2frames_flow_shape():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, _ = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    # b=1, n-1=3, 2 channels, h=16, w=16 at scale i=0 → shape (1, 3, 2, 16, 16)
    # flows_backward has 4 elements (one per scale)
    assert len(flows_backward) == 4
    assert tuple(flows_backward[0].shape) == (1, 3, 2, 16, 16)


@pytest.mark.integration
def test_vrt_get_flow_2frames_uses_flow_spike_not_x():
    vrt = VRT.__new__(VRT)
    spy = _DummySpikeFlow()
    vrt.spynet = spy
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    vrt.get_flow_2frames(x, flow_spike=flow_spike)
    # spynet received 25-channel spike frames, not 7-channel x frames
    assert spy.calls[0][0].shape[1] == 25
