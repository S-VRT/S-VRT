from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from data.spike_recc.encoding25 import (
    build_output_dir,
    compute_center_index,
    validate_center_bounds,
    build_centered_window,
    validate_encoding25_tensor,
    compute_subframe_centers,
    validate_subframes_tensor,
    build_output_dir_subframes,
)
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


@pytest.mark.unit
def test_validate_encoding25_tensor_rejects_non_25_channels():
    bad = np.zeros((11, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 25"):
        validate_encoding25_tensor(bad)


@pytest.mark.unit
def test_build_output_dir_uses_dataset_local_convention(tmp_path):
    clip_dir = tmp_path / "GOPR0001"
    out = build_output_dir(clip_dir, dt=10)
    assert out.name == "encoding25_dt10"


@pytest.mark.unit
def test_scflow_wrapper_rejects_non_25_channels():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="channels=25"):
        wrapper._validate_spike_pair(torch.randn(1, 11, 8, 8), torch.randn(1, 25, 8, 8))


@pytest.mark.unit
def test_model_plain_requires_l_flow_spike_for_scflow():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 6, 7, 16, 16),
        "H": torch.randn(1, 6, 3, 16, 16),
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data)


@pytest.mark.unit
def test_vrt_scflow_branch_requires_flow_spike():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    with pytest.raises(ValueError, match="flow_spike"):
        vrt.get_flow_2frames(torch.randn(1, 3, 7, 8, 8), flow_spike=None)


@pytest.mark.unit
def test_vrt_scflow_branch_accepts_25ch_flow_spike():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    flow_spike = torch.randn(1, 4, 25, 16, 16)
    flows_backward, flows_forward = vrt.get_flow_2frames(x, flow_spike=flow_spike)
    assert len(flows_backward) == 4
    assert len(flows_forward) == 4


@pytest.mark.unit
def test_dataset_rejects_non_encoding25_representation_for_scflow():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    with pytest.raises(ValueError, match="representation='encoding25'"):
        ds._parse_spike_flow_config(
            {"spike_flow": {"representation": "tfp", "dt": 10, "root": "auto"}},
            optical_flow_module="scflow",
        )


@pytest.mark.unit
def test_dataset_missing_encoding25_artifact_reports_path(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    with pytest.raises(ValueError, match="Missing encoding25 artifact"):
        ds._load_encoded_flow_spike("clip_a", 1)


# ---------------------------------------------------------------------------
# Group A — encoding25 utility contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_validate_encoding25_tensor_rejects_2d_input():
    with pytest.raises(ValueError, match="ndim"):
        validate_encoding25_tensor(np.zeros((16, 16)))


@pytest.mark.unit
def test_validate_encoding25_tensor_rejects_4d_input():
    with pytest.raises(ValueError, match="ndim"):
        validate_encoding25_tensor(np.zeros((25, 1, 16, 16)))


@pytest.mark.unit
def test_validate_encoding25_tensor_accepts_valid():
    validate_encoding25_tensor(np.zeros((25, 16, 16), dtype=np.float32))  # no error


@pytest.mark.unit
def test_build_output_dir_rejects_zero_dt(tmp_path):
    with pytest.raises(ValueError, match="dt"):
        build_output_dir(tmp_path / "clip", dt=0)


@pytest.mark.unit
def test_build_output_dir_rejects_negative_dt(tmp_path):
    with pytest.raises(ValueError, match="dt"):
        build_output_dir(tmp_path / "clip", dt=-1)


@pytest.mark.unit
def test_compute_center_index_formula():
    # center_offset + (frame_index - clip_start_frame) * dt = 40 + 3*10 = 70
    result = compute_center_index(frame_index=5, clip_start_frame=2, dt=10, center_offset=40)
    assert result == 70


@pytest.mark.unit
def test_validate_center_bounds_rejects_left_boundary():
    # center=50, edge_margin=40: center - 12 = 38 < 40 → ValueError
    with pytest.raises(ValueError, match="edge_margin"):
        validate_center_bounds(center=50, total_length=500, edge_margin=40)


@pytest.mark.unit
def test_validate_center_bounds_rejects_right_boundary():
    # center=448, total_length=500, edge_margin=40: center+12=460 >= 500-40=460 → ValueError
    with pytest.raises(ValueError, match="edge_margin"):
        validate_center_bounds(center=448, total_length=500, edge_margin=40)


@pytest.mark.unit
def test_validate_center_bounds_accepts_valid_center():
    # center=250, total_length=500, edge_margin=40: 250-12=238>=40, 250+12=262<460 → ok
    validate_center_bounds(center=250, total_length=500, edge_margin=40)  # no error


@pytest.mark.unit
def test_build_centered_window_rejects_2d_spike_matrix():
    with pytest.raises(ValueError, match="spike_matrix"):
        build_centered_window(np.zeros((100, 16)), center=50)


@pytest.mark.unit
def test_build_centered_window_rejects_length_not_25():
    with pytest.raises(ValueError, match="length=25"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=50, length=24)


@pytest.mark.unit
def test_build_centered_window_rejects_center_too_close_to_start():
    # center=5: st=5-12=-7 < 0 → ValueError
    with pytest.raises(ValueError, match="center"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=5)


@pytest.mark.unit
def test_build_centered_window_rejects_center_too_close_to_end():
    # center=95, T=100: ed=95+12+1=108 > 100 → ValueError
    with pytest.raises(ValueError, match="center"):
        build_centered_window(np.zeros((100, 8, 8), dtype=np.float32), center=95)


# ---------------------------------------------------------------------------
# Group B — SCFlowWrapper contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_scflow_wrapper_rejects_3d_input():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="ndim=4"):
        wrapper._validate_spike_pair(torch.randn(25, 8, 8), torch.randn(1, 25, 8, 8))


@pytest.mark.unit
def test_scflow_wrapper_rejects_5d_input():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="ndim=4"):
        wrapper._validate_spike_pair(torch.randn(1, 1, 25, 8, 8), torch.randn(1, 25, 8, 8))


@pytest.mark.unit
def test_scflow_wrapper_rejects_wrong_channels_on_spk2():
    wrapper = SCFlowWrapper.__new__(SCFlowWrapper)
    with pytest.raises(ValueError, match="channels=25"):
        wrapper._validate_spike_pair(torch.randn(1, 25, 8, 8), torch.randn(1, 11, 8, 8))


# ---------------------------------------------------------------------------
# Group C — ModelPlain contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_model_plain_rejects_l_flow_spike_ndim_4():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 25, 8, 8),  # ndim=4, should be 5
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data, need_H=False)


@pytest.mark.unit
def test_model_plain_rejects_l_flow_spike_wrong_channels():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 11, 8, 8),  # channels=11, not 25
    }
    with pytest.raises(ValueError, match="L_flow_spike"):
        model.feed_data(data, need_H=False)


@pytest.mark.unit
def test_model_plain_stores_l_flow_spike_on_valid_input():
    model = _build_stub_plain_model(flow_module="scflow")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
        "L_flow_spike": torch.randn(1, 4, 25, 8, 8),
    }
    model.feed_data(data, need_H=False)
    assert model.L_flow_spike is not None
    assert tuple(model.L_flow_spike.shape) == (1, 4, 25, 8, 8)


@pytest.mark.unit
def test_model_plain_flow_module_alias_spike_flow():
    model = _build_stub_plain_model(flow_module="spike_flow")
    assert model._flow_module_name() == "scflow"


@pytest.mark.unit
def test_model_plain_clears_l_flow_spike_for_non_scflow():
    model = _build_stub_plain_model(flow_module="spynet")
    data = {
        "L": torch.randn(1, 4, 7, 8, 8),
    }
    model.feed_data(data, need_H=False)
    assert model.L_flow_spike is None


# ---------------------------------------------------------------------------
# Group D — VRT contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_vrt_rejects_flow_spike_ndim_4():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    # pass ndim=4 tensor instead of required ndim=5
    with pytest.raises(ValueError, match="ndim"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(4, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_batch_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="B,T"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(2, 4, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_time_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="B,T"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 3, 25, 16, 16))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_spatial_mismatch():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="spatial"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 4, 25, 8, 8))


@pytest.mark.unit
def test_vrt_rejects_flow_spike_wrong_channels():
    vrt = VRT.__new__(VRT)
    vrt.spynet = _DummySpikeFlow()
    x = torch.randn(1, 4, 7, 16, 16)
    with pytest.raises(ValueError, match="channels=25"):
        vrt.get_flow_2frames(x, flow_spike=torch.randn(1, 4, 11, 16, 16))


# ---------------------------------------------------------------------------
# Group E — Dataset contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_dataset_no_spike_flow_config_disables_encoding25():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds._parse_spike_flow_config({}, optical_flow_module="")
    assert ds.use_encoding25_flow is False


@pytest.mark.unit
def test_dataset_load_path_construction_auto_mode(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    # artifact does not exist; we just check the error message contains tmp_path
    with pytest.raises(ValueError, match=str(tmp_path)):
        ds._load_encoded_flow_spike("clip_a", 1)


@pytest.mark.unit
def test_dataset_load_path_construction_explicit_root(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path / "other"
    ds.spike_flow_root = "/explicit/root"
    ds.spike_flow_dt = 10
    ds.filename_tmpl = "06d"
    with pytest.raises(ValueError, match="/explicit/root"):
        ds._load_encoded_flow_spike("clip_a", 1)


# ---------------------------------------------------------------------------
# Group F — Subframe encoding25 contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_compute_subframe_centers_t56_s4():
    """T_raw=56, S=4: centers in [12, 43], evenly spaced, sub_dt ~10.3."""
    centers = compute_subframe_centers(t_raw=56, num_subframes=4)
    assert len(centers) == 4
    assert centers[0] == 12
    assert centers[-1] == 43
    assert all(12 <= c <= 43 for c in centers)
    assert centers == sorted(centers)

@pytest.mark.unit
def test_compute_subframe_centers_t88_s4():
    """T_raw=88, S=4: centers in [12, 75], evenly spaced."""
    centers = compute_subframe_centers(t_raw=88, num_subframes=4)
    assert len(centers) == 4
    assert centers[0] == 12
    assert centers[-1] == 75
    assert centers == sorted(centers)

@pytest.mark.unit
def test_compute_subframe_centers_s1_returns_midpoint():
    """S=1: single center at midpoint of valid range."""
    centers = compute_subframe_centers(t_raw=56, num_subframes=1)
    assert len(centers) == 1
    assert centers[0] == (12 + 43) // 2  # 27

@pytest.mark.unit
def test_compute_subframe_centers_rejects_too_short():
    """T_raw=24 can't fit a 25-wide window."""
    with pytest.raises(ValueError, match="t_raw"):
        compute_subframe_centers(t_raw=24, num_subframes=1)

@pytest.mark.unit
def test_compute_subframe_centers_rejects_zero_subframes():
    with pytest.raises(ValueError, match="num_subframes"):
        compute_subframe_centers(t_raw=56, num_subframes=0)

@pytest.mark.unit
def test_validate_subframes_tensor_accepts_valid():
    arr = np.zeros((4, 25, 8, 8), dtype=np.float32)
    validate_subframes_tensor(arr, num_subframes=4)

@pytest.mark.unit
def test_validate_subframes_tensor_rejects_wrong_s():
    arr = np.zeros((3, 25, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="subframes"):
        validate_subframes_tensor(arr, num_subframes=4)

@pytest.mark.unit
def test_validate_subframes_tensor_rejects_wrong_channels():
    arr = np.zeros((4, 11, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="25"):
        validate_subframes_tensor(arr, num_subframes=4)

@pytest.mark.unit
def test_validate_subframes_tensor_rejects_3d():
    arr = np.zeros((25, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="ndim"):
        validate_subframes_tensor(arr, num_subframes=1)

@pytest.mark.unit
def test_build_output_dir_subframes_s4(tmp_path):
    out = build_output_dir_subframes(tmp_path / "clip", dt=10, num_subframes=4)
    assert out.name == "encoding25_dt10_s4"

@pytest.mark.unit
def test_build_output_dir_subframes_s1_backward_compat(tmp_path):
    out = build_output_dir_subframes(tmp_path / "clip", dt=10, num_subframes=1)
    assert out.name == "encoding25_dt10"


# ---------------------------------------------------------------------------
# Group G — build_scflow_subframe_windows contracts
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_scflow_subframe_windows_shape_t56_s4():
    from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import (
        build_scflow_subframe_windows,
    )
    spike = np.random.rand(56, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=4)
    assert result.shape == (4, 25, 8, 8)
    assert result.dtype == np.float32

@pytest.mark.unit
def test_build_scflow_subframe_windows_shape_t88_s4():
    from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import (
        build_scflow_subframe_windows,
    )
    spike = np.random.rand(88, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=4)
    assert result.shape == (4, 25, 8, 8)

@pytest.mark.unit
def test_build_scflow_subframe_windows_s1_backward_compat():
    from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import (
        build_scflow_subframe_windows,
    )
    spike = np.random.rand(56, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=1)
    assert result.shape == (1, 25, 8, 8)

@pytest.mark.unit
def test_build_scflow_subframe_windows_rejects_short():
    from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import (
        build_scflow_subframe_windows,
    )
    spike = np.random.rand(20, 8, 8).astype(np.float32)
    with pytest.raises(ValueError):
        build_scflow_subframe_windows(spike, num_subframes=4)
