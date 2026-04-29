import numpy as np
import pytest

from data.dataset_video_test import TestDataset
from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike


def _write_meta(tmp_path):
    meta = tmp_path / "meta.txt"
    meta.write_text("clipA 2 (16,16,3) 0")
    return str(meta)


def _train_opt(tmp_path, **overrides):
    opt = {
        "dataroot_gt": tmp_path / "gt",
        "dataroot_lq": tmp_path / "lq",
        "dataroot_spike": tmp_path / "spike",
        "meta_info_file": _write_meta(tmp_path),
        "name": "raw-window-train",
        "val_partition": "",
        "test_mode": False,
        "io_backend": {"type": "disk"},
        "use_hflip": False,
        "use_rot": False,
        "num_frame": 2,
        "scale": 4,
        "tfp_devices": [],
        "tfp_device": "cpu",
        "filename_tmpl": "08d",
        "filename_ext": "png",
        "spike": {
            "representation": "raw_window",
            "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
            "raw_window_length": None,
        },
        "tfp_half_win_length": 5,
    }
    opt.update(overrides)
    return opt


def test_train_dataset_raw_window_defaults_length_from_tfp_half_window(tmp_path):
    dataset = TrainDatasetRGBSpike(_train_opt(tmp_path))

    assert dataset.spike_representation == "raw_window"
    assert dataset.raw_window_length == 11
    assert dataset.spike_channels == 11


def test_train_dataset_rejects_raw_window_spike_channel_mismatch(tmp_path):
    opt = _train_opt(
        tmp_path,
        spike_channels=9,
        spike={
            "representation": "raw_window",
            "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
            "raw_window_length": 11,
        },
    )

    with pytest.raises(ValueError, match="raw_window_length=11"):
        TrainDatasetRGBSpike(opt)


def test_train_dataset_load_spike_voxel_returns_centered_raw_window(tmp_path, monkeypatch):
    dataset = TrainDatasetRGBSpike(
        _train_opt(
            tmp_path,
            spike={
                "representation": "raw_window",
                "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                "raw_window_length": 5,
            },
        )
    )
    spike_matrix = np.arange(9 * 2 * 2, dtype=np.float32).reshape(9, 2, 2)

    monkeypatch.setattr(
        "data.dataset_video_train_rgbspike.SpikeStream",
        lambda **kwargs: type(
            "StreamStub",
            (),
            {"get_spike_matrix": staticmethod(lambda flipud=True: spike_matrix)},
        )(),
    )

    spike = dataset._load_spike_voxel("clipA", 0, tmp_path / "clipA.dat")

    assert spike.shape == (5, 2, 2)
    assert np.array_equal(spike, spike_matrix[2:7])


def test_train_dataset_rejects_precomputed_raw_window_mode(tmp_path):
    opt = _train_opt(
        tmp_path,
        spike={
            "representation": "raw_window",
            "raw_window_length": 11,
            "precomputed": {"enable": True, "format": "npy", "root": "auto"},
            "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
        },
    )

    with pytest.raises(ValueError, match="precomputed spike artifacts"):
        TrainDatasetRGBSpike(opt)
