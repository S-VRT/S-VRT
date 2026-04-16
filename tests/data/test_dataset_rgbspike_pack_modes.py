import types

import cv2
import numpy as np
import pytest
import torch

import utils.utils_video as utils_video
from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike


@pytest.fixture(autouse=True)
def _patch_video_utils(monkeypatch):
    monkeypatch.setattr(
        utils_video,
        "paired_random_crop",
        lambda gts, lqs, gt_size, scale, path: (gts, lqs),
    )
    monkeypatch.setattr(utils_video, "augment", lambda imgs, hflip, rot: imgs)

    def _img2tensor(imgs, bgr2rgb=False):
        return [
            torch.from_numpy(np.transpose(img.astype(np.float32), (2, 0, 1)).copy())
            for img in imgs
        ]

    monkeypatch.setattr(utils_video, "img2tensor", _img2tensor)


def _write_meta(tmp_path):
    meta = tmp_path / "meta.txt"
    meta.write_text("clipA 2 (16,16,3) 0")
    return str(meta)


def _fake_loader(shape=(8, 8, 3), spike_channels=None):
    def _loader(self, key):
        h, w = shape[:2]
        value = float(int(key.split("/")[-1]))
        channels = self.spike_channels if spike_channels is None else spike_channels
        return {
            "lq": np.full((h, w, 3), value, dtype=np.float32),
            "gt": np.full((h, w, 3), value + 0.1, dtype=np.float32),
            "spike": np.full((channels, h, w), value + 0.2, dtype=np.float32),
            "gt_path": "clipA/00000000",
        }

    return _loader


def _build_opt(tmp_path, **overrides):
    opt = {
        "dataroot_gt": tmp_path / "gt",
        "dataroot_lq": tmp_path / "lq",
        "dataroot_spike": tmp_path / "spike",
        "meta_info_file": _write_meta(tmp_path),
        "name": "custom",
        "val_partition": "",
        "test_mode": False,
        "io_backend": {"type": "disk"},
        "use_hflip": False,
        "use_rot": False,
        "num_frame": 2,
        "scale": 4,
        "spike_channels": 2,
        "tfp_devices": [],
        "tfp_device": "cpu",
        "filename_tmpl": "08d",
        "filename_ext": "png",
    }
    opt.update(overrides)
    return opt


def _build_dataset(tmp_path, **overrides):
    opt = _build_opt(tmp_path, **overrides)
    dataset = TrainDatasetRGBSpike(opt)
    dataset._load_raw_frame = types.MethodType(_fake_loader(), dataset)
    return dataset


def test_concat_pack_mode_retains_L(tmp_path):
    expected_spike_channels = 2
    dataset = _build_dataset(tmp_path, spike_channels=expected_spike_channels)
    sample = dataset[0]
    assert set(sample.keys()) == {"L", "H", "key"}
    assert sample["L"].shape[1] == 3 + expected_spike_channels


def test_dual_pack_mode_exposes_rgb_spike(tmp_path):
    expected_spike_channels = 2
    dataset = _build_dataset(
        tmp_path,
        input_pack_mode="dual",
        keep_legacy_l=True,
        spike_channels=expected_spike_channels,
    )
    sample = dataset[0]
    assert {"L", "L_rgb", "L_spike", "H", "key"} == set(sample.keys())
    assert sample["L_rgb"].shape[1] == 3
    assert sample["L_spike"].shape[1] == expected_spike_channels
    assert sample["L_rgb"].shape[0] == sample["L_spike"].shape[0]
    assert sample["L_rgb"].shape[2:] == sample["L_spike"].shape[2:]
    assert torch.allclose(
        sample["L"], torch.cat([sample["L_rgb"], sample["L_spike"]], dim=1)
    )


def test_dual_pack_mode_can_drop_legacy_L(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual", keep_legacy_l=False)
    sample = dataset[0]
    assert "L" not in sample
    assert {"L_rgb", "L_spike", "H", "key"} == set(sample.keys())


def test_dual_pack_mode_defaults_to_keep_legacy_l(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual")
    sample = dataset[0]
    assert "L" in sample
    assert {"L", "L_rgb", "L_spike", "H", "key"} == set(sample.keys())


def test_concat_pack_mode_ignores_keep_legacy_flag(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="concat", keep_legacy_l=False)
    sample = dataset[0]
    assert set(sample.keys()) == {"L", "H", "key"}


def test_input_pack_mode_normalizes_whitespace_and_case(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode=" DuAl ")
    sample = dataset[0]
    assert {"L", "L_rgb", "L_spike", "H", "key"} == set(sample.keys())


def test_invalid_input_pack_mode_raises(tmp_path):
    opt = _build_opt(tmp_path, input_pack_mode="invalid")
    with pytest.raises(ValueError, match="input_pack_mode must be one of"):
        TrainDatasetRGBSpike(opt)


def test_dual_mode_spike_channel_mismatch_raises(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual", spike_channels=2)
    dataset._load_raw_frame = types.MethodType(_fake_loader(spike_channels=3), dataset)
    with pytest.raises(ValueError, match="Spike channels mismatch"):
        _ = dataset[0]


def test_dual_mode_spatial_mismatch_raises(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual", spike_channels=2)

    def _bad_resize(src, dsize, interpolation=None):
        bad_h = dsize[1] + 1
        bad_w = dsize[0]
        return np.zeros((bad_h, bad_w), dtype=src.dtype)

    monkeypatch.setattr(cv2, "resize", _bad_resize)
    with pytest.raises(ValueError, match="matching spatial shape"):
        _ = dataset[0]


def test_nested_spike_reconstruction_num_bins_drives_default_channels(tmp_path):
    opt = _build_opt(
        tmp_path,
        spike_channels=None,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 3}},
    )
    opt.pop("spike_channels", None)
    dataset = TrainDatasetRGBSpike(opt)
    assert dataset.spike_channels == 3


def test_compat_keep_legacy_l_overrides_top_level_flag(tmp_path):
    opt = _build_opt(
        tmp_path,
        input_pack_mode="dual",
        keep_legacy_l=True,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 2}},
    )
    opt["compat"] = {"keep_legacy_L": False}
    dataset = TrainDatasetRGBSpike(opt)
    assert dataset.keep_legacy_l is False


def test_conflicting_nested_and_legacy_reconstruction_raises(tmp_path):
    opt = _build_opt(
        tmp_path,
        spike_channels=4,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 4}},
        spike_reconstruction="middle_tfp",
    )
    with pytest.raises(ValueError, match="Conflicting reconstruction types"):
        TrainDatasetRGBSpike(opt)


def test_legacy_reconstruction_dict_matching_nested_does_not_raise(tmp_path):
    opt = _build_opt(
        tmp_path,
        spike_channels=4,
        spike={"reconstruction": {"type": "spikecv_tfp", "num_bins": 4}},
        spike_reconstruction={"type": "spikecv_tfp"},
    )
    dataset = TrainDatasetRGBSpike(opt)
    assert dataset.spike_reconstruction == "spikecv_tfp"
