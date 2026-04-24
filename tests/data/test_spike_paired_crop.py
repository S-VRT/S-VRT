import types

import numpy as np
import pytest
import torch

import utils.utils_video as utils_video
from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike


CROP_TOP = 8
CROP_LEFT = 16
CROP_PATCH = 16


def _write_meta(tmp_path):
    meta = tmp_path / "meta.txt"
    meta.write_text("clipA 2 (16,16,3) 0")
    return str(meta)


def _build_opt(tmp_path, **overrides):
    opt = {
        "dataroot_gt": tmp_path / "gt",
        "dataroot_lq": tmp_path / "lq",
        "dataroot_spike": tmp_path / "spike",
        "meta_info_file": _write_meta(tmp_path),
        "name": "test",
        "val_partition": "",
        "test_mode": False,
        "io_backend": {"type": "disk"},
        "use_hflip": False,
        "use_rot": False,
        "num_frame": 2,
        "scale": 1,
        "spike_channels": 2,
        "spike_h": 50,
        "spike_w": 80,
        "gt_size": CROP_PATCH,
        "tfp_devices": [],
        "tfp_device": "cpu",
        "filename_tmpl": "08d",
        "filename_ext": "png",
    }
    opt.update(overrides)
    return opt


def _fake_loader(rgb_shape, spike_shape):
    def _loader(self, key):
        h, w = rgb_shape
        spike_h, spike_w = spike_shape
        rgb_ch = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        rgb = np.stack([rgb_ch, rgb_ch, rgb_ch], axis=-1)
        spike_ch = np.tile(np.linspace(0, 1, spike_w, dtype=np.float32), (spike_h, 1))
        spike = np.stack([spike_ch] * self.spike_channels, axis=0).astype(np.float32)
        return {
            "lq": rgb.copy(),
            "gt": rgb.copy(),
            "spike": spike.copy(),
            "gt_path": "clipA/00000000",
        }

    return _loader


@pytest.fixture(autouse=True)
def _patch_video_utils(monkeypatch):
    def _deterministic_crop(gts, lqs, gt_size, scale, path):
        if not isinstance(gts, list):
            gts = [gts]
        if not isinstance(lqs, list):
            lqs = [lqs]
        patch = gt_size // scale
        cropped_gts = [g[CROP_TOP:CROP_TOP + patch, CROP_LEFT:CROP_LEFT + patch, ...] for g in gts]
        cropped_lqs = [l[CROP_TOP:CROP_TOP + patch, CROP_LEFT:CROP_LEFT + patch, ...] for l in lqs]
        crop_params = {"top": CROP_TOP, "left": CROP_LEFT, "lq_patch_size": patch}
        return cropped_gts, cropped_lqs, crop_params

    monkeypatch.setattr(utils_video, "paired_random_crop", _deterministic_crop)
    monkeypatch.setattr(utils_video, "augment", lambda imgs, hflip, rot: imgs)

    def _img2tensor(imgs, bgr2rgb=False):
        return [
            torch.from_numpy(np.transpose(img.astype(np.float32), (2, 0, 1)).copy())
            for img in imgs
        ]

    monkeypatch.setattr(utils_video, "img2tensor", _img2tensor)


def _build_dataset(tmp_path):
    dataset = TrainDatasetRGBSpike(_build_opt(tmp_path))
    dataset._load_raw_frame = types.MethodType(_fake_loader((32, 64), (50, 80)), dataset)
    return dataset


def test_spike_crop_uses_proportional_coordinates(tmp_path):
    dataset = _build_dataset(tmp_path)
    sample = dataset[0]

    rgb_patch = sample["L"][0, :3, :, :]
    rgb_left_col_mean = rgb_patch[0, :, 0].mean().item()
    rgb_right_col_mean = rgb_patch[0, :, -1].mean().item()

    spike_patch = sample["L"][0, 3:, :, :]
    spike_left_col_mean = spike_patch[0, :, 0].mean().item()
    spike_right_col_mean = spike_patch[0, :, -1].mean().item()

    assert spike_left_col_mean > 0.15
    assert spike_right_col_mean < 0.6
    assert abs(spike_left_col_mean - rgb_left_col_mean) < 0.1
    assert abs(spike_right_col_mean - rgb_right_col_mean) < 0.1


def test_spike_output_spatial_shape_matches_rgb(tmp_path):
    dataset = _build_dataset(tmp_path)
    sample = dataset[0]

    assert sample["L"].shape[2] == CROP_PATCH
    assert sample["L"].shape[3] == CROP_PATCH
