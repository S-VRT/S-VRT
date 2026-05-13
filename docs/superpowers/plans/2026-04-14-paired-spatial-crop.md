# Paired Spatial Crop 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正训练 data 层的 Spike crop 逻辑，使 RGB 和 Spike 在空间上对齐裁剪，而非将整张 Spike resize 到 RGB crop 尺寸。

**Architecture:** 修改 `paired_random_crop` 返回 LQ 坐标系的 crop 参数 `{top, left, lq_patch_size}`。在 `dataset_video_train_rgbspike.py` 中，用这些坐标按 RGB/Spike 分辨率比例换算出 Spike 上的对应 crop 区域，先 crop 再 resize。同时修正 flow_spike 的相同问题。

**Tech Stack:** Python, NumPy, OpenCV, PyTorch, pytest

**Spec:** `docs/superpowers/specs/2026-04-14-paired-spatial-crop-design.md`

---

### Task 1: `paired_random_crop` 返回 crop 参数

**Files:**
- Modify: `utils/utils_video.py:248-313`
- Test: `tests/utils/test_paired_random_crop.py` (create)

- [ ] **Step 1: Write failing test — 验证返回 3-tuple 含 crop_params**

```python
# tests/utils/test_paired_random_crop.py
import numpy as np
import utils.utils_video as utils_video


def test_returns_crop_params_dict():
    """paired_random_crop must return (gts, lqs, crop_params) 3-tuple."""
    h, w = 64, 80
    patch = 16
    gt = np.random.rand(h, w, 3).astype(np.float32)
    lq = np.random.rand(h, w, 3).astype(np.float32)
    result = utils_video.paired_random_crop(gt, lq, patch, scale=1)
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}"
    gts, lqs, crop_params = result
    assert isinstance(crop_params, dict)
    assert set(crop_params.keys()) == {"top", "left", "lq_patch_size"}
    assert crop_params["lq_patch_size"] == patch


def test_crop_params_coordinates_in_bounds():
    """top/left must be non-negative and within (h - patch, w - patch)."""
    h, w = 64, 80
    patch = 16
    gt = np.random.rand(h, w, 3).astype(np.float32)
    lq = np.random.rand(h, w, 3).astype(np.float32)
    for _ in range(50):
        _, _, crop_params = utils_video.paired_random_crop(gt, lq, patch, scale=1)
        assert 0 <= crop_params["top"] <= h - patch
        assert 0 <= crop_params["left"] <= w - patch


def test_crop_params_with_scale_factor():
    """With scale > 1, top/left are in LQ coordinate space, lq_patch_size = gt_patch_size // scale."""
    scale = 4
    gt_patch = 64
    lq_h, lq_w = 32, 40
    gt_h, gt_w = lq_h * scale, lq_w * scale
    gt = np.random.rand(gt_h, gt_w, 3).astype(np.float32)
    lq = np.random.rand(lq_h, lq_w, 3).astype(np.float32)
    _, _, crop_params = utils_video.paired_random_crop(gt, lq, gt_patch, scale=scale)
    assert crop_params["lq_patch_size"] == gt_patch // scale  # 16
    assert 0 <= crop_params["top"] <= lq_h - crop_params["lq_patch_size"]
    assert 0 <= crop_params["left"] <= lq_w - crop_params["lq_patch_size"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/utils/test_paired_random_crop.py -v`
Expected: FAIL — `len(result) == 3` assertion fails (currently returns 2-tuple)

- [ ] **Step 3: Modify `paired_random_crop` to return crop_params**

In `utils/utils_video.py`, change the return statement and docstring:

```python
# Line 264-266: Update docstring Returns section
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
        dict: Crop parameters with keys 'top', 'left', 'lq_patch_size'
            in LQ coordinate space.
```

```python
# Line 313: Change return statement from:
    return img_gts, img_lqs
# to:
    crop_params = {'top': top, 'left': left, 'lq_patch_size': lq_patch_size}
    return img_gts, img_lqs, crop_params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/utils/test_paired_random_crop.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add utils/utils_video.py tests/utils/test_paired_random_crop.py
git commit -m "feat: paired_random_crop returns crop_params dict

Return (gts, lqs, crop_params) 3-tuple so callers can use crop
coordinates for multi-modal spatial alignment."
```

---

### Task 2: 适配现有调用方

**Files:**
- Modify: `data/dataset_video_train.py:175`
- Modify: `tests/data/test_dataset_rgbspike_pack_modes.py:17`

- [ ] **Step 1: Fix `dataset_video_train.py` — ignore third return value**

```python
# Line 175, change from:
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)
# to:
        img_gts, img_lqs, _ = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)
```

- [ ] **Step 2: Fix test fixture — monkeypatch must return 3-tuple**

```python
# tests/data/test_dataset_rgbspike_pack_modes.py, line 17, change from:
        lambda gts, lqs, gt_size, scale, path: (gts, lqs),
# to:
        lambda gts, lqs, gt_size, scale, path: (gts, lqs, {"top": 0, "left": 0, "lq_patch_size": gt_size // scale}),
```

- [ ] **Step 3: Run existing tests to confirm no regressions**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/data/test_dataset_rgbspike_pack_modes.py tests/utils/test_paired_random_crop.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add data/dataset_video_train.py tests/data/test_dataset_rgbspike_pack_modes.py
git commit -m "fix: adapt callers to paired_random_crop 3-tuple return"
```

---

### Task 3: Spike paired crop — 核心改动

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py:291-324`
- Test: `tests/data/test_spike_paired_crop.py` (create)

- [ ] **Step 1: Write failing test — 验证 spike 按 RGB crop 坐标做空间对齐裁剪**

```python
# tests/data/test_spike_paired_crop.py
"""Verify that spike voxels are spatially cropped to match the RGB crop region,
not just resized from the full frame."""
import types

import cv2
import numpy as np
import pytest
import torch

import utils.utils_video as utils_video
from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike


# --- helpers ---

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
        "gt_size": 16,
        "tfp_devices": [],
        "tfp_device": "cpu",
        "filename_tmpl": "08d",
        "filename_ext": "png",
    }
    opt.update(overrides)
    return opt


def _build_dataset(tmp_path, **overrides):
    opt = _build_opt(tmp_path, **overrides)
    ds = TrainDatasetRGBSpike(opt)
    return ds


def _fake_loader(rgb_shape, spike_shape):
    """Return a _load_raw_frame that produces fixed-size arrays with spatial gradients."""
    def _loader(self, key):
        h, w = rgb_shape
        sh, sw = spike_shape
        # RGB: horizontal gradient [0..1] so we can verify crop position
        rgb = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        rgb = np.stack([rgb, rgb, rgb], axis=-1)  # (H, W, 3)
        # Spike: same horizontal gradient per channel
        spike_ch = np.tile(np.linspace(0, 1, sw, dtype=np.float32), (sh, 1))
        spike = np.stack([spike_ch] * self.spike_channels, axis=0).astype(np.float32)
        return {
            "lq": rgb.copy(),
            "gt": rgb.copy(),
            "spike": spike.copy(),
            "gt_path": "clipA/00000000",
        }
    return _loader


CROP_TOP = 8
CROP_LEFT = 16
CROP_PATCH = 16


@pytest.fixture(autouse=True)
def _patch_video_utils(monkeypatch):
    """Patch paired_random_crop to return a deterministic crop at known coordinates."""
    def _deterministic_crop(gts, lqs, gt_size, scale, path):
        if not isinstance(gts, list):
            gts = [gts]
        if not isinstance(lqs, list):
            lqs = [lqs]
        patch = gt_size // scale
        cropped_gts = [g[CROP_TOP:CROP_TOP+patch, CROP_LEFT:CROP_LEFT+patch, ...] for g in gts]
        cropped_lqs = [l[CROP_TOP:CROP_TOP+patch, CROP_LEFT:CROP_LEFT+patch, ...] for l in lqs]
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


class TestSpikePairedCrop:
    """The spike voxel must be cropped from the region corresponding to the
    RGB crop, not resized from the full frame."""

    RGB_H, RGB_W = 32, 64
    SPIKE_H, SPIKE_W = 50, 80

    def _get_dataset(self, tmp_path):
        ds = _build_dataset(
            tmp_path,
            spike_h=self.SPIKE_H,
            spike_w=self.SPIKE_W,
            gt_size=CROP_PATCH,
            scale=1,
        )
        ds._load_raw_frame = types.MethodType(
            _fake_loader((self.RGB_H, self.RGB_W), (self.SPIKE_H, self.SPIKE_W)),
            ds,
        )
        return ds

    def test_spike_crop_uses_proportional_coordinates(self, tmp_path):
        """After paired crop, the spike patch should correspond to the same
        physical region as the RGB patch, not be a downscaled full-frame."""
        ds = self._get_dataset(tmp_path)
        sample = ds[0]

        # The RGB LQ has a horizontal gradient [0..1] across 64 pixels.
        # Crop at left=16, patch=16 means we get columns 16..31 of 64 →
        # gradient range [16/64, 31/64] = [0.25, ~0.484].
        rgb_patch = sample["L"][0, :3, :, :]  # (3, 16, 16)
        rgb_left_col_mean = rgb_patch[0, :, 0].mean().item()
        rgb_right_col_mean = rgb_patch[0, :, -1].mean().item()

        # Spike should have the same proportional gradient range,
        # because it is cropped from the corresponding region and then resized.
        spike_patch = sample["L"][0, 3:, :, :]  # (S, 16, 16)
        spike_left_col_mean = spike_patch[0, :, 0].mean().item()
        spike_right_col_mean = spike_patch[0, :, -1].mean().item()

        # If spike were wrongly resized from full frame, its range would be [0, 1].
        # With correct paired crop, it should be close to the RGB range.
        assert spike_left_col_mean > 0.15, (
            f"Spike left column mean {spike_left_col_mean:.3f} is too low — "
            "spike may not have been cropped to the RGB region"
        )
        assert spike_right_col_mean < 0.6, (
            f"Spike right column mean {spike_right_col_mean:.3f} is too high — "
            "spike may not have been cropped to the RGB region"
        )
        # Tighter check: spike gradient range should approximate RGB gradient range
        assert abs(spike_left_col_mean - rgb_left_col_mean) < 0.1
        assert abs(spike_right_col_mean - rgb_right_col_mean) < 0.1

    def test_spike_output_spatial_shape_matches_rgb(self, tmp_path):
        """After crop + resize, spike spatial dims must match RGB crop dims."""
        ds = self._get_dataset(tmp_path)
        sample = ds[0]
        L = sample["L"]
        # All channels share the same H, W (ensured by concat)
        assert L.shape[2] == CROP_PATCH
        assert L.shape[3] == CROP_PATCH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/data/test_spike_paired_crop.py -v`
Expected: `test_spike_crop_uses_proportional_coordinates` FAILS — spike gradient range is [0, 1] (full frame resize)

- [ ] **Step 3: Implement paired spike crop in `dataset_video_train_rgbspike.py`**

Replace lines 291–324 with:

```python
        # Record LQ spatial size before crop (needed for spike coordinate mapping)
        lq_h_orig, lq_w_orig = img_lqs[0].shape[:2]

        # randomly crop RGB frames
        img_gts, img_lqs, crop_params = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path_reference)

        # Compute spike crop region corresponding to the RGB crop
        cropped_h, cropped_w = img_lqs[0].shape[:2]
        ratio_h = self.spike_h / lq_h_orig
        ratio_w = self.spike_w / lq_w_orig
        sp_top = round(crop_params['top'] * ratio_h)
        sp_left = round(crop_params['left'] * ratio_w)
        sp_crop_h = max(round(crop_params['lq_patch_size'] * ratio_h), 1)
        sp_crop_w = max(round(crop_params['lq_patch_size'] * ratio_w), 1)
        sp_top = min(sp_top, self.spike_h - sp_crop_h)
        sp_left = min(sp_left, self.spike_w - sp_crop_w)

        # Crop and resize spike voxels to match the cropped RGB size
        spike_voxels_resized = []
        for spike_voxel in spike_voxels:
            if spike_voxel.ndim != 3:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Spike voxel must be [S,H,W], got shape {spike_voxel.shape}."
                )
            if spike_voxel.shape[0] != self.spike_channels:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Spike channels mismatch before resize: "
                    f"expected {self.spike_channels}, got {spike_voxel.shape[0]}."
                )
            # Crop spike to the region corresponding to the RGB crop
            spike_cropped = spike_voxel[:, sp_top:sp_top + sp_crop_h, sp_left:sp_left + sp_crop_w]
            # Resize each channel to match RGB crop spatial size
            spike_voxel_resized = []
            for ch in range(self.spike_channels):
                spike_ch = spike_cropped[ch]  # (crop_h, crop_w)
                spike_ch_resized = cv2.resize(spike_ch, (cropped_w, cropped_h), interpolation=cv2.INTER_LINEAR)
                spike_voxel_resized.append(spike_ch_resized)
            spike_voxel_resized = np.stack(spike_voxel_resized, axis=0)  # (S, H, W)
            spike_voxels_resized.append(spike_voxel_resized)

        # Crop and resize flow spikes with the same spatial region
        flow_spikes_resized = []
        if self.use_encoding25_flow:
            for flow_spike in flow_spikes:
                validate_encoding25_tensor(flow_spike)
                flow_cropped = flow_spike[:, sp_top:sp_top + sp_crop_h, sp_left:sp_left + sp_crop_w]
                flow_resized = []
                for ch in range(flow_cropped.shape[0]):
                    flow_ch_resized = cv2.resize(flow_cropped[ch], (cropped_w, cropped_h), interpolation=cv2.INTER_LINEAR)
                    flow_resized.append(flow_ch_resized)
                flow_spikes_resized.append(np.stack(flow_resized, axis=0).astype(np.float32))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/data/test_spike_paired_crop.py tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add data/dataset_video_train_rgbspike.py tests/data/test_spike_paired_crop.py
git commit -m "fix: spatially align spike crop to RGB crop region

Previously, spike voxels were resized from the full frame to match the
RGB crop size, breaking spatial alignment. Now spike is cropped from the
proportionally corresponding region before resizing."
```

---

### Task 4: 全套回归测试

**Files:** (no new files)

- [ ] **Step 1: Run full test suite**

Run: `cd /home/wuhy/projects/S-VRT && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: If any failures, fix and re-run**

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address regression from paired spatial crop changes"
```
