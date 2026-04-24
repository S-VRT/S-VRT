I'm using the writing-plans skill to create the implementation plan.
# Dual Input Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `TrainDatasetRGBSpike` to expose both concat and dual pack contracts while keeping legacy `L` available when requested.

**Architecture:** Parse new config flags, split the normalized tensor into RGB and spike portions inside `__getitem__`, and switch the returned dict based on the requested pack mode so downstream consumers always get a stable API.

**Tech Stack:** Python 3, PyTorch, NumPy, OpenCV, PyTest.

---

### Task 1: Pack mode contract tests

**Files:**
- Modify: `tests/data/test_dataset_rgbspike_pack_modes.py`

- [ ] **Step 1: Write the failing tests**

```python
import types

import numpy as np
import torch
import pytest

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


def _fake_loader(shape=(8, 8, 3)):
    def _loader(self, key):
        h, w = shape[:2]
        value = float(int(key.split("/")[-1]))
        return {
            "lq": np.full((h, w, 3), value, dtype=np.float32),
            "gt": np.full((h, w, 3), value + 0.1, dtype=np.float32),
            "spike": np.full(
                (self.spike_channels, h, w), value + 0.2, dtype=np.float32
            ),
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
    dataset = _build_dataset(tmp_path)
    sample = dataset[0]
    assert set(sample.keys()) == {"L", "H", "key"}
    assert sample["L"].shape[1] == 3 + dataset.spike_channels


def test_dual_pack_mode_exposes_rgb_spike(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual", keep_legacy_l=True)
    sample = dataset[0]
    assert {"L", "L_rgb", "L_spike", "H", "key"} == set(sample.keys())
    assert sample["L_rgb"].shape[1] == 3
    assert sample["L_spike"].shape[1] == dataset.spike_channels
    assert torch.allclose(
        sample["L"], torch.cat([sample["L_rgb"], sample["L_spike"]], dim=1)
    )


def test_dual_pack_mode_can_drop_legacy_L(tmp_path):
    dataset = _build_dataset(tmp_path, input_pack_mode="dual", keep_legacy_l=False)
    sample = dataset[0]
    assert "L" not in sample
    assert {"L_rgb", "L_spike", "H", "key"} == set(sample.keys())
```

- [ ] **Step 2: Run the pack-mode test script and confirm it fails**

Run: `pytest tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: FAIL with `AssertionError: {'L_rgb', 'L_spike', ...}` not present because the dataset currently only returns `L`.

### Task 2: Dataset pack mode implementation

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`

- [ ] **Step 1: Parse the new config options**

```python
        self.input_pack_mode = str(opt.get("input_pack_mode", "concat")).lower()
        if self.input_pack_mode not in {"concat", "dual"}:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Unsupported input_pack_mode "
                f"{self.input_pack_mode!r}; choose 'concat' or 'dual'."
            )
        self.keep_legacy_l = bool(opt.get("keep_legacy_l", True))
```

- [ ] **Step 2: Split channels inside `__getitem__` and switch outputs**

```python
        # Existing normalization check stays in place
        if img_lqs.size(1) != self.expected_lq_channels:
            ...

        rgb_part = img_lqs[:, :3, :, :]
        spike_part = img_lqs[:, 3:, :, :]

        if self.input_pack_mode == "dual":
            if rgb_part.shape[2:] != spike_part.shape[2:]:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Spatial mismatch between RGB {rgb_part.shape[2:]} "
                    f"and spike {spike_part.shape[2:]} tensors in dual mode."
                )
            payload = {
                "L_rgb": rgb_part,
                "L_spike": spike_part,
                "H": img_gts,
                "key": key,
            }
            if self.keep_legacy_l:
                payload["L"] = img_lqs
            return payload
        return {"L": img_lqs, "H": img_gts, "key": key}
```

- [ ] **Step 3: Run the new pack-mode tests to confirm success**

Run: `pytest tests/data/test_dataset_rgbspike_pack_modes.py -v`
Expected: PASS

## Self-Review
- Spec coverage: Task 1 materializes Step 1 (failing tests) and Task 2 handles Steps 2–5 (config parsing, dual outputs, validations, and final test run).
- Placeholder scan: All steps include concrete code/commands instead of placeholders.
- Type consistency: New `input_pack_mode` and `keep_legacy_l` names match references in both tests and dataset logic.

Plan complete and saved to `docs/superpowers/plans/2026-04-11-dual-input-packing.md`. Two execution options:
1. **Subagent-Driven (recommended)** – dispatch per task with review checkpoints.
2. **Inline Execution** – keep working in this session using executing-plans.

Which approach would you like?
