# PASE-Residual Raw-Window Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `fusion.operator="pase_residual"` so it can consume centered raw spike windows per RGB frame while preserving the existing structured early-fusion contract and keeping TFP as a supported comparison path.

**Architecture:** Add a small spike-representation helper that extracts centered raw windows from decoded `spike_matrix[T,H,W]`, then wire both train and test datasets to resolve a new `spike.representation` mode of `raw_window`. Keep `PaseResidualFusionOperator` source-agnostic, add VRT-side validation so raw-window input is only legal with `pase_residual` in phase 1, and surface representation metadata for analysis and logging.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing RGB+Spike dataset pipeline, existing VRT/fusion stack

---

## File Map

- Create: `data/spike_recc/raw_window.py`
  - Focused helper for validating and extracting centered raw spike windows from `spike_matrix[T,H,W]`.
- Modify: `data/spike_recc/__init__.py`
  - Re-export the raw-window helper for dataset use and tests.
- Create: `tests/data/test_spike_raw_window.py`
  - Unit tests for centered extraction and validation failures.
- Modify: `data/dataset_video_train_rgbspike.py`
  - Parse `spike.representation` and `raw_window_length`, resolve effective spike channels, reject unsupported raw-window combinations, and branch `_load_spike_voxel()` between TFP and raw-window paths.
- Modify: `data/dataset_video_test.py`
  - Mirror the train-dataset representation parsing and raw-window loading behavior for evaluation/inference.
- Create: `tests/data/test_dataset_rgbspike_raw_window.py`
  - Dataset-specific tests for raw-window config resolution and direct `.dat` decoding behavior in both train and test datasets.
- Modify: `models/architectures/vrt/vrt.py`
  - Resolve dataset spike representation at build time, reject phase-1 raw-window input for non-`pase_residual` operators, and attach representation metadata to `_last_fusion_meta`.
- Modify: `tests/models/test_vrt_fusion_integration.py`
  - Lock VRT constructor behavior and forward-time metadata for `pase_residual + raw_window`.
- Create: `options/gopro_rgbspike_server_pase_residual_raw_window.json`
  - Runnable ablation config with `raw_ingress_chans=44` and `spike.representation="raw_window"`.

### Task 1: Add a Reusable Centered Raw-Window Extraction Helper

**Files:**
- Create: `data/spike_recc/raw_window.py`
- Modify: `data/spike_recc/__init__.py`
- Create: `tests/data/test_spike_raw_window.py`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/data/test_spike_raw_window.py` with these tests:

```python
import numpy as np
import pytest

from data.spike_recc import extract_centered_raw_window


def test_extract_centered_raw_window_uses_middle_by_default():
    spike_matrix = np.arange(9 * 2 * 3, dtype=np.float32).reshape(9, 2, 3)

    window = extract_centered_raw_window(spike_matrix, window_length=5)

    assert window.shape == (5, 2, 3)
    assert np.array_equal(window, spike_matrix[2:7])


def test_extract_centered_raw_window_accepts_explicit_center():
    spike_matrix = np.arange(11, dtype=np.float32).reshape(11, 1, 1)

    window = extract_centered_raw_window(spike_matrix, window_length=3, center_index=7)

    assert window.shape == (3, 1, 1)
    assert np.array_equal(window[:, 0, 0], np.array([6.0, 7.0, 8.0], dtype=np.float32))


@pytest.mark.parametrize("bad_length", [0, 4, -3])
def test_extract_centered_raw_window_rejects_non_positive_or_even_lengths(bad_length):
    spike_matrix = np.zeros((9, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="window_length"):
        extract_centered_raw_window(spike_matrix, window_length=bad_length)


def test_extract_centered_raw_window_rejects_window_larger_than_available_time_axis():
    spike_matrix = np.zeros((7, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="window_length=9"):
        extract_centered_raw_window(spike_matrix, window_length=9)
```

- [ ] **Step 2: Run the focused helper tests and confirm they fail**

Run:

```bash
python -m pytest tests/data/test_spike_raw_window.py -v
```

Expected: FAIL with `cannot import name 'extract_centered_raw_window'`.

- [ ] **Step 3: Implement the helper and export it**

Create `data/spike_recc/raw_window.py`:

```python
from __future__ import annotations

import numpy as np


def extract_centered_raw_window(
    spike_matrix: np.ndarray,
    window_length: int,
    center_index: int | None = None,
) -> np.ndarray:
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be 3D (T, H, W), got shape {spike_matrix.shape}")
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError(f"window_length must be a positive odd integer, got {window_length}")

    total_steps = int(spike_matrix.shape[0])
    if window_length > total_steps:
        raise ValueError(
            f"window_length={window_length} exceeds available spike steps T={total_steps}"
        )

    resolved_center = total_steps // 2 if center_index is None else int(center_index)
    half = window_length // 2
    start = resolved_center - half
    end = resolved_center + half + 1
    if start < 0 or end > total_steps:
        raise ValueError(
            f"Centered raw window [{start}:{end}] is out of bounds for T={total_steps}"
        )

    return spike_matrix[start:end].astype(np.float32, copy=False)


__all__ = ["extract_centered_raw_window"]
```

Update `data/spike_recc/__init__.py`:

```python
from .raw_window import (
    extract_centered_raw_window,
)
```

```python
    'extract_centered_raw_window',
```

- [ ] **Step 4: Re-run the focused helper tests and confirm they pass**

Run:

```bash
python -m pytest tests/data/test_spike_raw_window.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/spike_recc/raw_window.py data/spike_recc/__init__.py tests/data/test_spike_raw_window.py
git commit -m "feat(data): add centered raw spike window helper"
```

### Task 2: Teach `TrainDatasetRGBSpike` to Resolve and Load `raw_window` Spike Representation

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`
- Create: `tests/data/test_dataset_rgbspike_raw_window.py`

- [ ] **Step 1: Write the failing train-dataset tests**

Create `tests/data/test_dataset_rgbspike_raw_window.py` with these train-dataset tests:

```python
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
```

- [ ] **Step 2: Run the focused train-dataset tests and confirm they fail**

Run:

```bash
python -m pytest tests/data/test_dataset_rgbspike_raw_window.py -k "train_dataset" -v
```

Expected: FAIL because `TrainDatasetRGBSpike` does not yet parse `spike.representation`, does not resolve `raw_window_length`, and always returns TFP voxels.

- [ ] **Step 3: Implement train-dataset raw-window parsing and loading**

Update `data/dataset_video_train_rgbspike.py` near spike-config parsing:

```python
from data.spike_recc import SpikeStream, extract_centered_raw_window, voxelize_spikes_tfp
```

```python
        self.spike_representation = str(spike_cfg.get('representation', 'tfp')).strip().lower()
        if self.spike_representation not in {'tfp', 'raw_window'}:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Unsupported spike.representation={self.spike_representation!r}."
            )

        raw_window_length_cfg = spike_cfg.get('raw_window_length', None)
        if self.spike_representation == 'raw_window':
            if raw_window_length_cfg is None:
                self.raw_window_length = int(2 * int(self.tfp_half_win_length) + 1)
            else:
                self.raw_window_length = int(raw_window_length_cfg)
            if self.raw_window_length <= 0 or self.raw_window_length % 2 == 0:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] raw_window_length must be a positive odd integer, "
                    f"got {self.raw_window_length}."
                )
        else:
            self.raw_window_length = None
```

Replace the current default-channel resolution block with:

```python
        if self.spike_representation == 'raw_window':
            default_spike_channels = int(self.raw_window_length)
        else:
            default_spike_channels = int(nested_num_bins) if nested_num_bins is not None else 4
        self.spike_channels = int(opt.get('spike_channels', default_spike_channels))

        if self.spike_representation == 'raw_window':
            if 'spike_channels' in opt and int(opt['spike_channels']) != int(self.raw_window_length):
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Conflicting raw-window channels: "
                    f"spike_channels={int(opt['spike_channels'])} vs raw_window_length={int(self.raw_window_length)}."
                )
        elif nested_num_bins is not None and 'spike_channels' in opt and int(opt['spike_channels']) != int(nested_num_bins):
            raise ValueError(
                f"[TrainDatasetRGBSpike] Conflicting channel settings: spike_channels={int(opt['spike_channels'])} "
                f"vs spike.reconstruction.num_bins={int(nested_num_bins)}."
            )
```

Add raw-window-specific guards after precomputed spike settings are parsed:

```python
        if self.spike_representation == 'raw_window':
            if self.use_precomputed_spike:
                raise ValueError(
                    "[TrainDatasetRGBSpike] raw_window representation does not support precomputed spike artifacts."
                )
            if self.spike_reconstruction not in {'spikecv_tfp'}:
                raise ValueError(
                    "[TrainDatasetRGBSpike] raw_window representation requires spike.reconstruction.type='spikecv_tfp' "
                    "as the baseline center-alignment source."
                )
```

Update `_load_spike_voxel()`:

```python
        if self.spike_representation == 'raw_window':
            return extract_centered_raw_window(
                spike_matrix,
                window_length=self.raw_window_length,
            )

        if self.spike_reconstruction in {'middle_tfp', 'middle-tfp'}:
            spike_frame = self._middle_tfp_reconstructor(spike_matrix)
            return spike_frame[np.newaxis, ...].astype(np.float32)
```

- [ ] **Step 4: Re-run the focused train-dataset tests and confirm they pass**

Run:

```bash
python -m pytest tests/data/test_dataset_rgbspike_raw_window.py -k "train_dataset" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/dataset_video_train_rgbspike.py tests/data/test_dataset_rgbspike_raw_window.py
git commit -m "feat(data): add raw-window train spike representation"
```

### Task 3: Mirror `raw_window` Support into `TestDataset` and Gate VRT to `pase_residual`

**Files:**
- Modify: `data/dataset_video_test.py`
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `tests/data/test_dataset_rgbspike_raw_window.py`
- Modify: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Write the failing test-dataset and VRT tests**

Append these tests to `tests/data/test_dataset_rgbspike_raw_window.py`:

```python
def _test_opt(tmp_path, **overrides):
    opt = {
        "dataroot_gt": str(tmp_path / "gt"),
        "dataroot_lq": str(tmp_path / "lq"),
        "dataroot_spike": str(tmp_path / "spike"),
        "cache_data": False,
        "io_backend": {"type": "disk"},
        "name": "raw-window-test",
        "num_frame": 2,
        "padding": "reflection",
        "spike": {
            "representation": "raw_window",
            "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
            "raw_window_length": None,
        },
        "tfp_half_win_length": 6,
    }
    opt.update(overrides)
    return opt


def test_test_dataset_raw_window_defaults_length_from_tfp_half_window(tmp_path):
    dataset = TestDataset(_test_opt(tmp_path))

    assert dataset.spike_representation == "raw_window"
    assert dataset.raw_window_length == 13
    assert dataset.spike_channels == 13


def test_test_dataset_load_spike_voxel_returns_centered_raw_window(tmp_path, monkeypatch):
    dataset = TestDataset(
        _test_opt(
            tmp_path,
            spike={"representation": "raw_window", "raw_window_length": 5, "reconstruction": {"type": "spikecv_tfp", "num_bins": 4}},
        )
    )
    spike_matrix = np.arange(9 * 2 * 2, dtype=np.float32).reshape(9, 2, 2)

    monkeypatch.setattr(
        "data.dataset_video_test.SpikeStream",
        lambda **kwargs: type(
            "StreamStub",
            (),
            {"get_spike_matrix": staticmethod(lambda flipud=True: spike_matrix)},
        )(),
    )

    spike = dataset._load_spike_voxel(str(tmp_path / "clipA.dat"))

    assert spike.shape == (5, 2, 2)
    assert np.array_equal(spike, spike_matrix[2:7])
```

Append these tests to `tests/models/test_vrt_fusion_integration.py`:

```python
def test_vrt_builds_with_pase_residual_raw_window_config():
    model = VRT(
        upscale=1,
        in_chans=44,
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
        opt={
            "datasets": {
                "train": {
                    "spike": {
                        "representation": "raw_window",
                        "raw_window_length": 41,
                        "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                    }
                }
            },
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 44},
                "fusion": {
                    "placement": "early",
                    "operator": "pase_residual",
                    "out_chans": 3,
                    "early": {"frame_contract": "collapsed"},
                },
            },
        },
    )

    assert model.fusion_operator.spike_chans == 41
    assert model._fusion_spike_representation == "raw_window"


def test_vrt_rejects_raw_window_with_non_pase_residual_operator():
    with pytest.raises(ValueError, match="raw_window"):
        VRT(
            upscale=1,
            in_chans=44,
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
            opt={
                "datasets": {
                    "train": {
                        "spike": {
                            "representation": "raw_window",
                            "raw_window_length": 41,
                            "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                        }
                    }
                },
                "netG": {
                    "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 44},
                    "fusion": {
                        "placement": "early",
                        "operator": "gated",
                        "out_chans": 3,
                    },
                },
            },
        )


def test_vrt_forward_records_raw_window_representation_metadata(monkeypatch):
    model = VRT(
        upscale=1,
        in_chans=44,
        out_chans=3,
        img_size=[6, 8, 8],
        window_size=[6, 4, 4],
        depths=[1] * 8,
        indep_reconsts=[],
        embed_dims=[16] * 8,
        num_heads=[1] * 8,
        pa_frames=2,
        use_flash_attn=False,
        optical_flow={"module": "spynet", "checkpoint": None, "params": {}},
        opt={
            "datasets": {
                "train": {
                    "spike": {
                        "representation": "raw_window",
                        "raw_window_length": 41,
                        "reconstruction": {"type": "spikecv_tfp", "num_bins": 4},
                    }
                }
            },
            "netG": {
                "input": {"strategy": "fusion", "mode": "dual", "raw_ingress_chans": 44},
                "fusion": {
                    "placement": "early",
                    "operator": "pase_residual",
                    "out_chans": 3,
                    "early": {"frame_contract": "collapsed"},
                },
                "output_mode": "restoration",
            },
        },
    )

    monkeypatch.setattr(model.fusion_adapter.operator, "forward", lambda rgb, spike: rgb)

    dummy_flows = [
        torch.zeros(1, 5, 2, 8, 8),
        torch.zeros(1, 5, 2, 4, 4),
        torch.zeros(1, 5, 2, 2, 2),
        torch.zeros(1, 5, 2, 1, 1),
    ]

    monkeypatch.setattr(model, "get_flows", lambda _x, flow_spike=None: (dummy_flows, dummy_flows))
    monkeypatch.setattr(
        model,
        "get_aligned_image_2frames",
        lambda _x, _fb, _ff: [
            torch.zeros(1, 6, model.backbone_in_chans * 4, 8, 8),
            torch.zeros(1, 6, model.backbone_in_chans * 4, 8, 8),
        ],
    )
    monkeypatch.setattr(model, "forward_features", lambda _x, *_args, **_kwargs: torch.zeros_like(_x))

    x = torch.randn(1, 6, 44, 8, 8)
    with torch.no_grad():
        _ = model(x)

    assert model._last_fusion_meta["spike_representation"] == "raw_window"
    assert model._last_fusion_meta["effective_spike_channels"] == 41
    assert model._last_fusion_meta["spike_window_length"] == 41
```

- [ ] **Step 2: Run the focused test-dataset and VRT tests and confirm they fail**

Run:

```bash
python -m pytest tests/data/test_dataset_rgbspike_raw_window.py -k "test_dataset_raw_window" -v
python -m pytest tests/models/test_vrt_fusion_integration.py -k "raw_window" -v
```

Expected: FAIL because `TestDataset` does not parse `spike.representation`, VRT does not know about raw-window legality, and `_last_fusion_meta` does not expose representation metadata.

- [ ] **Step 3: Implement `TestDataset` raw-window support and VRT representation gating**

Update `data/dataset_video_test.py` using the same pattern as the train dataset:

```python
from data.spike_recc import SpikeStream, extract_centered_raw_window, voxelize_spikes_tfp
```

```python
        self.spike_representation = str(spike_cfg.get('representation', 'tfp')).strip().lower()
        if self.spike_representation not in {'tfp', 'raw_window'}:
            raise ValueError(
                f"[TestDataset] Unsupported spike.representation={self.spike_representation!r}."
            )

        raw_window_length_cfg = spike_cfg.get('raw_window_length', None)
        if self.spike_representation == 'raw_window':
            if raw_window_length_cfg is None:
                self.raw_window_length = int(2 * int(self.tfp_half_win_length) + 1)
            else:
                self.raw_window_length = int(raw_window_length_cfg)
            if self.raw_window_length <= 0 or self.raw_window_length % 2 == 0:
                raise ValueError(
                    f"[TestDataset] raw_window_length must be a positive odd integer, got {self.raw_window_length}."
                )
        else:
            self.raw_window_length = None
```

```python
        if self.spike_representation == 'raw_window':
            default_spike_channels = int(self.raw_window_length)
        else:
            default_spike_channels = int(nested_num_bins) if nested_num_bins is not None else 4
        self.spike_channels = int(opt.get('spike_channels', default_spike_channels))
```

```python
        if self.spike_representation == 'raw_window':
            if self.use_precomputed_spike:
                raise ValueError("[TestDataset] raw_window representation does not support precomputed spike artifacts.")
            if self.spike_reconstruction not in {'spikecv_tfp'}:
                raise ValueError(
                    "[TestDataset] raw_window representation requires spike.reconstruction.type='spikecv_tfp'."
                )
```

```python
                if self.spike_representation == 'raw_window':
                    spike_voxel = extract_centered_raw_window(
                        spike_matrix,
                        window_length=self.raw_window_length,
                    )
                elif self.spike_reconstruction in {'middle_tfp', 'middle-tfp'}:
                    spike_frame = self._middle_tfp_reconstructor(spike_matrix)
                    spike_voxel = spike_frame[np.newaxis, ...].astype(np.float32)
```

Update `models/architectures/vrt/vrt.py`:

```python
    def _resolve_fusion_spike_representation(self, opt) -> tuple[str, int | None]:
        datasets_cfg = ((opt or {}).get("datasets", {}) or {})
        candidate_cfgs = []
        for split in ("train", "test"):
            split_cfg = (datasets_cfg.get(split, {}) or {})
            spike_cfg = split_cfg.get("spike", {}) if isinstance(split_cfg.get("spike", {}), dict) else {}
            if spike_cfg:
                candidate_cfgs.append((split, spike_cfg))

        representation = "tfp"
        raw_window_length = None
        for split, spike_cfg in candidate_cfgs:
            split_representation = str(spike_cfg.get("representation", "tfp")).strip().lower()
            if split_representation not in {"tfp", "raw_window"}:
                raise ValueError(f"[VRT] Unsupported datasets.{split}.spike.representation={split_representation!r}.")
            split_raw_window_length = spike_cfg.get("raw_window_length", None)
            if split_representation == "raw_window" and split_raw_window_length is not None:
                split_raw_window_length = int(split_raw_window_length)

            if representation == "tfp":
                representation = split_representation
                raw_window_length = split_raw_window_length
            elif split_representation != representation:
                raise ValueError(
                    f"[VRT] Conflicting spike representations across datasets: {representation!r} vs {split_representation!r}."
                )
            elif split_raw_window_length is not None and raw_window_length not in {None, split_raw_window_length}:
                raise ValueError(
                    f"[VRT] Conflicting raw_window_length values across datasets: {raw_window_length} vs {split_raw_window_length}."
                )
        return representation, raw_window_length
```

Call it near fusion setup:

```python
        self._fusion_spike_representation, self._fusion_raw_window_length = self._resolve_fusion_spike_representation(opt)
```

Add the phase-1 guard:

```python
            if self._fusion_spike_representation == 'raw_window' and normalized_operator_name != 'pase_residual':
                raise ValueError(
                    "fusion spike representation 'raw_window' is only supported with fusion.operator='pase_residual' in phase 1."
                )
```

Augment `_last_fusion_meta` right after `meta = fusion_result["meta"]`:

```python
                meta = {
                    **meta,
                    "spike_representation": self._fusion_spike_representation,
                    "effective_spike_channels": spike_bins,
                }
                if self._fusion_spike_representation == "raw_window":
                    meta["spike_window_length"] = spike_bins
```

- [ ] **Step 4: Re-run the focused test-dataset and VRT tests and confirm they pass**

Run:

```bash
python -m pytest tests/data/test_dataset_rgbspike_raw_window.py -k "test_dataset_raw_window" -v
python -m pytest tests/models/test_vrt_fusion_integration.py -k "raw_window" -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/dataset_video_test.py models/architectures/vrt/vrt.py tests/data/test_dataset_rgbspike_raw_window.py tests/models/test_vrt_fusion_integration.py
git commit -m "feat(fusion): gate raw-window spikes to pase residual"
```

### Task 4: Add a Runnable Raw-Window Ablation Config

**Files:**
- Create: `options/gopro_rgbspike_server_pase_residual_raw_window.json`

- [ ] **Step 1: Create the raw-window ablation config**

Create `options/gopro_rgbspike_server_pase_residual_raw_window.json` by copying `options/gopro_rgbspike_server_pase_residual.json` and applying these targeted changes:

```json
"task": "gopro_raw41_scflow_pase_residual"
```

```json
"netG": {
  "input": {
    "strategy": "fusion",
    "mode": "dual",
    "raw_ingress_chans": 44
  }
}
```

```json
"datasets": {
  "train": {
    "spike": {
      "representation": "raw_window",
      "reconstruction": {
        "type": "spikecv_tfp",
        "num_bins": 4
      },
      "raw_window_length": null
    }
  },
  "test": {
    "spike": {
      "representation": "raw_window",
      "reconstruction": {
        "type": "spikecv_tfp",
        "num_bins": 4
      },
      "raw_window_length": null
    }
  }
}
```

Leave `tfp_half_win_length: 20` unchanged so the derived default window becomes `41`.

- [ ] **Step 2: Validate the config parses and resolves the expected ingress width**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_pase_residual_raw_window.json', is_train=True); print(opt['task']); print(opt['netG']['input']['raw_ingress_chans']); print(opt['datasets']['train']['spike']['representation'])"
```

Expected output:

```text
gopro_raw41_scflow_pase_residual
44
raw_window
```

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_server_pase_residual_raw_window.json
git commit -m "chore(options): add pase residual raw-window ablation config"
```

### Task 5: Run the Focused Regression Slice

**Files:**
- Modify: none
- Test: `tests/data/test_spike_raw_window.py`
- Test: `tests/data/test_dataset_rgbspike_raw_window.py`
- Test: `tests/models/test_vrt_fusion_integration.py`

- [ ] **Step 1: Run the complete raw-window regression slice**

Run:

```bash
python -m pytest tests/data/test_spike_raw_window.py tests/data/test_dataset_rgbspike_raw_window.py tests/models/test_vrt_fusion_integration.py -k "raw_window or pase_residual" -v
```

Expected: PASS

- [ ] **Step 2: Re-run the dedicated config parser check**

Run:

```bash
python -c "from utils import utils_option as option; opt = option.parse('options/gopro_rgbspike_server_pase_residual_raw_window.json', is_train=True); print(opt['netG']['fusion']['operator']); print(opt['datasets']['test']['spike']['representation'])"
```

Expected output:

```text
pase_residual
raw_window
```

- [ ] **Step 3: Commit the verification checkpoint**

```bash
git commit --allow-empty -m "test(fusion): verify pase residual raw-window wiring"
```

## Self-Review

- Spec coverage: the plan covers the helper extraction path, train/test dataset parsing, `pase_residual`-only gating in VRT, metadata surfacing, config wiring, and fair regression checks.
- Placeholder scan: there are no `TODO`/`TBD` placeholders; every task includes exact file paths, concrete tests, commands, and expected outcomes.
- Type consistency: the plan uses `spike.representation`, `raw_window_length`, `spike_representation`, `effective_spike_channels`, and `extract_centered_raw_window()` consistently across helper, dataset, VRT, and config tasks.
