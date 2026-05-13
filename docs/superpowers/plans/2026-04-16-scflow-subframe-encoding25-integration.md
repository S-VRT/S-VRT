# SCFlow Subframe Encoding25 Integration with Early Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate per-subframe encoding25 windows so `flow_spike` has shape `[frames*S, 25, H, W]`, matching the early-fusion-expanded temporal axis, enabling SCFlow optical flow at sub-frame granularity.

**Architecture:** Each `.dat` file (one per video frame, T_raw=56 or 88) gets S evenly-spaced 25-bin windows extracted via `linspace` over its valid center range `[12, T_raw-13]`. The dataset loads `[S, 25, H, W]` per frame, crops/augments each sub-window independently, then flattens to `[frames*S, 25, H, W]`. VRT's existing `get_flow_2frames()` check `flow_spike.size(1) == x.size(1)` passes naturally since both axes are `frames*S`. SCFlow weights stay frozen.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing S-VRT dataset/model stack

**Specs:** Prior plans `2026-04-13-svrt-scflow-strict-semantic-implementation.md`, `2026-04-14-early-fusion-temporal-expansion-implementation.md`

**Key constraint:** `spike_flow.subframes` (S) MUST equal `spike_channels` (early fusion expansion factor). Default S=4.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `data/spike_recc/encoding25.py` | Modify | Add `compute_subframe_centers()`, `validate_subframes_tensor()`, `build_output_dir_subframes()` |
| `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py` | Modify | Add `--subframes` CLI arg, `build_scflow_subframe_windows()`, update `process_clip()` to generate `[S, 25, H, W]` |
| `data/dataset_video_train_rgbspike.py` | Modify | Parse `spike_flow.subframes`, load `[S, 25, H, W]`, crop/augment per sub-window, flatten to `[frames*S, 25, H, W]` |
| `models/model_plain.py` | Modify | Update comment on `L_flow_spike` time dim semantics |
| `models/architectures/vrt/vrt.py` | Modify | Improve `get_flow_2frames()` error message for subframes/spike_channels mismatch |
| `options/gopro_rgbspike_local.json` | Modify | Add `spike_flow.subframes: 4` |
| `options/gopro_rgbspike_server.json` | Modify | Add `spike_flow.subframes: 4` |
| `tests/models/test_optical_flow_scflow_contract.py` | Modify | Add tests for subframe center computation, subframe tensor validation, subframe window generation |

---

### Task 1: Add Subframe Center Computation to encoding25.py

**Files:**
- Modify: `data/spike_recc/encoding25.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing tests for `compute_subframe_centers()`**

Add to `tests/models/test_optical_flow_scflow_contract.py`:

```python
from data.spike_recc.encoding25 import (
    compute_subframe_centers,
    validate_subframes_tensor,
    build_output_dir_subframes,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "subframe_centers" -v`
Expected: FAIL with `ImportError: cannot import name 'compute_subframe_centers'`

- [ ] **Step 3: Implement `compute_subframe_centers()` in encoding25.py**

Add to `data/spike_recc/encoding25.py`:

```python
from typing import List


def compute_subframe_centers(
    t_raw: int,
    num_subframes: int,
    margin: int = WINDOW_HALF,
) -> List[int]:
    """Compute S evenly-spaced sub-centers within a single .dat file.

    Each center allows a full 25-wide window: [center-12, center+12].
    Centers are based on the .dat's own T_raw, not a global offset.

    Args:
        t_raw: Raw spike time length of the .dat file.
        num_subframes: Number of sub-windows (S). Must be >= 1.
        margin: Minimum distance from center to array boundary (default 12).

    Returns:
        List of S integer center indices, sorted ascending.
    """
    if t_raw < WINDOW_LENGTH:
        raise ValueError(
            f"t_raw={t_raw} too short for a {WINDOW_LENGTH}-wide window"
        )
    if num_subframes < 1:
        raise ValueError(f"num_subframes must be >= 1, got {num_subframes}")

    lo = margin
    hi = t_raw - margin - 1

    if num_subframes == 1:
        return [(lo + hi) // 2]

    raw_centers = np.linspace(lo, hi, num_subframes)
    return [int(round(c)) for c in raw_centers]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "subframe_centers" -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write failing tests for `validate_subframes_tensor()` and `build_output_dir_subframes()`**

Add to test file:

```python
@pytest.mark.unit
def test_validate_subframes_tensor_accepts_valid():
    arr = np.zeros((4, 25, 8, 8), dtype=np.float32)
    validate_subframes_tensor(arr, num_subframes=4)  # no error


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
```

- [ ] **Step 6: Implement `validate_subframes_tensor()` and `build_output_dir_subframes()`**

Add to `data/spike_recc/encoding25.py`:

```python
def validate_subframes_tensor(tensor: np.ndarray, num_subframes: int) -> None:
    """Validate shape [S, 25, H, W] for multi-subframe encoding."""
    if tensor.ndim != 4:
        raise ValueError(
            f"subframes tensor must be ndim=4 [S,25,H,W], got ndim={tensor.ndim}"
        )
    if tensor.shape[0] != num_subframes:
        raise ValueError(
            f"subframes tensor expected {num_subframes} subframes, got {tensor.shape[0]}"
        )
    if tensor.shape[1] != WINDOW_LENGTH:
        raise ValueError(
            f"subframes tensor expected {WINDOW_LENGTH} channels, got {tensor.shape[1]}"
        )


def build_output_dir_subframes(clip_dir: Path, dt: int, num_subframes: int) -> Path:
    """Return artifact directory with subframe suffix when S > 1."""
    if num_subframes <= 1:
        return build_output_dir(clip_dir, dt)
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    return Path(clip_dir) / f"encoding25_dt{int(dt)}_s{int(num_subframes)}"
```

- [ ] **Step 7: Run all new tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "subframe" -v`
Expected: All 11 tests PASS

- [ ] **Step 8: Commit**

```bash
git add data/spike_recc/encoding25.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(encoding25): add subframe center computation and validation for early fusion integration"
```

---

### Task 2: Update Preparation Script for Subframe Generation

**Files:**
- Modify: `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing test for `build_scflow_subframe_windows()`**

Add to test file:

```python
from scripts.data_preparation.spike_flow.prepare_scflow_encoding25 import (
    build_scflow_subframe_windows,
)


@pytest.mark.unit
def test_build_scflow_subframe_windows_shape_t56_s4():
    spike = np.random.rand(56, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=4)
    assert result.shape == (4, 25, 8, 8)
    assert result.dtype == np.float32


@pytest.mark.unit
def test_build_scflow_subframe_windows_shape_t88_s4():
    spike = np.random.rand(88, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=4)
    assert result.shape == (4, 25, 8, 8)


@pytest.mark.unit
def test_build_scflow_subframe_windows_s1_backward_compat():
    spike = np.random.rand(56, 8, 8).astype(np.float32)
    result = build_scflow_subframe_windows(spike, num_subframes=1)
    assert result.shape == (1, 25, 8, 8)


@pytest.mark.unit
def test_build_scflow_subframe_windows_rejects_short():
    spike = np.random.rand(20, 8, 8).astype(np.float32)
    with pytest.raises(ValueError):
        build_scflow_subframe_windows(spike, num_subframes=4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "build_scflow_subframe" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `build_scflow_subframe_windows()`**

Add to `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py`:

```python
from data.spike_recc.encoding25 import (
    build_output_dir_subframes,
    compute_subframe_centers,
    build_centered_window,
    validate_subframes_tensor,
    validate_encoding25_tensor,
)


def build_scflow_subframe_windows(
    spike_matrix: np.ndarray,
    num_subframes: int,
) -> np.ndarray:
    """Extract S sub-windows from a spike matrix based on its own T_raw.

    Args:
        spike_matrix: [T_raw, H, W] raw spike data from one .dat file.
        num_subframes: Number of sub-windows (S).

    Returns:
        np.ndarray of shape [S, 25, H, W], float32.
    """
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be [T,H,W], got shape={tuple(spike_matrix.shape)}")

    centers = compute_subframe_centers(
        t_raw=spike_matrix.shape[0],
        num_subframes=num_subframes,
    )
    windows = [build_centered_window(spike_matrix, center) for center in centers]
    result = np.stack(windows, axis=0)
    validate_subframes_tensor(result, num_subframes)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "build_scflow_subframe" -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Update `process_clip()` to use subframes**

Modify `process_clip()` in `prepare_scflow_encoding25.py`:
- Add `num_subframes: int` parameter
- Replace `build_output_dir(clip_dir, dt=dt)` with `build_output_dir_subframes(clip_dir, dt=dt, num_subframes=num_subframes)`
- Replace `build_scflow_window(spike_matrix, short_policy)` call with `build_scflow_subframe_windows(spike_matrix, num_subframes)` when `num_subframes > 1`
- For `num_subframes == 1`, keep existing `build_scflow_window()` behavior for backward compat

- [ ] **Step 6: Thread `num_subframes` through `_process_clip_worker()`, `process_all_clips()`, and `main()`**

Add `--subframes` CLI argument to `main()`:

```python
parser.add_argument("--subframes", type=int, default=4,
                    help="Number of sub-windows per .dat file (S). Default: 4.")
```

Thread through all orchestration functions.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -v`
Expected: All existing + new tests PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(prepare): add subframe encoding25 generation with --subframes CLI arg"
```

---

### Task 3: Update Dataset to Load and Expand Subframe Flow Spikes

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing test for subframes config parsing**

```python
@pytest.mark.unit
def test_dataset_parses_spike_flow_subframes():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds._parse_spike_flow_config(
        {"spike_flow": {"representation": "encoding25", "dt": 10, "root": "auto", "subframes": 4}},
        optical_flow_module="scflow",
    )
    assert ds.spike_flow_subframes == 4


@pytest.mark.unit
def test_dataset_spike_flow_subframes_default_1():
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds._parse_spike_flow_config(
        {"spike_flow": {"representation": "encoding25", "dt": 10, "root": "auto"}},
        optical_flow_module="scflow",
    )
    assert ds.spike_flow_subframes == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "spike_flow_subframes" -v`
Expected: FAIL with `AttributeError: 'TrainDatasetRGBSpike' object has no attribute 'spike_flow_subframes'`

- [ ] **Step 3: Add `spike_flow_subframes` parsing to `_parse_spike_flow_config()`**

In `data/dataset_video_train_rgbspike.py`, add after line 507 (`self.spike_flow_root = ...`):

```python
self.spike_flow_subframes = int(spike_flow_cfg.get('subframes', 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "spike_flow_subframes" -v`
Expected: PASS

- [ ] **Step 5: Write failing test for subframe loading**

```python
@pytest.mark.unit
def test_dataset_load_subframe_flow_spike(tmp_path):
    ds = TrainDatasetRGBSpike.__new__(TrainDatasetRGBSpike)
    ds.spike_root = tmp_path
    ds.spike_flow_root = "auto"
    ds.spike_flow_dt = 10
    ds.spike_flow_subframes = 4
    ds.filename_tmpl = "06d"

    # Create artifact
    clip_dir = tmp_path / "clip_a" / "encoding25_dt10_s4"
    clip_dir.mkdir(parents=True)
    arr = np.random.rand(4, 25, 8, 8).astype(np.float32)
    np.save(clip_dir / "000001.npy", arr)

    result = ds._load_encoded_flow_spike("clip_a", 1)
    assert result.shape == (4, 25, 8, 8)
```

- [ ] **Step 6: Update `_load_encoded_flow_spike()` for subframes**

Replace the method in `data/dataset_video_train_rgbspike.py`:

```python
def _load_encoded_flow_spike(self, clip_name, frame_idx):
    flow_root = self.spike_root if str(self.spike_flow_root).strip().lower() == 'auto' else Path(self.spike_flow_root)
    frame_name = f'{frame_idx:{self.filename_tmpl}}'

    if self.spike_flow_subframes > 1:
        dir_name = f'encoding25_dt{self.spike_flow_dt}_s{self.spike_flow_subframes}'
    else:
        dir_name = f'encoding25_dt{self.spike_flow_dt}'

    path = flow_root / clip_name / dir_name / f'{frame_name}.npy'
    if not path.exists():
        raise ValueError(
            f"Missing encoding25 artifact: {path}. "
            "Run scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py first."
        )
    arr = np.load(path).astype(np.float32)

    if self.spike_flow_subframes > 1:
        from data.spike_recc.encoding25 import validate_subframes_tensor
        validate_subframes_tensor(arr, self.spike_flow_subframes)
    else:
        validate_encoding25_tensor(arr)
    return arr
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "load_subframe" -v`
Expected: PASS

- [ ] **Step 8: Update `__getitem__()` crop/resize for subframes**

In `__getitem__()`, replace the flow spike crop/resize block (lines 343-347):

```python
flow_spikes_resized = []
if self.use_encoding25_flow:
    for flow_spike in flow_spikes:
        if self.spike_flow_subframes > 1:
            sub_resized = []
            for s_idx in range(self.spike_flow_subframes):
                sub_window = flow_spike[s_idx]  # [25, H, W]
                validate_encoding25_tensor(sub_window)
                sub_resized.append(_crop_resize_chw(sub_window, 25, "Flow spike sub-window"))
            flow_spikes_resized.append(np.stack(sub_resized, axis=0))  # [S, 25, H, W]
        else:
            validate_encoding25_tensor(flow_spike)
            flow_spikes_resized.append(_crop_resize_chw(flow_spike, 25, "Flow spike"))
```

- [ ] **Step 9: Update `__getitem__()` augmentation and tensor assembly for subframes**

Replace the augmentation block (lines 365-404):

```python
# Build flow HWC list for augmentation
if self.use_encoding25_flow and self.spike_flow_subframes > 1:
    flow_hwc_list = []
    for arr_s25hw in flow_spikes_resized:
        for s_idx in range(self.spike_flow_subframes):
            flow_hwc_list.append(np.transpose(arr_s25hw[s_idx], (1, 2, 0)))
elif self.use_encoding25_flow:
    flow_hwc_list = [np.transpose(arr, (1, 2, 0)) for arr in flow_spikes_resized]
else:
    flow_hwc_list = []

merge_list = img_lqs_with_spike + flow_hwc_list + img_gts
img_results = utils_video.augment(merge_list, self.opt['use_hflip'], self.opt['use_rot'])

img_results = utils_video.img2tensor(img_results, bgr2rgb=False)
lq_count = len(img_lqs_with_spike)
flow_count = len(flow_hwc_list)
img_lqs = torch.stack(img_results[:lq_count], dim=0)
flow_tensor = None
if flow_count > 0:
    flow_tensor = torch.stack(img_results[lq_count:lq_count + flow_count], dim=0)
img_gts = torch.stack(img_results[lq_count + flow_count:], dim=0)
```

Update the final flow_tensor validation:

```python
if self.use_encoding25_flow:
    if flow_tensor is None:
        raise ValueError("SCFlow strict mode expected non-empty flow_tensor.")
    if flow_tensor.ndim != 4 or flow_tensor.size(1) != 25:
        raise ValueError(
            f"Expected L_flow_spike shape [frames*S,25,H,W], got {tuple(flow_tensor.shape)}"
        )
    sample['L_flow_spike'] = flow_tensor.float()
```

- [ ] **Step 10: Add subframes/spike_channels consistency assertion in `__init__()`**

After `_parse_spike_flow_config()` call and `spike_channels` assignment, add:

```python
if self.use_encoding25_flow and self.spike_flow_subframes > 1:
    if self.spike_flow_subframes != self.spike_channels:
        raise ValueError(
            f"spike_flow.subframes ({self.spike_flow_subframes}) must equal "
            f"spike_channels ({self.spike_channels}) for early-fusion temporal "
            f"axis alignment."
        )
```

- [ ] **Step 11: Run existing + new tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -v`
Expected: All PASS

- [ ] **Step 12: Commit**

```bash
git add data/dataset_video_train_rgbspike.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(dataset): load and expand subframe encoding25 flow spikes for early fusion"
```

---

### Task 4: Update model_plain.py and vrt.py

**Files:**
- Modify: `models/model_plain.py:299-306`
- Modify: `models/architectures/vrt/vrt.py:702-705`

- [ ] **Step 1: Update comment in model_plain.py `feed_data()`**

At line 299-306, update the comment/docstring:

```python
if self._flow_module_name() == 'scflow':
    if 'L_flow_spike' not in data:
        raise ValueError(
            "module=scflow requires data['L_flow_spike'] with shape [B,T,25,H,W] "
            "where T = frames (subframes=1) or frames*subframes (subframes>1)"
        )
    self.L_flow_spike = data['L_flow_spike'].to(self.device)
    if self.L_flow_spike.ndim != 5 or self.L_flow_spike.size(2) != 25:
        raise ValueError(
            f"module=scflow requires L_flow_spike shape [B,T,25,H,W], got {tuple(self.L_flow_spike.shape)}"
        )
```

- [ ] **Step 2: Improve error message in vrt.py `get_flow_2frames()`**

At line 702-705, replace the error message:

```python
if flow_spike.size(0) != b or flow_spike.size(1) != n:
    raise ValueError(
        f"SCFlow flow_spike temporal dim mismatch: "
        f"flow_spike.shape={tuple(flow_spike.shape)}, x.shape={tuple(x.shape)}. "
        f"After early fusion, x has {n} temporal steps. "
        f"Ensure spike_flow.subframes matches spike_channels."
    )
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add models/model_plain.py models/architectures/vrt/vrt.py
git commit -m "docs(model): clarify flow_spike temporal dim semantics for subframe mode"
```

---

### Task 5: Update Config Files

**Files:**
- Modify: `options/gopro_rgbspike_local.json`
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: Add `subframes` to spike_flow config in both files**

In each file's `datasets.train.spike_flow` and `datasets.test.spike_flow` sections, add:

```json
"spike_flow": {
    "representation": "encoding25",
    "dt": 10,
    "root": "auto",
    "subframes": 4
}
```

- [ ] **Step 2: Ensure `spike_channels` matches `subframes`**

Verify that `spike_channels` (or `spike.reconstruction.num_bins`) is set to 4 in both config files. If currently 8, update to 4 to match subframes.

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_local.json options/gopro_rgbspike_server.json
git commit -m "config: add spike_flow.subframes=4 and align spike_channels"
```

---

## Verification

1. Unit tests: `pytest tests/models/test_optical_flow_scflow_contract.py -v`
2. Prep script dry-run: `python -m scripts.data_preparation.spike_flow.prepare_scflow_encoding25 --spike-root <path> --subframes 4 --dry-run`
3. Shape trace: construct mock data, verify `flow_spike` flows from dataset `[frames*4, 25, H, W]` → model_plain `[B, frames*4, 25, H, W]` → VRT `get_flow_2frames()` passes `flow_spike.size(1) == x.size(1)` after early fusion
