# SCFlow Strict Semantic Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate SCFlow into S-VRT with strict spike semantics so optical flow consumes true 25-slice spike sequences (`L_flow_spike`) rather than restoration channels.

**Architecture:** Keep restoration input (`L`) unchanged, add a dedicated flow spike tensor (`L_flow_spike`) produced from offline `encoding25_dt{dt}` artifacts, and route SCFlow branch in VRT to that tensor only. Enforce fail-fast contracts in config/dataset/model/wrapper and lock behavior with targeted tests.

**Tech Stack:** Python, PyTorch, pytest, existing S-VRT dataset/model stack, Cybertron server terminal execution.

---

## File Structure Map

- Create: `data/spike_recc/encoding25.py`
  - Utility to build centered fixed-length 25-slice windows and validate encoded spike sequences.
- Create: `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py`
  - Offline encoder that writes `<clip>/encoding25_dt{dt}/<frame>.npy` artifacts.
- Modify: `data/dataset_video_train_rgbspike.py`
  - Optional SCFlow flow-input loading path and `L_flow_spike` output contract.
- Modify: `data/dataset_video_test.py`
  - Test split support for `L_flow_spike` with same contract.
- Modify: `models/model_plain.py`
  - Ingress contract: require `L_flow_spike` when `optical_flow.module=scflow`; pass it to netG.
- Modify: `models/architectures/vrt/vrt.py`
  - Accept optional `flow_spike` tensor; SCFlow branch consumes only this tensor.
- Modify: `models/optical_flow/scflow/wrapper.py`
  - Strict shape validation (`ndim=4`, `channels=25`) with explicit error messages.
- Modify: `options/gopro_rgbspike_local.json`
  - Add `datasets.train/test.spike_flow` config for strict semantics.
- Modify: `options/gopro_rgbspike_server.json`
  - Same as local config, keep server default optical flow backend choice as-is unless explicitly switching.
- Create: `tests/models/test_optical_flow_scflow_contract.py`
  - Contract-focused tests for wrapper, model ingress, and VRT flow routing.

---

### Task 1: Add Encoding25 Utility Module

**Files:**
- Create: `data/spike_recc/encoding25.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write the failing test for basic encoding25 contract helper**

```python
from data.spike_recc.encoding25 import validate_encoding25_tensor
import numpy as np
import pytest


def test_validate_encoding25_tensor_rejects_non_25_channels():
    bad = np.zeros((11, 16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 25"):
        validate_encoding25_tensor(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py::test_validate_encoding25_tensor_rejects_non_25_channels -v`
Expected: FAIL with `ImportError`/missing function before implementation.

- [ ] **Step 3: Write minimal implementation in `encoding25.py`**

```python
import numpy as np


def validate_encoding25_tensor(tensor: np.ndarray) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"encoding25 tensor must be [25,H,W], got ndim={tensor.ndim}")
    if tensor.shape[0] != 25:
        raise ValueError(f"encoding25 tensor expected 25 channels, got {tensor.shape[0]}")
```

- [ ] **Step 4: Add centered-window helper used by offline encoder**

```python
def build_centered_window(spike_matrix: np.ndarray, center: int, length: int = 25) -> np.ndarray:
    if length != 25:
        raise ValueError("SCFlow strict mode requires length=25")
    half = length // 2
    st = center - half
    ed = center + half + 1
    if st < 0 or ed > spike_matrix.shape[0]:
        raise ValueError(f"center={center} out of valid range for length={length}")
    window = spike_matrix[st:ed].astype(np.float32)
    validate_encoding25_tensor(window)
    return window
```

- [ ] **Step 5: Run targeted test to verify pass**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py::test_validate_encoding25_tensor_rejects_non_25_channels -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add data/spike_recc/encoding25.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(spike): add encoding25 contract utilities"
```

---

### Task 2: Add Offline Encoding Script

**Files:**
- Create: `scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py`
- Modify: `scripts/data_preparation/README.md`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing test for output path policy helper**

```python
from data.spike_recc.encoding25 import build_output_dir


def test_build_output_dir_uses_dataset_local_convention(tmp_path):
    clip_dir = tmp_path / "GOPR0001"
    out = build_output_dir(clip_dir, dt=10)
    assert out.name == "encoding25_dt10"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py::test_build_output_dir_uses_dataset_local_convention -v`
Expected: FAIL due to missing script helper.

- [ ] **Step 3: Implement offline encoder script with deterministic policy**

```python
def encode_one_frame(spike_matrix: np.ndarray, frame_index: int, clip_start_frame: int, dt: int, center_offset: int = 40) -> np.ndarray:
    local_index = frame_index - clip_start_frame
    center = center_offset + local_index * dt
    return build_centered_window(spike_matrix, center=center, length=25)
```

- [ ] **Step 4: Implement CLI entry and safe execution options**

```python
parser.add_argument("--spike-root", required=True)
parser.add_argument("--dt", type=int, default=10)
parser.add_argument("--center-offset", type=int, default=40)
parser.add_argument("--edge-margin", type=int, default=40)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--overwrite", action="store_true")
```

- [ ] **Step 5: Update data-preparation README with command examples**

```bash
python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /path/to/GOPRO_Large_spike_seq/train --dt 10
```

- [ ] **Step 6: Run targeted tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k output_dir -v`
Expected: PASS for output-dir policy test.

- [ ] **Step 7: Commit**

```bash
git add scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py scripts/data_preparation/README.md tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(data): add offline scflow encoding25 preparation script"
```

---

### Task 3: Dataset Contract for `L_flow_spike`

**Files:**
- Modify: `data/dataset_video_train_rgbspike.py`
- Modify: `data/dataset_video_test.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing test for SCFlow dataset output contract**

```python
def test_dataset_requires_l_flow_spike_when_scflow_enabled(tmp_path):
    from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike

    opt = make_minimal_rgbspike_train_opt(tmp_path)
    opt["spike_flow"] = {"representation": "encoding25", "dt": 10, "root": "auto"}
    ds = TrainDatasetRGBSpike(opt)

    with pytest.raises(ValueError, match="Missing encoding25 artifact"):
        _ = ds[0]
```

- [ ] **Step 2: Run failing contract test**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k dataset_requires_l_flow_spike -v`
Expected: FAIL before implementation.

- [ ] **Step 3: Add spike-flow config parsing in dataset init**

```python
spike_flow_cfg = opt.get("spike_flow", {}) if isinstance(opt.get("spike_flow", {}), dict) else {}
self.spike_flow_representation = str(spike_flow_cfg.get("representation", "")).strip().lower()
self.spike_flow_dt = int(spike_flow_cfg.get("dt", 10))
self.spike_flow_root = spike_flow_cfg.get("root", "auto")
```

- [ ] **Step 4: Add encoded flow loader with strict validation**

```python
def _load_encoded_flow_spike(self, clip_name, frame_idx):
    flow_root = self.spike_root if self.spike_flow_root == "auto" else Path(self.spike_flow_root)
    path = flow_root / clip_name / f"encoding25_dt{self.spike_flow_dt}" / f"{frame_idx:{self.filename_tmpl}}.npy"
    if not path.exists():
        raise ValueError(f"Missing encoding25 artifact: {path}")
    arr = np.load(path).astype(np.float32)
    validate_encoding25_tensor(arr)
    return arr
```

- [ ] **Step 5: Emit `L_flow_spike` in sample dict when configured**

```python
if self.spike_flow_representation == "encoding25":
    sample["L_flow_spike"] = flow_spike_tensor  # [T,25,H,W]
```

- [ ] **Step 6: Mirror contract in test dataset class**

Run the same parse/load path in `data/dataset_video_test.py` so train/test behavior is aligned.

- [ ] **Step 7: Run dataset contract tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k dataset -v`
Expected: PASS for strict artifact and shape checks.

- [ ] **Step 8: Commit**

```bash
git add data/dataset_video_train_rgbspike.py data/dataset_video_test.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(dataset): add strict L_flow_spike contract for scflow"
```

---

### Task 4: Model Ingress and VRT Routing

**Files:**
- Modify: `models/model_plain.py`
- Modify: `models/architectures/vrt/vrt.py`
- Modify: `models/model_vrt.py` (if forwarding wrapper path requires update)
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing test for model ingress requirement**

```python
def test_model_plain_requires_l_flow_spike_for_scflow(monkeypatch):
    from models.model_plain import ModelPlain

    opt = make_minimal_model_opt(flow_module="scflow")
    model = ModelPlain(opt)
    data = {"L": torch.randn(1, 6, 11, 16, 16), "H": torch.randn(1, 6, 3, 16, 16)}

    with pytest.raises(ValueError, match="module=scflow requires data\['L_flow_spike'\]"):
        model.feed_data(data)
```

- [ ] **Step 2: Run failing model ingress test**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k model_plain_requires_l_flow_spike -v`
Expected: FAIL before code change.

- [ ] **Step 3: Add helper in `ModelPlain` to detect flow backend**

```python
def _flow_module_name(self):
    return str(self.opt.get("netG", {}).get("optical_flow", {}).get("module", "spynet")).strip().lower()
```

- [ ] **Step 4: Require/cache `L_flow_spike` for SCFlow in `feed_data`**

```python
if self._flow_module_name() == "scflow":
    if "L_flow_spike" not in data:
        raise ValueError("module=scflow requires data['L_flow_spike'] with shape [B,T,25,H,W]")
    self.L_flow_spike = data["L_flow_spike"].to(self.device)
else:
    self.L_flow_spike = None
```

- [ ] **Step 5: Forward `flow_spike` into netG path**

```python
self.E = self.netG(self.L, flow_spike=self.L_flow_spike)
```

- [ ] **Step 6: Update VRT forward signature and flow path**

```python
def forward(self, x, flow_spike=None):
    # x remains restoration input; flow_spike is only for SCFlow optical-flow branch.
    flows_backward, flows_forward = self.get_flows(x, flow_spike=flow_spike)
```

```python
def get_flow_2frames(self, x, flow_spike=None):
    if getattr(self.spynet, "input_type", "rgb") == "spike":
        if flow_spike is None:
            raise ValueError("SCFlow requires flow_spike input [B,T,25,H,W]")
        x_flow = flow_spike
    else:
        x_flow = self.extract_rgb(x)
```

- [ ] **Step 7: Run routing tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k "model_plain or vrt" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add models/model_plain.py models/architectures/vrt/vrt.py models/model_vrt.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "feat(vrt): route scflow to dedicated flow_spike tensor"
```

---

### Task 5: SCFlow Wrapper Contract Hardening

**Files:**
- Modify: `models/optical_flow/scflow/wrapper.py`
- Test: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Write failing wrapper shape-validation tests**

```python
import torch
import pytest
from models.optical_flow.scflow.wrapper import SCFlowWrapper


def test_scflow_wrapper_rejects_non_25_channels():
    of = SCFlowWrapper(checkpoint=None, device="cpu")
    with pytest.raises(ValueError, match="expected.*25"):
        of(torch.randn(1, 11, 16, 16), torch.randn(1, 11, 16, 16))
```

- [ ] **Step 2: Run failing wrapper test**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k wrapper_rejects_non_25 -v`
Expected: FAIL before wrapper validation.

- [ ] **Step 3: Add explicit validation in wrapper forward**

```python
def _validate_spike_pair(self, spk1, spk2):
    for name, t in (("spk1", spk1), ("spk2", spk2)):
        if t.ndim != 4:
            raise ValueError(f"SCFlow expects {name} ndim=4 [B,25,H,W], got {tuple(t.shape)}")
        if t.size(1) != 25:
            raise ValueError(f"SCFlow expects {name} channels=25, got {t.size(1)} with shape {tuple(t.shape)}")
```

- [ ] **Step 4: Run wrapper tests**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -k wrapper -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/optical_flow/scflow/wrapper.py tests/models/test_optical_flow_scflow_contract.py
git commit -m "fix(scflow): enforce strict 25-channel input contract"
```

---

### Task 6: Config Wiring and Contract Tests

**Files:**
- Modify: `options/gopro_rgbspike_local.json`
- Modify: `options/gopro_rgbspike_server.json`
- Create/Modify: `tests/models/test_optical_flow_scflow_contract.py`

- [ ] **Step 1: Add spike_flow config blocks to train/test datasets**

```json
"spike_flow": {
  "representation": "encoding25",
  "dt": 10,
  "root": "auto"
}
```

- [ ] **Step 2: Add config-contract test**

```python
def test_scflow_requires_encoding25_representation(tmp_path):
    opt = make_minimal_rgbspike_train_opt(tmp_path)
    opt["netG"]["optical_flow"] = {"module": "scflow", "checkpoint": None, "params": {}}
    opt["spike_flow"] = {"representation": "tfp", "dt": 10, "root": "auto"}

    with pytest.raises(ValueError, match="representation.*encoding25"):
        _ = build_dataset_with_opt(opt)
```

- [ ] **Step 3: Run full SCFlow contract test file**

Run: `pytest tests/models/test_optical_flow_scflow_contract.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add options/gopro_rgbspike_local.json options/gopro_rgbspike_server.json tests/models/test_optical_flow_scflow_contract.py
git commit -m "test(config): lock scflow strict semantic contracts"
```

---

### Task 7: Server Verification via Cybertron

**Files:**
- No source changes required (execution/verification task)

- [ ] **Step 1: Sync branch to server workspace**

Run (Cybertron terminal):
```bash
git branch --show-current
git status --short
```
Expected: Correct branch and clean/known diff.

- [ ] **Step 2: Generate encoding25 artifacts on target split**

Run:
```bash
python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root <SERVER_SPIKE_SPLIT_ROOT> --dt 10
```
Expected: Summary includes generated/skipped counts and zero fatal errors.

- [ ] **Step 3: Run contract tests on server**

Run:
```bash
pytest tests/models/test_optical_flow_scflow_contract.py -v
```
Expected: PASS.

- [ ] **Step 4: Run one integration smoke using SCFlow config**

Run:
```bash
pytest tests/models/test_vrt_integration.py -k scflow -v
```
If no existing SCFlow marker exists, run a dedicated new smoke test added in previous tasks.
Expected: PASS with no channel-shape mismatch errors.

- [ ] **Step 5: Collect artifacts/log snippets and summarize**

Capture:
- encoding generation summary lines
- pytest pass summary
- one forward shape log

- [ ] **Step 6: Commit verification notes if needed**

```bash
git add docs/ tests/
git commit -m "test(scflow): verify strict-semantic path on server"
```

---

## Spec Coverage Check

- Strict 25-slice SCFlow semantics: covered in Tasks 1, 3, 4, 5.
- Offline encoding and artifact placement: covered in Task 2.
- Config contracts and explicit errors: covered in Tasks 3 and 6.
- Dedicated `L_flow_spike` data path: covered in Tasks 3 and 4.
- Server-only execution/verification requirement: covered in Task 7.

## Placeholder Scan

- No `TODO`, `TBD`, or deferred implementation placeholders present.
- All tasks include concrete file paths and runnable commands.

## Type/Interface Consistency Check

- `L_flow_spike` shape consistently specified as `[B,T,25,H,W]` at model boundary.
- Wrapper-level inputs consistently specified as `[B,25,H,W]` after frame-pair reshape.
- SCFlow strict-gating is consistently keyed off `optical_flow.module == "scflow"`.

