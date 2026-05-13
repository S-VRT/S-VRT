# S-VRT Test Suite Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modernize `tests/` into a stable full-layer test system (unit/integration/smoke/e2e) aligned with current dual-input/fusion contracts and server dataset paths.

**Architecture:** Keep existing test files where practical, add strict marker governance, and refactor legacy script-style tests into assertion-based pytest tests. Introduce a server-option-driven e2e path using `options/gopro_rgbspike_server.json` with explicit conditional skip behavior when platform prerequisites are missing.

**Tech Stack:** Python, pytest, PyTorch, existing S-VRT test modules, `options/gopro_rgbspike_server.json`.

---

## File Structure Map

### Modify
1. `pytest.ini`
2. `tests/conftest.py`
3. `tests/run_tests.py`
4. `tests/README.md`
5. `tests/models/test_vrt_smoke.py`
6. `tests/models/test_smoke_vrt.py`
7. `tests/smoke/test_searaft_integration.py`

### Create
1. `tests/e2e/test_gopro_rgbspike_server_e2e.py`

---

### Task 1: Marker Governance and Shared Test Infrastructure

**Files:**
- Modify: `pytest.ini`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write failing marker test for e2e marker strictness**

```python
# tests/test_markers_contract.py
import pytest

@pytest.mark.e2e
def test_e2e_marker_registered():
    assert True
```

- [ ] **Step 2: Run marker contract test to confirm failure before config update**

Run: `python -m pytest tests/test_markers_contract.py -v`
Expected: FAIL with strict-marker error for unregistered `e2e` marker.

- [ ] **Step 3: Add unified marker definitions in `pytest.ini`**

```ini
markers =
    smoke: Quick smoke tests
    integration: Integration tests
    unit: Unit tests
    e2e: End-to-end tests on real platform paths
    slow: Slow running tests
    optical_flow: Optical flow related tests
    vrt: VRT model related tests
    models: Model related tests
```

- [ ] **Step 4: Extend `tests/conftest.py` with server option fixture + skip helpers**

```python
import json
from pathlib import Path

@pytest.fixture(scope="session")
def server_option_path():
    return Path(project_root) / "options" / "gopro_rgbspike_server.json"

@pytest.fixture(scope="session")
def server_option(server_option_path):
    with server_option_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_paths_from_opt(opt_dict):
    train = opt_dict.get("datasets", {}).get("train", {})
    test = opt_dict.get("datasets", {}).get("test", {})
    return {
        "train_gt": train.get("dataroot_gt"),
        "train_lq": train.get("dataroot_lq"),
        "train_spike": train.get("dataroot_spike"),
        "test_gt": test.get("dataroot_gt"),
        "test_lq": test.get("dataroot_lq"),
        "test_spike": test.get("dataroot_spike"),
    }


def require_paths_or_skip(paths, reason_prefix="dataset path missing"):
    missing = [p for p in paths if not p or not Path(p).exists()]
    if missing:
        pytest.skip(f"{reason_prefix}: {missing}")
```

- [ ] **Step 5: Re-run marker contract test and conftest sanity import**

Run: `python -m pytest tests/test_markers_contract.py -v`
Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add pytest.ini tests/conftest.py tests/test_markers_contract.py
git commit -m "test(infra): add e2e marker and shared server option fixtures"
```

---

### Task 2: Modernize Legacy Smoke/Script-Style Tests

**Files:**
- Modify: `tests/models/test_vrt_smoke.py`
- Modify: `tests/models/test_smoke_vrt.py`
- Modify: `tests/smoke/test_searaft_integration.py`

- [ ] **Step 1: Add failing expectation for pytest-native assertions (remove script-only pass path)**

```python
def test_smoke_entrypoint_pytest_contract():
    result = run_smoke()
    assert result is None or result is True
```

- [ ] **Step 2: Run only these legacy files and confirm at least one fails pre-refactor**

Run: `python -m pytest tests/models/test_vrt_smoke.py tests/models/test_smoke_vrt.py tests/smoke/test_searaft_integration.py -v`
Expected: FAIL or unstable behavior due to script/main style coupling.

- [ ] **Step 3: Refactor tests to assertion-first pytest style with explicit markers**

```python
import pytest

@pytest.mark.smoke
@pytest.mark.integration
def test_vrt_smoke_forward_contract(...):
    out = model(x)
    assert out.shape == expected_shape

@pytest.mark.smoke
def test_searaft_integration_contract(...):
    if not sea_raft_available:
        pytest.skip("SeaRAFT optional dependency unavailable")
    assert flows is not None
```

- [ ] **Step 4: Remove `if __name__ == "__main__"` driven logic from test behavior path**

```python
# keep optional debug utility functions only, but no pass/fail decision via prints
```

- [ ] **Step 5: Re-run modernized smoke/integration subset**

Run: `python -m pytest tests/models/test_vrt_smoke.py tests/models/test_smoke_vrt.py tests/smoke/test_searaft_integration.py -v`
Expected: PASS or explicit SKIP with clear reason; no script-style false positives.

- [ ] **Step 6: Commit Task 2**

```bash
git add tests/models/test_vrt_smoke.py tests/models/test_smoke_vrt.py tests/smoke/test_searaft_integration.py
git commit -m "test(smoke): refactor legacy script-style tests to pytest assertions"
```

---

### Task 3: Add Server-Option-Driven E2E Test

**Files:**
- Create: `tests/e2e/test_gopro_rgbspike_server_e2e.py`
- Modify: `tests/conftest.py` (reuse fixtures/helpers from Task 1)

- [ ] **Step 1: Write failing e2e test skeleton (RED)**

```python
import pytest

@pytest.mark.e2e
def test_server_option_e2e_minimal_forward(server_option):
    assert "datasets" in server_option
    raise AssertionError("RED: implement minimal e2e flow")
```

- [ ] **Step 2: Run e2e skeleton to confirm intentional failure**

Run: `python -m pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -v`
Expected: FAIL with `RED: implement minimal e2e flow`.

- [ ] **Step 3: Implement minimal real-data e2e flow**

```python
import copy
import pytest
import torch

from data.dataset_video_train_rgbspike import TrainDatasetRGBSpike
from models.architectures.vrt.vrt import VRT

@pytest.mark.e2e
@pytest.mark.slow
def test_server_option_e2e_minimal_forward(server_option):
    opt = copy.deepcopy(server_option)
    train_opt = opt.get("datasets", {}).get("train", {})
    require_paths_or_skip([
        train_opt.get("dataroot_gt"),
        train_opt.get("dataroot_lq"),
        train_opt.get("dataroot_spike"),
    ], reason_prefix="server dataset not ready")

    dataset = TrainDatasetRGBSpike(train_opt)
    sample = dataset[0]

    x = sample["L"].unsqueeze(0)  # [B,T,C,H,W]
    net_cfg = opt.get("netG", {})
    model = VRT(
        upscale=net_cfg.get("upscale", 1),
        in_chans=net_cfg.get("in_chans", x.size(2)),
        out_chans=3,
        img_size=[x.size(1), x.size(3), x.size(4)],
        window_size=net_cfg.get("window_size", [2, 8, 8]),
        depths=net_cfg.get("depths", [1] * 8),
        indep_reconsts=net_cfg.get("indep_reconsts", []),
        embed_dims=net_cfg.get("embed_dims", [16] * 8),
        num_heads=net_cfg.get("num_heads", [1] * 8),
        pa_frames=net_cfg.get("pa_frames", 2),
        use_flash_attn=False,
        optical_flow=net_cfg.get("optical_flow", {"module": "spynet", "checkpoint": None, "params": {}}),
        opt=opt,
    )

    model.eval()
    with torch.no_grad():
        y = model(x)

    assert y.ndim == 5
    assert y.size(0) == 1
    assert y.size(1) == x.size(1)
    assert y.size(2) == 3
```

- [ ] **Step 4: Run e2e test on compute platform**

Run: `python -m pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -v`
Expected: PASS on available data path, or SKIP with explicit missing-path reason.

- [ ] **Step 5: Commit Task 3**

```bash
git add tests/e2e/test_gopro_rgbspike_server_e2e.py tests/conftest.py
git commit -m "test(e2e): add server-option real-data minimal forward coverage"
```

---

### Task 4: Upgrade Test Runner and Documentation

**Files:**
- Modify: `tests/run_tests.py`
- Modify: `tests/README.md`

- [ ] **Step 1: Add failing CLI behavior test target (manual contract)**

```bash
python tests/run_tests.py --e2e
```
Expected (before change): parser error for unknown argument `--e2e`.

- [ ] **Step 2: Extend runner options for layered execution**

```python
parser.add_argument('--e2e', action='store_true', help='Run only end-to-end tests')
...
elif args.e2e:
    pytest_args.extend(['-m', 'e2e'])
```

- [ ] **Step 3: Align `--smoke` to marker-based execution (not `-k`)**

```python
elif args.smoke:
    pytest_args.extend(['-m', 'smoke'])
```

- [ ] **Step 4: Update `tests/README.md` with authoritative command matrix**

```markdown
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --smoke
python tests/run_tests.py --e2e
python -m pytest -m "unit or integration or smoke" -q
```

- [ ] **Step 5: Verify runner commands**

Run:
1. `python tests/run_tests.py --unit`
2. `python tests/run_tests.py --integration`
3. `python tests/run_tests.py --smoke`
4. `python tests/run_tests.py --e2e`

Expected: commands dispatch to correct markers; e2e yields PASS or explicit SKIP reason.

- [ ] **Step 6: Commit Task 4**

```bash
git add tests/run_tests.py tests/README.md
git commit -m "test(tooling): add e2e runner mode and marker-aligned smoke execution"
```

---

### Task 5: Full-Layer Verification and Final Acceptance

**Files:**
- Verify: `tests/`

- [ ] **Step 1: Run unit suite**

Run: `python -m pytest -m unit -q`
Expected: PASS.

- [ ] **Step 2: Run integration suite**

Run: `python -m pytest -m integration -q`
Expected: PASS.

- [ ] **Step 3: Run smoke suite**

Run: `python -m pytest -m smoke -q`
Expected: PASS or explicit optional skips.

- [ ] **Step 4: Run combined fast gate**

Run: `python -m pytest -m "unit or integration or smoke" -q`
Expected: PASS.

- [ ] **Step 5: Run e2e suite on compute platform**

Run: `python -m pytest -m e2e -q`
Expected: PASS with real server data or explicit skip reason if prerequisites are missing.

- [ ] **Step 6: Commit verification artifacts/update docs if needed**

```bash
git add tests/README.md
# add other touched files if verification required tiny follow-up edits
git commit -m "test: finalize full-layer verification matrix"
```

---

## Self-Review

1. **Spec coverage:**
- Layered taxonomy (unit/integration/smoke/e2e): covered in Tasks 1-5.
- Legacy test modernization: covered in Task 2.
- Server-option-based e2e: covered in Task 3.
- Runner/docs consistency: covered in Task 4.
- Verification matrix: covered in Task 5.

2. **Placeholder scan:**
- No `TBD/TODO/implement later` placeholders.
- Each coding task includes concrete code blocks and concrete commands.

3. **Type consistency:**
- Marker names are consistent across `pytest.ini`, `run_tests.py`, and README.
- E2E fixture names (`server_option`, `require_paths_or_skip`) are consistently referenced.
