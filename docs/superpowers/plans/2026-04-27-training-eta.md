# Training ETA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ETA field to training progress logs that estimates remaining wall time until `train.total_iter`.

**Architecture:** Keep ETA calculation as pure helper functions in `main_train_vrt.py` so it is cheap to test and easy to reuse in the existing progress log block. The training loop will use existing per-iteration timing metrics and append `eta: HH:MM:SS` to the rank-0 logger message.

**Tech Stack:** Python, pytest, existing `main_train_vrt.py` training logger.

---

### Task 1: ETA Helper Tests

**Files:**
- Create: `tests/test_training_eta.py`
- Modify: `main_train_vrt.py`

- [x] **Step 1: Write the failing test**

Create `tests/test_training_eta.py`:

```python
from main_train_vrt import compute_training_eta, format_eta


def test_format_eta_uses_hours_minutes_seconds():
    assert format_eta(3661.2) == "01:01:01"


def test_format_eta_clamps_negative_and_invalid_values():
    assert format_eta(-3.0) == "00:00:00"
    assert format_eta(float("nan")) == "00:00:00"


def test_compute_training_eta_uses_remaining_iters_and_step_time():
    assert compute_training_eta(current_step=25, total_iter=100, seconds_per_iter=2.0) == "00:02:30"


def test_compute_training_eta_returns_zero_when_training_is_complete():
    assert compute_training_eta(current_step=100, total_iter=100, seconds_per_iter=2.0) == "00:00:00"
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_eta.py -q`
Expected: FAIL because `compute_training_eta` and `format_eta` do not exist in `main_train_vrt.py`.

- [ ] **Step 3: Write minimal implementation**

Add helpers near the existing timing helper in `main_train_vrt.py`:

```python
def format_eta(seconds):
    if not isinstance(seconds, numbers.Real) or not math.isfinite(float(seconds)) or seconds <= 0:
        total_seconds = 0
    else:
        total_seconds = int(float(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def compute_training_eta(current_step, total_iter, seconds_per_iter):
    remaining_iters = max(int(total_iter) - int(current_step), 0)
    seconds = remaining_iters * float(seconds_per_iter)
    return format_eta(seconds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training_eta.py -q`
Expected: PASS.

### Task 2: Integrate ETA Into Training Logs

**Files:**
- Modify: `main_train_vrt.py`

- [ ] **Step 1: Add ETA to the existing progress message**

In the rank-0 progress logging block, after `logs` is available and before appending scalar log entries, compute:

```python
seconds_per_iter = logs.get("time_total", logs.get("time_iter"))
if isinstance(seconds_per_iter, numbers.Real):
    message += "eta: {:s} ".format(
        compute_training_eta(current_step, opt["train"]["total_iter"], seconds_per_iter)
    )
```

- [ ] **Step 2: Run targeted tests**

Run: `uv run pytest tests/test_training_eta.py -q`
Expected: PASS.

- [ ] **Step 3: Run syntax check**

Run: `python -m py_compile main_train_vrt.py tests/test_training_eta.py`
Expected: exit code 0.

### Self-Review

- Spec coverage: ETA is based on `total_iter`, appended to the existing rank-0 training progress logger, and tested through pure helpers.
- Placeholder scan: no placeholders remain.
- Type consistency: helper names and call sites match.
