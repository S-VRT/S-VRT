# Two-Run Training Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a true process-internal two-run training orchestrator so one launch command can execute phase-1 fusion warmup and phase-2 fresh optimization with one experiment directory and one continuous external metric run.

**Architecture:** Introduce a small `utils/utils_two_run.py` helper module for config resolution and manifest state, then refactor `main_train_vrt.py` so the current monolithic training loop becomes a reusable single-phase runner wrapped by a two-run orchestrator. Keep `ModelPlain` and `ModelVRT` mostly unchanged; phase boundaries live at the training-entry layer, not inside model optimization internals.

**Tech Stack:** Python, PyTorch, JSON config parsing, pytest, TensorBoard, wandb, SwanLab

**Spec:** [docs/superpowers/specs/2026-04-30-two-run-training-orchestrator-design.md](../specs/2026-04-30-two-run-training-orchestrator-design.md)

---

## File Structure

### New files

- `utils/utils_two_run.py`
  - Owns two-run config merge, validation, resolved-config dumping, manifest persistence, and resume-phase selection.
- `tests/test_two_run_config.py`
  - Covers config merge, validation, forced phase-2 runtime overrides, and resolved-config artifact dumping.
- `tests/test_two_run_manifest.py`
  - Covers manifest initialization, phase transitions, and resume routing.
- `tests/test_two_run_orchestrator.py`
  - Covers global-step mapping, phase transition semantics, logger lifetime, and orchestrator control flow.

### Modified files

- `main_train_vrt.py`
  - Extract single-phase runtime setup and training-loop execution into reusable helpers.
  - Add the top-level two-run orchestrator and remove `sys.exit()` dependence from the training loop.
- `utils/utils_option.py`
  - Add `train.two_run` defaults and parse-time validation glue.
- `options/gopro_rgbspike_server_pase_residual_snapshot.json`
  - Convert the current formal snapshot config to the new `train.two_run` shape.

### Existing tests to keep passing

- `tests/test_phasewise_loader_config.py`
- `tests/test_training_eta.py`
- `tests/models/test_training_timer_boundaries.py`
- `tests/models/test_two_phase_training.py`

---

### Task 1: Add two-run config resolution helpers

**Files:**
- Create: `utils/utils_two_run.py`
- Create: `tests/test_two_run_config.py`
- Modify: `utils/utils_option.py`

- [ ] **Step 1: Write the failing config-resolution tests**

```python
from pathlib import Path

import pytest

from utils.utils_two_run import (
    deep_merge_dict,
    dump_resolved_two_run_opts,
    resolve_two_run_phase_opts,
    validate_two_run_config,
)


def _base_opt(tmp_path: Path):
    return {
        "opt_path": "options/example.json",
        "path": {
            "task": str(tmp_path / "exp"),
            "options": str(tmp_path / "exp" / "options"),
            "pretrained_netG": "weights/base.pth",
            "pretrained_netE": None,
            "pretrained_optimizerG": None,
        },
        "datasets": {
            "train": {
                "gt_size": 128,
                "dataloader_batch_size": 8,
            }
        },
        "train": {
            "freeze_backbone": True,
            "G_optimizer_reuse": True,
            "G_optimizer_lr": 2e-4,
            "G_scheduler_type": "CosineAnnealingWarmRestarts",
            "G_scheduler_periods": 10000,
            "two_run": {
                "enable": True,
                "phase1": {
                    "total_iter": 4000,
                    "G_optimizer_lr": 4e-4,
                    "checkpoint_test": [4000],
                },
                "phase2": {
                    "total_iter": 6000,
                    "G_optimizer_lr": 2e-4,
                    "G_optimizer_reuse": False,
                    "checkpoint_test": [2000, 4000, 6000],
                },
            },
        },
    }


def test_deep_merge_dict_replaces_lists_and_merges_nested_dicts():
    merged = deep_merge_dict(
        {"train": {"a": 1, "lst": [1, 2], "nested": {"x": 1, "y": 2}}},
        {"train": {"lst": [3], "nested": {"y": 9}}},
    )
    assert merged["train"]["a"] == 1
    assert merged["train"]["lst"] == [3]
    assert merged["train"]["nested"] == {"x": 1, "y": 9}


def test_resolve_two_run_phase_opts_applies_phase_overrides(tmp_path):
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(_base_opt(tmp_path))
    assert phase1_opt["train"]["total_iter"] == 4000
    assert phase1_opt["train"]["G_optimizer_lr"] == 4e-4
    assert phase2_opt["train"]["total_iter"] == 6000
    assert phase2_opt["train"]["checkpoint_test"] == [2000, 4000, 6000]


def test_validate_two_run_config_requires_phase_total_iter(tmp_path):
    opt = _base_opt(tmp_path)
    del opt["train"]["two_run"]["phase2"]["total_iter"]
    with pytest.raises(ValueError, match="phase2.total_iter"):
        validate_two_run_config(opt)


def test_validate_two_run_config_rejects_phase2_optimizer_reuse(tmp_path):
    opt = _base_opt(tmp_path)
    opt["train"]["two_run"]["phase2"]["G_optimizer_reuse"] = True
    with pytest.raises(ValueError, match="phase2.*G_optimizer_reuse"):
        validate_two_run_config(opt)


def test_dump_resolved_two_run_opts_writes_three_json_files(tmp_path):
    base = _base_opt(tmp_path)
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(base)
    written = dump_resolved_two_run_opts(base, phase1_opt, phase2_opt)
    assert len(written) == 3
    assert written["base"].exists()
    assert written["phase1"].exists()
    assert written["phase2"].exists()
```

- [ ] **Step 2: Run the config-resolution tests to verify they fail**

Run:

```bash
python -m pytest tests/test_two_run_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.utils_two_run'`

- [ ] **Step 3: Implement the two-run config helper module**

Create `utils/utils_two_run.py`:

```python
import copy
import json
from datetime import datetime
from pathlib import Path


_FORBIDDEN_PHASE2_TRAIN_VALUES = {
    "G_optimizer_reuse": True,
}

_FORBIDDEN_PHASE2_PATH_KEYS = {
    "pretrained_optimizerG",
}


def deep_merge_dict(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_positive_int(value, label):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive int, got {value!r}")


def validate_two_run_config(opt):
    train = opt.get("train", {})
    two_run = train.get("two_run", {}) or {}
    if not two_run.get("enable", False):
        return

    for phase_name in ("phase1", "phase2"):
        phase_cfg = two_run.get(phase_name)
        if not isinstance(phase_cfg, dict):
            raise ValueError(f"train.two_run.{phase_name} must be a dict")
        _require_positive_int(phase_cfg.get("total_iter"), f"train.two_run.{phase_name}.total_iter")

    phase2_cfg = two_run["phase2"]
    for key, forbidden_value in _FORBIDDEN_PHASE2_TRAIN_VALUES.items():
        if phase2_cfg.get(key, False) == forbidden_value:
            raise ValueError(f"train.two_run.phase2.{key} must not be {forbidden_value!r}")

    phase2_path = phase2_cfg.get("path", {}) or {}
    for key in _FORBIDDEN_PHASE2_PATH_KEYS:
        if key in phase2_path:
            raise ValueError(f"train.two_run.phase2.path.{key} is runtime-owned and must not be set")


def _apply_phase_train_overrides(base_opt, phase_override):
    phase_opt = copy.deepcopy(base_opt)
    train_override = copy.deepcopy(phase_override)
    nested_path_override = train_override.pop("path", None)
    phase_opt["train"] = deep_merge_dict(phase_opt.get("train", {}), train_override)
    if nested_path_override:
        phase_opt["path"] = deep_merge_dict(phase_opt.get("path", {}), nested_path_override)
    return phase_opt


def resolve_two_run_phase_opts(opt):
    validate_two_run_config(opt)
    train = opt.get("train", {})
    two_run = train.get("two_run", {}) or {}
    if not two_run.get("enable", False):
        return None, None

    phase1_opt = _apply_phase_train_overrides(opt, two_run["phase1"])
    phase2_opt = _apply_phase_train_overrides(opt, two_run["phase2"])

    phase1_opt["train"]["two_run"] = copy.deepcopy(two_run)
    phase2_opt["train"]["two_run"] = copy.deepcopy(two_run)
    return phase1_opt, phase2_opt


def dump_resolved_two_run_opts(base_opt, phase1_opt, phase2_opt):
    options_dir = Path(base_opt["path"]["options"])
    options_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "base": options_dir / f"{stamp}_base.json",
        "phase1": options_dir / f"{stamp}_phase1_resolved.json",
        "phase2": options_dir / f"{stamp}_phase2_resolved.json",
    }
    payloads = {
        "base": base_opt,
        "phase1": phase1_opt,
        "phase2": phase2_opt,
    }
    for key, dump_path in paths.items():
        dump_path.write_text(json.dumps(payloads[key], indent=2), encoding="utf-8")
    return paths
```

- [ ] **Step 4: Wire parse-time defaults into `utils/utils_option.py`**

Modify `utils/utils_option.py` near the training defaults section:

```python
    if 'two_run' not in opt['train']:
        opt['train']['two_run'] = {'enable': False}
    elif not isinstance(opt['train']['two_run'], dict):
        raise ValueError("train.two_run must be a dict when provided.")

    two_run_cfg = opt['train']['two_run']
    if 'enable' not in two_run_cfg:
        two_run_cfg['enable'] = False
    if two_run_cfg.get('enable', False):
        if 'phase1' not in two_run_cfg or 'phase2' not in two_run_cfg:
            raise ValueError("train.two_run.enable=true requires both phase1 and phase2 blocks.")
```

- [ ] **Step 5: Run the config-resolution tests to verify they pass**

Run:

```bash
python -m pytest tests/test_two_run_config.py -v
```

Expected: PASS all tests in `tests/test_two_run_config.py`

- [ ] **Step 6: Commit**

```bash
git add utils/utils_two_run.py utils/utils_option.py tests/test_two_run_config.py
git commit -m "feat(train): add two-run config resolution helpers"
```

---

### Task 2: Add manifest persistence and resume routing

**Files:**
- Modify: `utils/utils_two_run.py`
- Create: `tests/test_two_run_manifest.py`

- [ ] **Step 1: Write the failing manifest/resume tests**

Create `tests/test_two_run_manifest.py`:

```python
import json
from pathlib import Path

from utils.utils_two_run import (
    build_initial_two_run_state,
    load_two_run_state,
    mark_phase1_completed,
    mark_phase2_started,
    resolve_resume_phase,
    save_two_run_state,
    two_run_state_path,
)


def test_build_initial_two_run_state_sets_phase1_as_entrypoint(tmp_path):
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    assert state["current_phase"] == "phase1"
    assert state["phase1_completed"] is False
    assert state["phase2_started"] is False
    assert state["global_step_offset"] == 0


def test_save_and_load_two_run_state_roundtrip(tmp_path):
    state_path = tmp_path / "two_run_state.json"
    state = build_initial_two_run_state(phase1_total_iter=4, phase2_total_iter=6)
    save_two_run_state(state_path, state)
    loaded = load_two_run_state(state_path)
    assert loaded == state


def test_mark_phase1_completed_records_boundary_checkpoints(tmp_path):
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e="models/4000_E.pth")
    assert state["phase1_completed"] is True
    assert state["phase1_final_G"] == "models/4000_G.pth"
    assert state["phase1_final_E"] == "models/4000_E.pth"
    assert state["global_step_offset"] == 4000


def test_resolve_resume_phase_routes_completed_phase1_to_phase2():
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e=None)
    assert resolve_resume_phase(state) == "phase2_fresh"


def test_resolve_resume_phase_routes_started_phase2_to_phase2_resume():
    state = build_initial_two_run_state(phase1_total_iter=4000, phase2_total_iter=6000)
    mark_phase1_completed(state, phase1_final_g="models/4000_G.pth", phase1_final_e=None)
    mark_phase2_started(state)
    assert resolve_resume_phase(state) == "phase2_resume"
```

- [ ] **Step 2: Run the manifest tests to verify they fail**

Run:

```bash
python -m pytest tests/test_two_run_manifest.py -v
```

Expected: FAIL with `ImportError` for missing manifest helpers

- [ ] **Step 3: Implement manifest helpers in `utils/utils_two_run.py`**

Append these helpers to `utils/utils_two_run.py`:

```python
def two_run_state_path(opt):
    return Path(opt["path"]["task"]) / "two_run_state.json"


def build_initial_two_run_state(*, phase1_total_iter, phase2_total_iter):
    return {
        "two_run_enabled": True,
        "current_phase": "phase1",
        "phase1_total_iter": int(phase1_total_iter),
        "phase2_total_iter": int(phase2_total_iter),
        "phase1_completed": False,
        "phase1_final_G": None,
        "phase1_final_E": None,
        "phase2_started": False,
        "global_step_offset": 0,
        "last_successful_phase_step": 0,
        "last_successful_global_step": 0,
    }


def load_two_run_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_two_run_state(state_path, state):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def mark_phase1_completed(state, *, phase1_final_g, phase1_final_e):
    state["phase1_completed"] = True
    state["current_phase"] = "phase2"
    state["phase1_final_G"] = phase1_final_g
    state["phase1_final_E"] = phase1_final_e
    state["global_step_offset"] = int(state["phase1_total_iter"])


def mark_phase2_started(state):
    state["phase2_started"] = True
    state["current_phase"] = "phase2"


def update_last_successful_step(state, *, phase_step, global_step):
    state["last_successful_phase_step"] = int(phase_step)
    state["last_successful_global_step"] = int(global_step)


def resolve_resume_phase(state):
    if state is None:
        return "phase1_fresh"
    if not state.get("phase1_completed", False):
        return "phase1_resume"
    if not state.get("phase2_started", False):
        return "phase2_fresh"
    return "phase2_resume"
```

- [ ] **Step 4: Add a safety test for the manifest-path helper**

Append to `tests/test_two_run_manifest.py`:

```python
def test_two_run_state_path_uses_task_directory(tmp_path):
    opt = {"path": {"task": str(tmp_path / "experiment")}}
    path = two_run_state_path(opt)
    assert path == tmp_path / "experiment" / "two_run_state.json"
```

- [ ] **Step 5: Run the manifest tests to verify they pass**

Run:

```bash
python -m pytest tests/test_two_run_manifest.py -v
```

Expected: PASS all tests in `tests/test_two_run_manifest.py`

- [ ] **Step 6: Commit**

```bash
git add utils/utils_two_run.py tests/test_two_run_manifest.py
git commit -m "feat(train): add two-run manifest and resume helpers"
```

---

### Task 3: Extract a reusable single-phase runner with global-step mapping

**Files:**
- Modify: `main_train_vrt.py`
- Create: `tests/test_two_run_orchestrator.py`

- [ ] **Step 1: Write the failing phase-runner tests**

Create `tests/test_two_run_orchestrator.py`:

```python
from types import SimpleNamespace

from main_train_vrt import (
    compute_global_step,
    should_finish_phase,
)


def test_compute_global_step_adds_phase_step_to_offset():
    assert compute_global_step(global_step_offset=0, phase_step=1) == 1
    assert compute_global_step(global_step_offset=4000, phase_step=1) == 4001
    assert compute_global_step(global_step_offset=4000, phase_step=6000) == 10000


def test_should_finish_phase_uses_greater_equal_boundary():
    assert should_finish_phase(current_phase_step=3, total_iter=4) is False
    assert should_finish_phase(current_phase_step=4, total_iter=4) is True
    assert should_finish_phase(current_phase_step=5, total_iter=4) is True


def test_run_phase_keeps_tracking_logger_open_until_explicit_close(monkeypatch):
    from main_train_vrt import run_phase

    calls = {"close": 0, "log_scalars": []}

    class _Logger:
        def log_scalars(self, step, scalar_dict, tag_prefix=""):
            calls["log_scalars"].append((step, scalar_dict, tag_prefix))

        def close(self):
            calls["close"] += 1

    class _Model:
        def __init__(self):
            self.timer = SimpleNamespace(current_timings={}, clear=lambda: None)

    monkeypatch.setattr("main_train_vrt.build_phase_runtime", lambda *args, **kwargs: {
        "model": _Model(),
        "train_loader": iter([{"L": 1}, {"L": 2}]),
        "train_sampler": None,
        "train_set": [1, 2],
        "active_train_dataset_opt": {"dataloader_batch_size": 1, "gt_size": 64},
    })

    monkeypatch.setattr("main_train_vrt.execute_training_iteration", lambda **kwargs: {"G_loss": 0.1})
    monkeypatch.setattr("main_train_vrt.finalize_phase", lambda **kwargs: {"final_checkpoint": "models/2_G.pth"})

    result = run_phase(
        phase_opt={"train": {"total_iter": 2, "checkpoint_print": 1}, "rank": 0},
        shared_runtime={"tb_logger": _Logger(), "logger": None},
        phase_name="phase1",
        global_step_offset=0,
        resume_state={"phase_step": 0},
    )

    assert result["last_phase_step"] == 2
    assert result["last_global_step"] == 2
    assert calls["close"] == 0
    assert calls["log_scalars"][0][0] == 1
    assert calls["log_scalars"][1][0] == 2
```

- [ ] **Step 2: Run the phase-runner tests to verify they fail**

Run:

```bash
python -m pytest tests/test_two_run_orchestrator.py -v
```

Expected: FAIL with missing `compute_global_step`, `should_finish_phase`, and `run_phase`

- [ ] **Step 3: Add small step-axis helpers and extract phase-runtime setup**

Add to `main_train_vrt.py` near the existing helper section:

```python
def compute_global_step(global_step_offset, phase_step):
    return int(global_step_offset) + int(phase_step)


def should_finish_phase(current_phase_step, total_iter):
    return int(current_phase_step) >= int(total_iter)


def build_phase_runtime(phase_opt, seed, logger, phase_name):
    train_dataset_opt_base = phase_opt["datasets"]["train"]
    is_phase1 = phase_name == "phase1"
    bundle = build_train_loader_bundle(phase_opt, train_dataset_opt_base, is_phase1, seed, logger)
    model = define_Model(phase_opt)
    model.init_train()
    return {
        "model": model,
        "train_loader": bundle["train_loader"],
        "train_sampler": bundle["train_sampler"],
        "train_set": bundle["train_set"],
        "active_train_dataset_opt": bundle["dataset_opt"],
    }
```

- [ ] **Step 4: Extract `execute_training_iteration()` and `run_phase()`**

Add to `main_train_vrt.py` above `main()`:

```python
def execute_training_iteration(*, model, train_data, phase_step, global_step):
    model.timer.current_timings.clear()
    model.update_learning_rate(phase_step)
    model.feed_data(train_data, current_step=phase_step)
    model.optimize_parameters(phase_step)
    logs = model.current_log()
    logs["phase_step"] = float(phase_step)
    logs["global_step"] = float(global_step)
    return logs


def run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
    logger = shared_runtime["logger"]
    tb_logger = shared_runtime["tb_logger"]
    seed = shared_runtime["seed"]
    runtime = build_phase_runtime(phase_opt, seed, logger, phase_name)
    model = runtime["model"]
    train_loader = runtime["train_loader"]
    phase_total_iter = phase_opt["train"]["total_iter"]
    last_phase_step = int((resume_state or {}).get("phase_step", 0))
    last_global_step = compute_global_step(global_step_offset, last_phase_step)

    train_loader_iter = iter(train_loader)
    while not should_finish_phase(last_phase_step, phase_total_iter):
        phase_step = last_phase_step + 1
        global_step = compute_global_step(global_step_offset, phase_step)
        train_data = next(train_loader_iter)
        logs = execute_training_iteration(
            model=model,
            train_data=train_data,
            phase_step=phase_step,
            global_step=global_step,
        )
        if tb_logger is not None and phase_step % phase_opt["train"]["checkpoint_print"] == 0:
            tb_logger.log_scalars(global_step, logs, tag_prefix="train")
        last_phase_step = phase_step
        last_global_step = global_step

    finalize = finalize_phase(
        model=model,
        phase_opt=phase_opt,
        phase_name=phase_name,
        last_phase_step=last_phase_step,
        last_global_step=last_global_step,
        shared_runtime=shared_runtime,
    )
    return {
        "model": model,
        "runtime": runtime,
        "last_phase_step": last_phase_step,
        "last_global_step": last_global_step,
        **finalize,
    }
```

- [ ] **Step 5: Run the phase-runner tests to verify they pass**

Run:

```bash
python -m pytest tests/test_two_run_orchestrator.py -v
```

Expected: PASS all tests in `tests/test_two_run_orchestrator.py`

- [ ] **Step 6: Commit**

```bash
git add main_train_vrt.py tests/test_two_run_orchestrator.py
git commit -m "refactor(train): extract reusable single-phase runner"
```

---

### Task 4: Add the top-level two-run orchestrator and finalization flow

**Files:**
- Modify: `main_train_vrt.py`
- Modify: `utils/utils_two_run.py`
- Modify: `tests/test_two_run_orchestrator.py`

- [ ] **Step 1: Write the failing orchestrator lifecycle tests**

Append to `tests/test_two_run_orchestrator.py`:

```python
def test_run_experiment_executes_phase1_then_phase2_with_continuous_offset(monkeypatch, tmp_path):
    from main_train_vrt import run_experiment

    phase_calls = []
    saved_states = []

    base_opt = {
        "path": {"task": str(tmp_path / "exp"), "options": str(tmp_path / "exp" / "options")},
        "train": {"two_run": {"enable": True}},
        "rank": 0,
        "datasets": {"train": {}, "test": {}},
    }
    phase1_opt = {"train": {"total_iter": 4}, "datasets": {"train": {}}, "rank": 0}
    phase2_opt = {"train": {"total_iter": 6}, "datasets": {"train": {}}, "rank": 0}

    monkeypatch.setattr("main_train_vrt.resolve_two_run_phase_opts", lambda _opt: (phase1_opt, phase2_opt))
    monkeypatch.setattr("main_train_vrt.dump_resolved_two_run_opts", lambda *args, **kwargs: {})
    monkeypatch.setattr("main_train_vrt.load_two_run_state", lambda _path: None)
    monkeypatch.setattr("main_train_vrt.two_run_state_path", lambda _opt: tmp_path / "two_run_state.json")
    monkeypatch.setattr("main_train_vrt.build_initial_two_run_state", lambda **kwargs: {
        "phase1_total_iter": 4,
        "phase2_total_iter": 6,
        "phase1_completed": False,
        "phase2_started": False,
        "global_step_offset": 0,
    })
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda _path, state: saved_states.append(dict(state)))

    def fake_run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
        phase_calls.append((phase_name, global_step_offset))
        if phase_name == "phase1":
            return {
                "last_phase_step": 4,
                "last_global_step": 4,
                "final_checkpoint_G": "models/4000_G.pth",
                "final_checkpoint_E": None,
                "runtime": {"model": object()},
            }
        return {
            "last_phase_step": 6,
            "last_global_step": 10,
            "final_checkpoint_G": "models/10000_G.pth",
            "final_checkpoint_E": None,
            "runtime": {"model": object()},
        }

    monkeypatch.setattr("main_train_vrt.run_phase", fake_run_phase)
    monkeypatch.setattr("main_train_vrt.build_shared_runtime", lambda _opt, _logger, _tb_logger, seed: {
        "logger": _logger,
        "tb_logger": _tb_logger,
        "seed": seed,
    })
    monkeypatch.setattr("main_train_vrt.mark_phase1_completed", lambda state, **kwargs: state.update({
        "phase1_completed": True,
        "phase1_final_G": kwargs["phase1_final_g"],
        "phase1_final_E": kwargs["phase1_final_e"],
        "global_step_offset": state["phase1_total_iter"],
    }))
    monkeypatch.setattr("main_train_vrt.mark_phase2_started", lambda state: state.update({"phase2_started": True}))
    monkeypatch.setattr("main_train_vrt.close_shared_runtime", lambda _runtime: None)

    result = run_experiment(base_opt, logger=None, tb_logger=None, seed=123)

    assert phase_calls == [("phase1", 0), ("phase2", 4)]
    assert result["last_global_step"] == 10
    assert saved_states[-1]["phase2_started"] is True


def test_prepare_phase2_opt_forces_boundary_checkpoint_and_disables_optimizer_reuse(tmp_path):
    from main_train_vrt import prepare_phase2_opt

    phase2_opt = {
        "path": {
            "pretrained_netG": "weights/old.pth",
            "pretrained_netE": None,
            "pretrained_optimizerG": "models/old_optimizer.pth",
        },
        "train": {"G_optimizer_reuse": True},
    }

    updated = prepare_phase2_opt(
        phase2_opt,
        phase1_final_g="models/4000_G.pth",
        phase1_final_e="models/4000_E.pth",
    )

    assert updated["path"]["pretrained_netG"] == "models/4000_G.pth"
    assert updated["path"]["pretrained_netE"] == "models/4000_E.pth"
    assert updated["path"]["pretrained_optimizerG"] is None
    assert updated["train"]["G_optimizer_reuse"] is False
```

- [ ] **Step 2: Run the orchestrator lifecycle tests to verify they fail**

Run:

```bash
python -m pytest tests/test_two_run_orchestrator.py -v
```

Expected: FAIL with missing `run_experiment`, `prepare_phase2_opt`, or incorrect control flow

- [ ] **Step 3: Add shared-runtime finalization and phase-2 opt preparation**

Add to `main_train_vrt.py`:

```python
from utils.utils_two_run import (
    build_initial_two_run_state,
    dump_resolved_two_run_opts,
    load_two_run_state,
    mark_phase1_completed,
    mark_phase2_started,
    resolve_resume_phase,
    resolve_two_run_phase_opts,
    save_two_run_state,
    two_run_state_path,
)


def build_shared_runtime(opt, logger, tb_logger, seed):
    return {
        "logger": logger,
        "tb_logger": tb_logger,
        "seed": seed,
        "opt": opt,
    }


def close_shared_runtime(shared_runtime):
    tb_logger = shared_runtime.get("tb_logger")
    if tb_logger is not None:
        tb_logger.close()


def prepare_phase2_opt(phase2_opt, *, phase1_final_g, phase1_final_e):
    updated = copy.deepcopy(phase2_opt)
    updated["path"]["pretrained_netG"] = phase1_final_g
    updated["path"]["pretrained_netE"] = phase1_final_e
    updated["path"]["pretrained_optimizerG"] = None
    updated["train"]["G_optimizer_reuse"] = False
    return updated
```

- [ ] **Step 4: Implement the top-level `run_experiment()` orchestrator**

Add to `main_train_vrt.py` above `main()`:

```python
def run_experiment(opt, logger, tb_logger, seed):
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(opt)
    if phase1_opt is None or phase2_opt is None:
        raise ValueError("run_experiment requires train.two_run.enable=true")

    dump_resolved_two_run_opts(opt, phase1_opt, phase2_opt)
    shared_runtime = build_shared_runtime(opt, logger, tb_logger, seed)
    state_path = two_run_state_path(opt)
    state = load_two_run_state(state_path)
    if state is None:
        state = build_initial_two_run_state(
            phase1_total_iter=phase1_opt["train"]["total_iter"],
            phase2_total_iter=phase2_opt["train"]["total_iter"],
        )
        save_two_run_state(state_path, state)

    resume_phase = resolve_resume_phase(state)
    result = None
    try:
        if resume_phase in {"phase1_fresh", "phase1_resume"}:
            result = run_phase(
                phase_opt=phase1_opt,
                shared_runtime=shared_runtime,
                phase_name="phase1",
                global_step_offset=0,
                resume_state={"phase_step": state.get("last_successful_phase_step", 0)},
            )
            mark_phase1_completed(
                state,
                phase1_final_g=result["final_checkpoint_G"],
                phase1_final_e=result["final_checkpoint_E"],
            )
            save_two_run_state(state_path, state)

        prepared_phase2_opt = prepare_phase2_opt(
            phase2_opt,
            phase1_final_g=state["phase1_final_G"],
            phase1_final_e=state["phase1_final_E"],
        )
        mark_phase2_started(state)
        save_two_run_state(state_path, state)

        phase2_resume_step = state.get("last_successful_phase_step", 0) if resume_phase == "phase2_resume" else 0
        result = run_phase(
            phase_opt=prepared_phase2_opt,
            shared_runtime=shared_runtime,
            phase_name="phase2",
            global_step_offset=state["global_step_offset"],
            resume_state={"phase_step": phase2_resume_step},
        )
        return result
    finally:
        close_shared_runtime(shared_runtime)
```

- [ ] **Step 5: Run the orchestrator lifecycle tests to verify they pass**

Run:

```bash
python -m pytest tests/test_two_run_orchestrator.py -v
```

Expected: PASS all tests in `tests/test_two_run_orchestrator.py`

- [ ] **Step 6: Commit**

```bash
git add main_train_vrt.py utils/utils_two_run.py tests/test_two_run_orchestrator.py
git commit -m "feat(train): add two-run experiment orchestrator"
```

---

### Task 5: Integrate finalize-phase behavior, resume bookkeeping, and snapshot config

**Files:**
- Modify: `main_train_vrt.py`
- Modify: `options/gopro_rgbspike_server_pase_residual_snapshot.json`
- Modify: `tests/test_two_run_orchestrator.py`
- Modify: `tests/test_phasewise_loader_config.py`

- [ ] **Step 1: Write the failing finalize/resume/config tests**

Append to `tests/test_two_run_orchestrator.py`:

```python
def test_finalize_phase_returns_global_step_checkpoint_labels(monkeypatch):
    from main_train_vrt import finalize_phase

    saved = []

    class _Model:
        def save(self, label):
            saved.append(("save", label))

        def save_merged(self, label):
            saved.append(("merge", label))

    result = finalize_phase(
        model=_Model(),
        phase_opt={"rank": 0, "train": {"use_lora": True}},
        phase_name="phase2",
        last_phase_step=6,
        last_global_step=10,
        shared_runtime={"logger": None, "tb_logger": None},
    )

    assert ("save", 10) in saved
    assert ("merge", 10) in saved
    assert result["final_checkpoint_G"] == "models/10_G.pth"


def test_resume_phase2_uses_zero_phase_step_when_phase2_not_started(monkeypatch, tmp_path):
    from main_train_vrt import run_experiment

    state = {
        "phase1_total_iter": 4,
        "phase2_total_iter": 6,
        "phase1_completed": True,
        "phase1_final_G": "models/4_G.pth",
        "phase1_final_E": None,
        "phase2_started": False,
        "global_step_offset": 4,
        "last_successful_phase_step": 4,
    }
    phase_calls = []
    monkeypatch.setattr("main_train_vrt.resolve_two_run_phase_opts", lambda _opt: (
        {"train": {"total_iter": 4}, "datasets": {"train": {}}, "rank": 0},
        {"train": {"total_iter": 6, "G_optimizer_reuse": False}, "path": {}, "datasets": {"train": {}}, "rank": 0},
    ))
    monkeypatch.setattr("main_train_vrt.dump_resolved_two_run_opts", lambda *args, **kwargs: {})
    monkeypatch.setattr("main_train_vrt.two_run_state_path", lambda _opt: tmp_path / "two_run_state.json")
    monkeypatch.setattr("main_train_vrt.load_two_run_state", lambda _path: state)
    monkeypatch.setattr("main_train_vrt.save_two_run_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("main_train_vrt.build_shared_runtime", lambda _opt, _logger, _tb_logger, seed: {"seed": seed, "logger": None, "tb_logger": None})
    monkeypatch.setattr("main_train_vrt.close_shared_runtime", lambda _runtime: None)
    monkeypatch.setattr("main_train_vrt.mark_phase2_started", lambda _state: _state.update({"phase2_started": True}))
    monkeypatch.setattr("main_train_vrt.prepare_phase2_opt", lambda phase2_opt, **kwargs: phase2_opt)

    def fake_run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
        phase_calls.append((phase_name, global_step_offset, resume_state["phase_step"]))
        return {"last_global_step": 10, "final_checkpoint_G": "models/10_G.pth", "final_checkpoint_E": None}

    monkeypatch.setattr("main_train_vrt.run_phase", fake_run_phase)

    run_experiment(
        {"path": {"task": str(tmp_path / "exp"), "options": str(tmp_path / "exp" / "options")}, "train": {"two_run": {"enable": True}}, "rank": 0},
        logger=None,
        tb_logger=None,
        seed=1,
    )

    assert phase_calls == [("phase2", 4, 0)]
```

Append to `tests/test_phasewise_loader_config.py`:

```python
def test_snapshot_config_parses_two_run_block():
    opt = utils_option.parse("options/gopro_rgbspike_server_pase_residual_snapshot.json", is_train=True)
    two_run = opt["train"]["two_run"]
    assert two_run["enable"] is True
    assert two_run["phase1"]["total_iter"] == 4000
    assert two_run["phase2"]["total_iter"] == 6000
```

- [ ] **Step 2: Run the finalize/resume/config tests to verify they fail**

Run:

```bash
python -m pytest tests/test_two_run_orchestrator.py tests/test_phasewise_loader_config.py -v
```

Expected: FAIL because `finalize_phase` is missing or because the snapshot config does not yet contain the new `two_run` block

- [ ] **Step 3: Implement `finalize_phase()` and update manifest bookkeeping**

Add to `main_train_vrt.py`:

```python
def finalize_phase(*, model, phase_opt, phase_name, last_phase_step, last_global_step, shared_runtime):
    logger = shared_runtime.get("logger")
    if phase_opt.get("rank", 0) == 0 and logger is not None:
        logger.info("[TWO_RUN] finalizing %s at phase_step=%d global_step=%d", phase_name, last_phase_step, last_global_step)

    model.save(last_global_step)
    if hasattr(model, "save_merged") and phase_opt.get("train", {}).get("use_lora", False):
        model.save_merged(last_global_step)

    return {
        "final_checkpoint_G": f"models/{last_global_step}_G.pth",
        "final_checkpoint_E": f"models/{last_global_step}_E.pth" if phase_opt.get("train", {}).get("E_decay", 0) > 0 else None,
    }
```

Update `run_phase()` so it calls `update_last_successful_step()` after each successful iteration and persists manifest state through `save_two_run_state()` when a `state_path` is provided in `shared_runtime`.

```python
        state = shared_runtime.get("two_run_state")
        state_path = shared_runtime.get("two_run_state_path")
        if state is not None and state_path is not None:
            update_last_successful_step(state, phase_step=phase_step, global_step=global_step)
            save_two_run_state(state_path, state)
```

- [ ] **Step 4: Convert the snapshot config to `train.two_run`**

Replace the current single-loop phase-control fields in `options/gopro_rgbspike_server_pase_residual_snapshot.json` with:

```json
"train": {
  "freeze_backbone": true,
  "partial_load": true,
  "G_lossfn_type": "charbonnier",
  "G_optimizer_type": "adam",
  "G_optimizer_betas": [0.9, 0.99],
  "G_optimizer_wd": 0,
  "checkpoint_print": 100,
  "two_run": {
    "enable": true,
    "phase1": {
      "total_iter": 4000,
      "use_lora": false,
      "fix_iter": 0,
      "fix_keys": [],
      "G_optimizer_lr": 4e-4,
      "G_scheduler_type": "CosineAnnealingWarmRestarts",
      "G_scheduler_periods": 4000,
      "G_scheduler_eta_min": 1e-7,
      "checkpoint_test": [4000],
      "checkpoint_save": 2000
    },
    "phase2": {
      "total_iter": 6000,
      "use_lora": true,
      "phase2_lora_mode": false,
      "trainable_extra_keys": ["spynet", "pa_deform"],
      "fix_iter": 0,
      "fix_keys": ["spynet", "pa_deform"],
      "fix_lr_mul": 0.1,
      "G_optimizer_lr": 2e-4,
      "G_optimizer_reuse": false,
      "G_scheduler_type": "CosineAnnealingWarmRestarts",
      "G_scheduler_periods": 6000,
      "G_scheduler_eta_min": 1e-7,
      "checkpoint_test": [2000, 4000, 6000],
      "checkpoint_save": 2000
    }
  }
}
```

- [ ] **Step 5: Run focused tests, then the broader regression slice**

Run:

```bash
python -m pytest tests/test_two_run_config.py tests/test_two_run_manifest.py tests/test_two_run_orchestrator.py tests/test_phasewise_loader_config.py -v
```

Expected: PASS

Then run:

```bash
python -m pytest tests/test_training_eta.py tests/models/test_training_timer_boundaries.py tests/models/test_two_phase_training.py tests/test_phasewise_loader_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add main_train_vrt.py utils/utils_two_run.py options/gopro_rgbspike_server_pase_residual_snapshot.json tests/test_two_run_config.py tests/test_two_run_manifest.py tests/test_two_run_orchestrator.py tests/test_phasewise_loader_config.py
git commit -m "feat(train): orchestrate true two-run training in one launch"
```

---

## Self-Review

### Spec coverage

- Single-launch UX: Task 4
- One experiment directory and continuous external steps: Tasks 3 and 4
- Fresh phase-2 optimizer/scheduler/scaler: Tasks 1, 4, and 5
- One-config `train.two_run` surface: Tasks 1 and 5
- Manifest-based resume semantics: Tasks 2 and 5
- Global-step checkpoint naming: Tasks 3 and 5
- Backward-compatible config parsing defaults: Task 1

No spec gaps remain.

### Placeholder scan

- No `TODO`, `TBD`, or deferred placeholders remain.
- Every code-changing step includes an explicit code block.
- Every test-running step includes an explicit command and expected result.

### Type consistency

Names used consistently across tasks:

- `resolve_two_run_phase_opts`
- `dump_resolved_two_run_opts`
- `build_initial_two_run_state`
- `mark_phase1_completed`
- `mark_phase2_started`
- `resolve_resume_phase`
- `compute_global_step`
- `run_phase`
- `run_experiment`
- `prepare_phase2_opt`
- `finalize_phase`

---

Plan complete and saved to `docs/superpowers/plans/2026-04-30-two-run-training-orchestrator-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
