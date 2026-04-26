# Mamba Profiler AMP Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add targeted profiler support and configurable mamba mixer precision so phase-1 GPU idle gaps and token mixer cost can be measured and reduced without bypassing the existing project logger.

**Architecture:** Keep scalar timing on the existing `Timer` -> `current_log()` -> training logger/TensorBoard/W&B/SwanLab path. Add a small training-profiler helper that wraps selected iterations with `torch.profiler` and emits trace files under the experiment directory while logging only summaries and paths through the existing logger. Add `mamba_amp_policy` to `MambaFusionOperator` so the mixer can run in current fp32-safe mode, AMP autocast mode, or explicit half/bfloat16 experiments.

**Tech Stack:** PyTorch `torch.profiler`, project `Timer`, existing `utils_logger.Logger`, DDP `torch.distributed`, pytest.

---

## File Structure

- `main_train_vrt.py`: remove unused `model.logger = logger`; integrate profiler lifecycle in the training loop; keep DDP timing summary on all ranks before rank0 logging.
- `utils/utils_profiler.py`: new focused helper for parsing profiler config, creating trace directories, starting/stopping `torch.profiler`, and logging trace paths through the existing logger.
- `models/fusion/operators/mamba.py`: add `mamba_amp_policy` parsing and execute `mamba_mixer` under fp32/autocast/fp16/bf16 policy without changing default behavior.
- `models/architectures/vrt/vrt.py`: keep existing timer propagation to fusion operator; no logger changes.
- `options/gopro_rgbspike_server.json`: add disabled-by-default profiler config and explicit default `mamba_amp_policy`.
- `tests/models/test_mamba_timing.py`: add mamba AMP policy unit tests.
- `tests/test_profiler_config.py`: new tests for profiler config parsing and trace path behavior.
- `tests/models/test_training_timer_boundaries.py`: keep existing DDP timing summary regression; add one assertion that non-time training keys are not duplicated into time logs if needed.

---

### Task 1: Clean Logger Integration and Keep Timing on Existing Route

**Files:**
- Modify: `main_train_vrt.py`
- Test: `tests/models/test_training_timer_boundaries.py`

- [ ] **Step 1: Write the failing test for timing summary staying scalar-only**

Append this test to `tests/models/test_training_timer_boundaries.py`:

```python
def test_timing_summary_keeps_non_time_keys_unchanged():
    from main_train_vrt import build_timing_summary

    logs = {
        "time_forward": 0.5,
        "fusion_warmup_stage": "writeback_only",
        "G_loss": 0.1,
    }

    summarized = build_timing_summary(logs, dist_enabled=False, device=None)

    assert summarized["fusion_warmup_stage"] == "writeback_only"
    assert summarized["G_loss"] == 0.1
    assert summarized["time_forward"] == 0.5
    assert summarized["time_forward_max"] == 0.5
    assert summarized["time_forward_mean"] == 0.5
    assert "fusion_warmup_stage_max" not in summarized
```

- [ ] **Step 2: Run the focused timing tests**

Run:

```bash
uv run pytest tests/models/test_training_timer_boundaries.py -q
```

Expected: PASS if the current timing summary behavior is already correct.

- [ ] **Step 3: Remove unused logger assignment**

In `main_train_vrt.py`, delete this line after `model = define_Model(opt)`:

```python
model.logger = logger
```

Do not add another logger attribute to the model.

- [ ] **Step 4: Verify timing tests still pass**

Run:

```bash
uv run pytest tests/models/test_training_timer_boundaries.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add main_train_vrt.py tests/models/test_training_timer_boundaries.py
git commit -m "chore(train): keep timing diagnostics on existing logger route"
```

---

### Task 2: Add Configurable Torch Profiler Helper

**Files:**
- Create: `utils/utils_profiler.py`
- Create: `tests/test_profiler_config.py`
- Modify: `main_train_vrt.py`
- Modify: `options/gopro_rgbspike_server.json`

- [ ] **Step 1: Write failing profiler config tests**

Create `tests/test_profiler_config.py`:

```python
from pathlib import Path


def test_profiler_config_disabled_by_default():
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt({}, experiment_dir=Path("experiments/task"), rank=0)

    assert cfg.enable is False
    assert cfg.should_profile_rank is False


def test_profiler_config_builds_trace_dir_for_rank_zero(tmp_path):
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt(
        {
            "profiler": {
                "enable": True,
                "start_iter": 100,
                "wait": 1,
                "warmup": 1,
                "active": 2,
                "repeat": 1,
                "ranks": [0],
                "record_shapes": True,
                "with_stack": False,
                "profile_memory": True,
            }
        },
        experiment_dir=tmp_path,
        rank=0,
    )

    assert cfg.enable is True
    assert cfg.should_profile_rank is True
    assert cfg.start_iter == 100
    assert cfg.trace_dir == tmp_path / "profiles" / "rank0"


def test_profiler_config_skips_unselected_rank(tmp_path):
    from utils.utils_profiler import TrainProfilerConfig

    cfg = TrainProfilerConfig.from_opt(
        {"profiler": {"enable": True, "ranks": [0]}},
        experiment_dir=tmp_path,
        rank=1,
    )

    assert cfg.enable is True
    assert cfg.should_profile_rank is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/test_profiler_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.utils_profiler'`.

- [ ] **Step 3: Implement `utils/utils_profiler.py`**

Create `utils/utils_profiler.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TrainProfilerConfig:
    enable: bool
    start_iter: int
    wait: int
    warmup: int
    active: int
    repeat: int
    ranks: tuple[int, ...] | None
    record_shapes: bool
    with_stack: bool
    profile_memory: bool
    trace_dir: Path
    rank: int

    @classmethod
    def from_opt(cls, train_opt: dict[str, Any], experiment_dir, rank: int) -> "TrainProfilerConfig":
        raw = train_opt.get("profiler", {}) or {}
        trace_root = Path(experiment_dir) / "profiles"
        raw_ranks = raw.get("ranks", [0])
        ranks = None if raw_ranks in (None, "all") else tuple(int(item) for item in raw_ranks)
        return cls(
            enable=bool(raw.get("enable", False)),
            start_iter=int(raw.get("start_iter", 0)),
            wait=max(int(raw.get("wait", 1)), 0),
            warmup=max(int(raw.get("warmup", 1)), 0),
            active=max(int(raw.get("active", 2)), 1),
            repeat=max(int(raw.get("repeat", 1)), 1),
            ranks=ranks,
            record_shapes=bool(raw.get("record_shapes", True)),
            with_stack=bool(raw.get("with_stack", False)),
            profile_memory=bool(raw.get("profile_memory", True)),
            trace_dir=trace_root / f"rank{rank}",
            rank=int(rank),
        )

    @property
    def should_profile_rank(self) -> bool:
        return self.enable and (self.ranks is None or self.rank in self.ranks)


class TrainProfiler:
    def __init__(self, cfg: TrainProfilerConfig, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.profiler = None

    def maybe_start(self):
        if not self.cfg.should_profile_rank:
            return
        self.cfg.trace_dir.mkdir(parents=True, exist_ok=True)
        schedule = torch.profiler.schedule(
            wait=self.cfg.wait,
            warmup=self.cfg.warmup,
            active=self.cfg.active,
            repeat=self.cfg.repeat,
        )
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.cfg.trace_dir)),
            record_shapes=self.cfg.record_shapes,
            profile_memory=self.cfg.profile_memory,
            with_stack=self.cfg.with_stack,
        )
        self.profiler.start()
        if self.logger is not None:
            self.logger.info(f"[profiler] enabled rank={self.cfg.rank} trace_dir={self.cfg.trace_dir}")

    def step(self, current_step: int):
        if self.profiler is None or current_step < self.cfg.start_iter:
            return
        self.profiler.step()

    def close(self):
        if self.profiler is None:
            return
        self.profiler.stop()
        if self.logger is not None:
            self.logger.info(f"[profiler] trace_dir={self.cfg.trace_dir}")
        self.profiler = None
```

- [ ] **Step 4: Run profiler config tests**

Run:

```bash
uv run pytest tests/test_profiler_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Integrate profiler into `main_train_vrt.py`**

Add import near other utils imports:

```python
from utils.utils_profiler import TrainProfiler, TrainProfilerConfig
```

After `model.init_train()` and before the epoch loop, add:

```python
profiler_cfg = TrainProfilerConfig.from_opt(
    opt.get("train", {}),
    experiment_dir=opt["path"]["task"],
    rank=opt["rank"],
)
train_profiler = TrainProfiler(profiler_cfg, logger=logger if opt["rank"] == 0 else None)
train_profiler.maybe_start()
```

After `model.optimize_parameters(current_step)`, add:

```python
train_profiler.step(current_step)
```

Before the final logger shutdown / process end, add a `finally`-safe close if the file has a natural cleanup block. If there is no `try/finally`, add near the end of `main()` before logger close:

```python
train_profiler.close()
```

- [ ] **Step 6: Add disabled-by-default config**

In `options/gopro_rgbspike_server.json`, inside `"train"`, add:

```json
"profiler": {
  "enable": false,
  "start_iter": 100,
  "wait": 1,
  "warmup": 1,
  "active": 2,
  "repeat": 1,
  "ranks": [0],
  "record_shapes": true,
  "with_stack": false,
  "profile_memory": true
}
```

- [ ] **Step 7: Verify syntax and tests**

Run:

```bash
python -m py_compile main_train_vrt.py utils/utils_profiler.py
uv run pytest tests/test_profiler_config.py tests/models/test_training_timer_boundaries.py -q
```

Expected: py_compile exits 0; tests pass.

- [ ] **Step 8: Commit**

```bash
git add main_train_vrt.py utils/utils_profiler.py tests/test_profiler_config.py options/gopro_rgbspike_server.json
git commit -m "feat(train): add configurable torch profiler window"
```

---

### Task 3: Add Profiler Labels Around Mamba Stages

**Files:**
- Modify: `models/fusion/operators/mamba.py`
- Test: `tests/models/test_mamba_timing.py`

- [ ] **Step 1: Write failing test for record_function labels**

Append this test to `tests/models/test_mamba_timing.py`:

```python
def test_mamba_operator_uses_record_function_labels(monkeypatch):
    from models.fusion.operators import mamba as mamba_module
    from models.fusion.operators.mamba import MambaFusionOperator

    class _FakeBlock(torch.nn.Module):
        def __init__(self, model_dim, d_state, d_conv, expand):
            super().__init__()

        def forward(self, tokens):
            return tokens

    labels = []

    class _Record:
        def __init__(self, name):
            labels.append(name)

        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(mamba_module, "_MambaBlock", _FakeBlock)
    monkeypatch.setattr(mamba_module.torch.profiler, "record_function", _Record)

    operator = MambaFusionOperator(
        rgb_chans=3,
        spike_chans=1,
        out_chans=3,
        operator_params={"token_dim": 4, "token_stride": 2, "num_layers": 1},
    )

    rgb = torch.randn(1, 2, 3, 8, 8)
    spike = torch.randn(1, 2, 4, 8, 8)
    operator(rgb, spike)

    assert "mamba_rgb_encoder" in labels
    assert "mamba_spike_encoder" in labels
    assert "mamba_token_pack" in labels
    assert "mamba_mixer" in labels
    assert "mamba_writeback" in labels
    assert "mamba_upsample" in labels
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
uv run pytest tests/models/test_mamba_timing.py::test_mamba_operator_uses_record_function_labels -q
```

Expected: FAIL because mamba stages use `Timer` but not `torch.profiler.record_function`.

- [ ] **Step 3: Implement combined timing/profile context**

In `models/fusion/operators/mamba.py`, add:

```python
from contextlib import contextmanager
```

Replace `from contextlib import nullcontext` with:

```python
from contextlib import contextmanager, nullcontext
```

Add this method to `MambaFusionOperator`:

```python
    @contextmanager
    def _profiled_timer(self, name: str):
        with torch.profiler.record_function(name):
            with self._timer(name):
                yield
```

Replace all `with self._timer("mamba_..."):` calls with `with self._profiled_timer("mamba_..."):` while preserving block bodies.

- [ ] **Step 4: Run mamba timing tests**

Run:

```bash
uv run pytest tests/models/test_mamba_timing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/fusion/operators/mamba.py tests/models/test_mamba_timing.py
git commit -m "feat(fusion): label mamba stages for torch profiler"
```

---

### Task 4: Add Configurable Mamba Mixer AMP Policy

**Files:**
- Modify: `models/fusion/operators/mamba.py`
- Modify: `options/gopro_rgbspike_server.json`
- Test: `tests/models/test_mamba_timing.py`

- [ ] **Step 1: Write failing tests for AMP policy parsing**

Append these tests to `tests/models/test_mamba_timing.py`:

```python
def test_mamba_operator_defaults_to_fp32_mixer_policy(monkeypatch):
    from models.fusion.operators import mamba as mamba_module
    from models.fusion.operators.mamba import MambaFusionOperator

    class _FakeBlock(torch.nn.Module):
        def __init__(self, model_dim, d_state, d_conv, expand):
            super().__init__()

        def forward(self, tokens):
            assert tokens.dtype == torch.float32
            return tokens

    monkeypatch.setattr(mamba_module, "_MambaBlock", _FakeBlock)
    operator = MambaFusionOperator(3, 1, 3, {"token_dim": 4, "token_stride": 2, "num_layers": 1})
    assert operator.mamba_amp_policy == "fp32"


def test_mamba_operator_rejects_unknown_amp_policy():
    from models.fusion.operators.mamba import MambaFusionOperator

    try:
        MambaFusionOperator(3, 1, 3, {"mamba_amp_policy": "bad"})
    except ValueError as exc:
        assert "mamba_amp_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad mamba_amp_policy")
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
uv run pytest tests/models/test_mamba_timing.py -q
```

Expected: FAIL because `mamba_amp_policy` is not defined.

- [ ] **Step 3: Implement policy parsing**

In `MambaFusionOperator.__init__`, after `enable_diagnostics`, add:

```python
        mamba_amp_policy = str(operator_params.get("mamba_amp_policy", "fp32")).strip().lower()
        if mamba_amp_policy not in {"fp32", "autocast", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mamba_amp_policy={mamba_amp_policy!r}; "
                "expected one of ['fp32', 'autocast', 'fp16', 'bf16']."
            )
```

Store:

```python
        self.mamba_amp_policy = mamba_amp_policy
```

- [ ] **Step 4: Implement mixer dtype context helper**

Add this method:

```python
    def _prepare_mixer_input(self, seq: torch.Tensor) -> torch.Tensor:
        if self.mamba_amp_policy == "fp32":
            return seq.float().contiguous()
        if self.mamba_amp_policy == "fp16":
            return seq.to(dtype=torch.float16).contiguous()
        if self.mamba_amp_policy == "bf16":
            return seq.to(dtype=torch.bfloat16).contiguous()
        return seq.contiguous()
```

Update the `mamba_mixer` block:

```python
        with self._profiled_timer("mamba_mixer"):
            if self.mamba_amp_policy == "fp32":
                with self._device_autocast_disabled_context(seq):
                    seq = self._prepare_mixer_input(seq)
                    seq = self._run_mamba_token_mixer(seq)
            else:
                seq = self._prepare_mixer_input(seq)
                seq = self._run_mamba_token_mixer(seq)
            seq = seq.to(dtype=mixer_input_dtype)
```

This preserves current default behavior and allows explicit experiments.

- [ ] **Step 5: Add explicit default config**

In `options/gopro_rgbspike_server.json`, inside `netG.fusion.operator_params`, add:

```json
"mamba_amp_policy": "fp32"
```

- [ ] **Step 6: Run tests and syntax check**

Run:

```bash
python -m py_compile models/fusion/operators/mamba.py
uv run pytest tests/models/test_mamba_timing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add models/fusion/operators/mamba.py tests/models/test_mamba_timing.py options/gopro_rgbspike_server.json
git commit -m "feat(fusion): add configurable mamba mixer amp policy"
```

---

### Task 5: Run Controlled Profiling Experiments

**Files:**
- Modify only runtime config if needed: `options/gopro_rgbspike_server.json`
- Outputs: `experiments/gopro_tfp4_scflow_mamba_tokenized_dcn4/profiles/`

- [ ] **Step 1: Stop current training before profiling**

Find current training processes:

```bash
ps -eo pid,etimes,cmd | rg "(torchrun|torch\\.distributed|main_train_vrt|launch_train|python3 -m torch)"
```

Terminate only the PIDs belonging to the current `S-VRT` training screen/log.

- [ ] **Step 2: Enable a short profiler window**

In runtime config or temporary copied config, set:

```json
"profiler": {
  "enable": true,
  "start_iter": 100,
  "wait": 1,
  "warmup": 1,
  "active": 2,
  "repeat": 1,
  "ranks": [0],
  "record_shapes": true,
  "with_stack": false,
  "profile_memory": true
}
```

Keep `mamba_amp_policy: "fp32"` for baseline.

- [ ] **Step 3: Start baseline profiling run**

Run:

```bash
bash ./launch_train.sh 4 options/gopro_rgbspike_server.json --detach
```

Expected: a new `terminal_*.log` path printed.

- [ ] **Step 4: Wait for profiler trace**

Monitor:

```bash
rg -n "profiler|<epoch|mamba_mixer|forward_max|batch_wait_max" experiments/gopro_tfp4_scflow_mamba_tokenized_dcn4/terminal_*.log
```

Expected: log contains `[profiler] enabled rank=0 trace_dir=...` and an iter line with `mamba_mixer_max`.

- [ ] **Step 5: Record baseline metrics**

Collect:

```bash
uv run python - <<'PY'
from pathlib import Path
logs = sorted(Path("experiments/gopro_tfp4_scflow_mamba_tokenized_dcn4").glob("train_*.log"))
print(logs[-1])
for line in logs[-1].read_text(errors="ignore").splitlines():
    if "<epoch" in line:
        print(line)
PY
```

Record `mamba_mixer_max`, `forward_max`, `backward_max`, and trace directory.

- [ ] **Step 6: Run AMP policy experiment**

Change only:

```json
"mamba_amp_policy": "fp16"
```

Repeat the profiling run. If training errors inside `mamba_ssm`, record the exact exception and revert policy to `fp32`.

- [ ] **Step 7: Compare**

Compare:

```text
fp32 mamba_mixer_max vs fp16 mamba_mixer_max
fp32 forward_max vs fp16 forward_max
fp32 loss/backward stability vs fp16 stability
GPU zero-sample percentage before/after
```

If fp16 is stable and significantly faster, propose making `fp16` the experiment config for this run. If it is unstable, keep `fp32` and proceed to a real compute-saving `writeback_only` design.

---

## Self-Review

- Spec coverage: The plan covers logger route cleanup, profiler support, profiler labels, mamba AMP policy, and controlled measurement.
- Placeholder scan: No TODO/TBD placeholders remain; each task has concrete files, snippets, commands, and expected outcomes.
- Type consistency: `TrainProfilerConfig`, `TrainProfiler`, `mamba_amp_policy`, `_profiled_timer`, and `build_timing_summary` names are consistent across tasks.
