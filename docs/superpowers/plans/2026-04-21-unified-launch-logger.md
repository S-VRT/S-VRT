# Unified Launch Logger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route all launch-phase shell output through the project's Python logger so every log line — from data prep through training — lands in `experiments/{task}/log/train_*.log` and Logfire.

**Architecture:** Derive `experiments/{task}/log/` from the config JSON before any logging begins; replace shell FIFO readers with Python stdin-reader subprocesses that call `emit_launch_wrapper_log`; rewrite `launch_echo` to spawn a Python subprocess that does the same. The main training process is unaffected — it initialises its own logger as before and appends to the same file.

**Tech Stack:** Python `logging`, `utils/utils_logger.py`, `launch_train.sh` bash

---

## File Map

| File | Change |
|---|---|
| `utils/utils_logger.py` | Add prefix injection in `emit_launch_wrapper_log` |
| `launch_train.sh` | Add path resolver; rewrite `ensure_launch_logger`, `launch_echo`, `run_with_wrapper`; update all call sites |
| `tests/utils/test_utils_logger_logfire.py` | Add tests for prefix injection and Logfire structured fields |

---

## Task 1: Prefix injection in `emit_launch_wrapper_log`

**Files:**
- Modify: `utils/utils_logger.py:166-197`
- Test: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/utils/test_utils_logger_logfire.py`:

```python
def test_emit_launch_wrapper_log_prefix_stdout(tmp_path):
    log_file = str(tmp_path / "train.log")
    utils_logger.logger_info("test_prefix", log_file, opt=None, add_stream_handler=False, verbose=False)
    utils_logger.emit_launch_wrapper_log(
        "test_prefix", "info", "hello world",
        launch_phase="prepare", launch_stream="stdout"
    )
    import logging
    logging.getLogger("test_prefix").handlers.clear()
    content = (tmp_path / "train.log").read_text()
    assert "[launch/prepare/stdout] hello world" in content


def test_emit_launch_wrapper_log_prefix_no_stream(tmp_path):
    log_file = str(tmp_path / "train2.log")
    utils_logger.logger_info("test_prefix2", log_file, opt=None, add_stream_handler=False, verbose=False)
    utils_logger.emit_launch_wrapper_log(
        "test_prefix2", "info", "dep check",
        launch_phase="dependency", launch_stream=None
    )
    import logging
    logging.getLogger("test_prefix2").handlers.clear()
    content = (tmp_path / "train2.log").read_text()
    assert "[launch/dependency/info] dep check" in content


def test_emit_launch_wrapper_log_logfire_structured_fields(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    opt = _make_opt(tmp_path, use_logfire=True)
    log_file = str(tmp_path / "train3.log")
    utils_logger.logger_info("test_lf", log_file, opt=opt, add_stream_handler=False, verbose=False)
    utils_logger.emit_launch_wrapper_log(
        "test_lf", "info", "data prep done",
        launch_phase="prepare", launch_stream="stdout", launch_mode="local_single"
    )
    import logging
    logging.getLogger("test_lf").handlers.clear()

    assert len(fake.events) == 1
    _, _, kwargs = fake.events[0]
    assert kwargs.get("launch_phase") == "prepare"
    assert kwargs.get("launch_stream") == "stdout"
    assert kwargs.get("launch_mode") == "local_single"
    assert kwargs.get("log_origin") == "launch_wrapper"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /root/projects/S-VRT && python -m pytest tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_prefix_stdout tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_prefix_no_stream tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_logfire_structured_fields -v
```

Expected: FAIL — prefix not present in output.

- [ ] **Step 3: Add prefix injection to `emit_launch_wrapper_log`**

In `utils/utils_logger.py`, replace the body of `emit_launch_wrapper_log` (lines 166-197):

```python
def emit_launch_wrapper_log(
    logger_name,
    level,
    message,
    log_origin='launch_wrapper',
    launch_stream=None,
    launch_phase=None,
    launch_mode=None,
    launch_command=None,
):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        raise RuntimeError(
            f'Logger "{logger_name}" is not initialized. Call logger_info() first.'
        )

    effective_level = str(level).lower()
    valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
    if effective_level not in valid_levels:
        raise ValueError(
            f'Invalid log level "{effective_level}". Must be one of {sorted(valid_levels)}.'
        )

    # Build [launch/{phase}/{stream}] prefix
    parts = ['launch']
    if launch_phase:
        parts.append(launch_phase)
    parts.append(launch_stream if launch_stream else effective_level)
    prefix = '[' + '/'.join(parts) + ']'
    prefixed_message = f'{prefix} {message}' if message else prefix

    extra = {
        'log_origin': log_origin,
        'launch_stream': launch_stream,
        'launch_phase': launch_phase,
        'launch_mode': launch_mode,
        'launch_command': launch_command,
    }
    log_method = getattr(logger, effective_level)
    log_method(prefixed_message, extra=extra)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /root/projects/S-VRT && python -m pytest tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_prefix_stdout tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_prefix_no_stream tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_logfire_structured_fields -v
```

Expected: PASS all three.

- [ ] **Step 5: Run full logger test suite**

```bash
cd /root/projects/S-VRT && python -m pytest tests/utils/test_utils_logger_logfire.py -v
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add utils/utils_logger.py tests/utils/test_utils_logger_logfire.py
git commit -m "feat(logger): add [launch/phase/stream] prefix injection in emit_launch_wrapper_log"
```

---

## Task 2: Early path resolution and logger bootstrap

**Files:**
- Modify: `launch_train.sh:347-452`

This task replaces the `PREP_LOG_DIR` approach with a proper `TRAIN_LOG_DIR` derived from the config, and modifies `ensure_launch_logger` to print the log file path so the shell can capture it.

- [ ] **Step 1: Add `resolve_train_log_dir` function after line 347 (`WRAPPER_LOG_DIR` block)**

Find this block in `launch_train.sh` (around line 349):
```bash
WRAPPER_LOG_DIR="${WRAPPER_LOG_DIR:-/tmp/s-vrt-launch-wrapper}"
mkdir -p "$WRAPPER_LOG_DIR"
```

Replace it with:
```bash
resolve_train_log_dir() {
    local opt_path="$1"
    "$PYTHON_BIN" - "$opt_path" <<'PY'
import sys
from utils import utils_option
opt = utils_option.parse(sys.argv[1], is_train=True)
print(opt['path']['log'])
PY
}
```

- [ ] **Step 2: Modify `ensure_launch_logger` to print the log file path**

Find `ensure_launch_logger` (around line 422):
```bash
ensure_launch_logger() {
    local logger_name="$1"
    local log_dir="$2"
    local opt_path="$3"

    mkdir -p "$log_dir"
    "$PYTHON_BIN" - "$logger_name" "$log_dir" "$opt_path" <<'PY'
import sys
from utils import utils_option, utils_logger

logger_name, log_dir, opt_path = sys.argv[1:4]
opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, f"{log_dir}/{logger_name}.log", opt=opt)
PY
}
```

Replace with:
```bash
ensure_launch_logger() {
    local logger_name="$1"
    local log_dir="$2"
    local opt_path="$3"

    mkdir -p "$log_dir"
    "$PYTHON_BIN" - "$logger_name" "$log_dir" "$opt_path" <<'PY'
import sys, logging
from utils import utils_option, utils_logger

logger_name, log_dir, opt_path = sys.argv[1:4]
opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, f"{log_dir}/{logger_name}.log", opt=opt, verbose=False)
log = logging.getLogger(logger_name)
for h in log.handlers:
    if hasattr(h, 'baseFilename'):
        print(h.baseFilename)
        break
PY
}
```

- [ ] **Step 3: Replace the bootstrap block (lines ~447-452)**

Find:
```bash
# Bootstrap the launch logger early so data-prep and training both have handlers.
# RUNTIME_CONFIG is not yet available here, so use CONFIG_PATH and a derived log dir.
# PREP_LOG_DIR uses the options (config) directory for the pre-training phase since
# RUNTIME_CONFIG has not been materialised yet.
PREP_LOG_DIR="$(dirname "$CONFIG_PATH")"
ensure_launch_logger "train" "$PREP_LOG_DIR" "$CONFIG_PATH"
```

Replace with:
```bash
# Resolve the training log directory from the config before any logging begins.
TRAIN_LOG_DIR="$(resolve_train_log_dir "$CONFIG_PATH")"
LAUNCH_LOG_FILE="$(ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH")"
```

- [ ] **Step 4: Remove the second `ensure_launch_logger` call (around line 587-592)**

Find and delete this block:
```bash
TRAIN_LOG_DIR="$(dirname "$RUNTIME_CONFIG")"
# Re-call ensure_launch_logger with the RUNTIME_CONFIG log dir.  For handlers
# this is intentionally a no-op (they were already attached above), but it
# ensures the correct RUNTIME_CONFIG-derived log directory is used for any
# log files written during the training phase.
ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH"
```

- [ ] **Step 5: Verify the script is syntactically valid**

```bash
bash -n /root/projects/S-VRT/launch_train.sh
```

Expected: no output (no syntax errors).

- [ ] **Step 6: Commit**

```bash
git add launch_train.sh
git commit -m "feat(launch): resolve TRAIN_LOG_DIR from config early, remove PREP_LOG_DIR"
```

---

## Task 3: Rewrite `launch_echo` and update all call sites

**Files:**
- Modify: `launch_train.sh`

`launch_echo` currently has 6 positional args + message. New signature: 4 args + message (drop `log_dir` and `opt_path`). It spawns a Python subprocess that re-initialises the logger in append mode and calls `emit_launch_wrapper_log`.

- [ ] **Step 1: Replace the `launch_echo` function body**

Find (around line 352):
```bash
launch_echo() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local log_dir="$4"
    local opt_path="$5"
    local level="$6"
    shift 6
    local message="$*"

    if [[ "$level" == "error" ]]; then
        printf '%s\n' "$message" >&2
    else
        printf '%s\n' "$message"
    fi
}
```

Replace with:
```bash
launch_echo() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local level="$4"
    shift 4
    local message="$*"

    "$PYTHON_BIN" - "$logger_name" "$level" "$message" "$launch_phase" "$launch_mode" \
        "$LAUNCH_LOG_FILE" "$CONFIG_PATH" <<'PY'
import sys
from utils import utils_option, utils_logger

logger_name, level, message, phase, mode, log_file, opt_path = sys.argv[1:8]
opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, log_file, opt=opt, verbose=False)
utils_logger.emit_launch_wrapper_log(logger_name, level, message,
    launch_phase=phase, launch_mode=mode)
PY
}
```

- [ ] **Step 2: Update all `launch_echo` call sites — remove args 4 and 5**

Every call currently looks like:
```bash
launch_echo "train" "launch" "local_single" "$PREP_LOG_DIR" "$CONFIG_PATH" "info" "message"
```

It must become:
```bash
launch_echo "train" "launch" "local_single" "info" "message"
```

Apply this to all 22 call sites. The full list (by approximate line number after prior edits):

Lines ~454-466 (launch phase, 9 calls):
```bash
launch_echo "train" "launch" "local_single" "info" "=========================================="
launch_echo "train" "launch" "local_single" "info" "VRT Training Launch Script"
launch_echo "train" "launch" "local_single" "info" "=========================================="
launch_echo "train" "launch" "local_single" "info" "Config: $CONFIG_PATH"
launch_echo "train" "launch" "local_single" "info" "Requested GPUs: $GPU_COUNT"
launch_echo "train" "launch" "local_single" "info" "GPU List: $GPU_LIST"
launch_echo "train" "launch" "local_single" "info" "Prepare Data: $PREPARE_DATA"
launch_echo "train" "launch" "local_single" "info" "Generate LMDB: $GENERATE_LMDB"
launch_echo "train" "launch" "local_single" "info" "Dataset Root: ${DATASET_ROOT:-<none>}"
launch_echo "train" "launch" "local_single" "info" "GoPro Root: $EFFECTIVE_GOPRO_ROOT"
launch_echo "train" "launch" "local_single" "info" "Spike Root: $EFFECTIVE_SPIKE_ROOT"
launch_echo "train" "launch" "local_single" "info" "Python: $PYTHON_BIN"
launch_echo "train" "launch" "local_single" "info" ""
```

Lines ~530-532 (dependency phase, 3 calls):
```bash
launch_echo "train" "dependency" "local_single" "info" "=========================================="
launch_echo "train" "dependency" "local_single" "info" "Dependency Preparation"
launch_echo "train" "dependency" "local_single" "info" "=========================================="
```

Lines ~540-543 (train phase, 4 calls):
```bash
launch_echo "train" "train" "local_single" "info" "=========================================="
launch_echo "train" "train" "local_single" "info" "Training Phase"
launch_echo "train" "train" "local_single" "info" "=========================================="
launch_echo "train" "train" "local_single" "info" ""
```

Lines ~650-659 (end of script, 6 calls):
```bash
launch_echo "train" "launch" "local_single" "info" ""
launch_echo "train" "launch" "local_single" "info" "=========================================="
launch_echo "train" "launch" "local_single" "info" "Training completed successfully"
launch_echo "train" "launch" "local_single" "info" "=========================================="
launch_echo "train" "launch" "local_single" "error" "Training exited with code: $EXIT_CODE"
launch_echo "train" "launch" "local_single" "error" "=========================================="
```

- [ ] **Step 3: Verify syntax**

```bash
bash -n /root/projects/S-VRT/launch_train.sh
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add launch_train.sh
git commit -m "feat(launch): route launch_echo through Python logger with prefix"
```

---

## Task 4: Rewrite `run_with_wrapper` with Python stdin readers

**Files:**
- Modify: `launch_train.sh`

Replace the two shell `while IFS= read -r line` background readers with Python stdin-reader subprocesses. Remove `WRAPPER_LOG_DIR` and `/tmp/` file writing. Remove `log_dir` and `opt_path` params from the function signature and update all 5 call sites.

- [ ] **Step 1: Replace the `run_with_wrapper` function**

Find the entire `run_with_wrapper` function (lines ~369-420) and replace with:

```bash
run_with_wrapper() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    shift 3

    local cmd=("$@")

    local stdout_pipe stderr_pipe
    stdout_pipe="$(mktemp -u /tmp/s-vrt-wrapper-stdout.XXXXXX)"
    stderr_pipe="$(mktemp -u /tmp/s-vrt-wrapper-stderr.XXXXXX)"
    mkfifo "$stdout_pipe" "$stderr_pipe"

    "$PYTHON_BIN" - "$logger_name" "stdout" "$launch_phase" "$launch_mode" \
        "$LAUNCH_LOG_FILE" "$CONFIG_PATH" <<'PY' < "$stdout_pipe" &
import sys
from utils import utils_option, utils_logger

logger_name, stream, phase, mode, log_file, opt_path = sys.argv[1:7]
opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, log_file, opt=opt, verbose=False)
for line in sys.stdin:
    line = line.rstrip('\n')
    if line:
        utils_logger.emit_launch_wrapper_log(
            logger_name, 'info', line,
            launch_stream=stream, launch_phase=phase, launch_mode=mode
        )
PY
    local stdout_reader_pid=$!

    "$PYTHON_BIN" - "$logger_name" "stderr" "$launch_phase" "$launch_mode" \
        "$LAUNCH_LOG_FILE" "$CONFIG_PATH" <<'PY' < "$stderr_pipe" &
import sys
from utils import utils_option, utils_logger

logger_name, stream, phase, mode, log_file, opt_path = sys.argv[1:7]
opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, log_file, opt=opt, verbose=False)
for line in sys.stdin:
    line = line.rstrip('\n')
    if line:
        utils_logger.emit_launch_wrapper_log(
            logger_name, 'warning', line,
            launch_stream=stream, launch_phase=phase, launch_mode=mode
        )
PY
    local stderr_reader_pid=$!

    exec 3>"$stdout_pipe" 4>"$stderr_pipe"
    "${cmd[@]}" >&3 2>&4 &
    local cmd_pid=$!
    exec 3>&- 4>&-

    wait "$cmd_pid"
    local cmd_exit_code=$?
    wait "$stdout_reader_pid"
    wait "$stderr_reader_pid"
    rm -f "$stdout_pipe" "$stderr_pipe"

    return "$cmd_exit_code"
}
```

- [ ] **Step 2: Update all 5 `run_with_wrapper` call sites — remove args 4 and 5**

Find each call and remove the `log_dir` and `opt_path` arguments (4th and 5th positional args):

```bash
# data prep (was: "train" "prepare" "local_single" "$PREP_LOG_DIR" "$CONFIG_PATH" \)
run_with_wrapper "train" "prepare" "local_single" \
    "$PYTHON_BIN" scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS

# dependency (was: "train" "dependency" "local_single" "$PREP_LOG_DIR" "$CONFIG_PATH" \)
run_with_wrapper "train" "dependency" "local_single" \
    /bin/bash -lc "..."

# platform DDP (was: "train" "train" "platform_ddp" "$TRAIN_LOG_DIR" "$CONFIG_PATH" \)
run_with_wrapper "train" "train" "platform_ddp" \
    "$PYTHON_BIN" -u main_train_vrt.py --opt "$RUNTIME_CONFIG"

# local multi-GPU (was: "train" "train" "local_multi" "$TRAIN_LOG_DIR" "$CONFIG_PATH" \)
run_with_wrapper "train" "train" "local_multi" \
    env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node="$GPU_COUNT" \
        --standalone \
        main_train_vrt.py --opt "$RUNTIME_CONFIG"

# local single-GPU (was: "train" "train" "local_single" "$TRAIN_LOG_DIR" "$CONFIG_PATH" \)
run_with_wrapper "train" "train" "local_single" \
    env CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
    "$PYTHON_BIN" main_train_vrt.py --opt "$RUNTIME_CONFIG"
```

- [ ] **Step 3: Verify syntax**

```bash
bash -n /root/projects/S-VRT/launch_train.sh
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add launch_train.sh
git commit -m "feat(launch): replace shell FIFO readers with Python stdin readers"
```

---

## Task 5: Cleanup

**Files:**
- Modify: `launch_train.sh` (remove `WRAPPER_LOG_DIR`)
- Delete: `options/train_260421_*.log` (all empty files)

- [ ] **Step 1: Remove `WRAPPER_LOG_DIR` from `launch_train.sh`**

Find and delete:
```bash
WRAPPER_LOG_DIR="${WRAPPER_LOG_DIR:-/tmp/s-vrt-launch-wrapper}"
```

(The `mkdir -p "$WRAPPER_LOG_DIR"` line was already removed when `resolve_train_log_dir` replaced it in Task 2.)

- [ ] **Step 2: Delete the empty log files from `options/`**

```bash
rm /root/projects/S-VRT/options/train_260421_*.log
```

- [ ] **Step 3: Verify syntax one final time**

```bash
bash -n /root/projects/S-VRT/launch_train.sh
```

Expected: no output.

- [ ] **Step 4: Run the full logger test suite**

```bash
cd /root/projects/S-VRT && python -m pytest tests/utils/test_utils_logger_logfire.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add launch_train.sh options/
git commit -m "chore(launch): remove WRAPPER_LOG_DIR and empty options/train_*.log files"
```
