# Launch Wrapper Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing project logger so `launch_train.sh` can funnel shell stdout/stderr, `torchrun` launcher output, platform DDP startup output, and fallback tracebacks into the same main logger and Logfire path as training logs.

**Architecture:** Keep `utils/utils_logger.py` as the only logging hub. Add a small helper that writes wrapper-originated lines into the existing main logger with structured `extra` fields, then update `launch_train.sh` so every final execution path runs through one shared shell wrapper that mirrors output to the terminal while forwarding lines into that helper. Validate with focused unit tests for the logger helper and shell-level smoke checks for command wrapping behavior.

**Tech Stack:** Bash, Python stdlib logging, pytest.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `utils/utils_logger.py` | Modify | Add helper APIs for launch-wrapper log forwarding into the existing main logger |
| `tests/utils/test_utils_logger_logfire.py` | Modify | Add tests proving wrapper-originated lines are accepted by the main logger and forwarded to Logfire text events with structured metadata |
| `launch_train.sh` | Modify | Add shared shell wrapper for stdout/stderr capture and route all launch modes through it |
| `docs/superpowers/specs/2026-04-19-launch-wrapper-logging-design.md` | Reference | Source-of-truth design for implementation decisions |

---

## Task 1: Extend the Main Logger With A Launch Wrapper Entry Point

**Files:**
- Modify: `utils/utils_logger.py`
- Modify: `tests/utils/test_utils_logger_logfire.py`

- [ ] **Step 1: Write the failing tests for wrapper log forwarding**

Append these tests to `tests/utils/test_utils_logger_logfire.py`:

```python
def test_emit_launch_wrapper_log_uses_existing_main_logger(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "train"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True)
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    utils_logger.emit_launch_wrapper_log(
        logger_name=logger_name,
        level="info",
        message="shell stdout line",
        log_origin="launch_wrapper",
        launch_stream="stdout",
        launch_phase="train",
        launch_mode="local_single",
        launch_command="python main_train_vrt.py --opt opt.json",
    )

    text_events = [event for event in fake.events if event[1] == "svrt log record"]

    assert text_events
    assert text_events[0][2]["message"] == "shell stdout line"
    assert text_events[0][2]["log_origin"] == "launch_wrapper"
    assert text_events[0][2]["launch_stream"] == "stdout"
    assert text_events[0][2]["launch_phase"] == "train"
    assert text_events[0][2]["launch_mode"] == "local_single"


def test_emit_launch_wrapper_log_maps_stderr_to_error(monkeypatch, tmp_path):
    fake = _FakeLogfire()
    monkeypatch.setattr(utils_logger, "LOGFIRE_AVAILABLE", True)
    monkeypatch.setattr(utils_logger, "logfire", fake)

    logger_name = "train"
    py_logger = logging.getLogger(logger_name)
    py_logger.handlers = []
    py_logger.propagate = False

    opt = _make_opt(tmp_path, use_logfire=True, logfire_log_text=True)
    utils_logger.logger_info(logger_name, str(tmp_path / "train.log"), opt=opt)

    utils_logger.emit_launch_wrapper_log(
        logger_name=logger_name,
        level="error",
        message="traceback line",
        log_origin="launch_wrapper",
        launch_stream="stderr",
        launch_phase="train",
        launch_mode="platform_ddp",
        launch_command="python -u main_train_vrt.py --opt opt.json",
    )

    error_events = [
        event for event in fake.events
        if event[1] == "svrt log record" and event[2]["level"] == "ERROR"
    ]

    assert error_events
    assert error_events[0][2]["message"] == "traceback line"
    assert error_events[0][2]["launch_stream"] == "stderr"
    assert error_events[0][2]["launch_mode"] == "platform_ddp"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_uses_existing_main_logger tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_maps_stderr_to_error -v`
Expected: FAIL with `AttributeError` because `emit_launch_wrapper_log` does not exist.

- [ ] **Step 3: Implement the minimal helper in `utils/utils_logger.py`**

Edit `utils/utils_logger.py` and add a helper below `logger_info()`:

```python
def emit_launch_wrapper_log(
    logger_name,
    level,
    message,
    log_origin="launch_wrapper",
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

    extra = {
        "log_origin": log_origin,
        "launch_stream": launch_stream,
        "launch_phase": launch_phase,
        "launch_mode": launch_mode,
        "launch_command": launch_command,
    }

    log_method = getattr(logger, str(level).lower(), logger.info)
    log_method(message, extra=extra)
```

Update `_LogfireLoggingHandler.emit()` so it forwards those extra fields when present:

```python
            log_method(
                'svrt log record',
                message=record.getMessage(),
                logger_name=record.name,
                level=record.levelname,
                pathname=record.pathname,
                lineno=record.lineno,
                log_origin=getattr(record, 'log_origin', 'train_core'),
                launch_stream=getattr(record, 'launch_stream', None),
                launch_phase=getattr(record, 'launch_phase', None),
                launch_mode=getattr(record, 'launch_mode', None),
                launch_command=getattr(record, 'launch_command', None),
                **self.bridge.context,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_uses_existing_main_logger tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_maps_stderr_to_error -v`
Expected: PASS

- [ ] **Step 5: Run the full logger regression file**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add utils/utils_logger.py tests/utils/test_utils_logger_logfire.py
git commit -m "feat(logging): add launch wrapper bridge to main logger"
```

---

## Task 2: Route All Launch Paths Through One Shared Shell Wrapper

**Files:**
- Modify: `launch_train.sh`

- [ ] **Step 1: Add a shared wrapper function without changing existing training commands yet**

Edit `launch_train.sh` and add these helpers above the “Data Preparation” section:

```bash
WRAPPER_LOG_DIR="${WRAPPER_LOG_DIR:-/tmp/s-vrt-launch-wrapper}"
mkdir -p "$WRAPPER_LOG_DIR"

emit_wrapper_line() {
    local logger_name="$1"
    local level="$2"
    local message="$3"
    local launch_stream="$4"
    local launch_phase="$5"
    local launch_mode="$6"
    local launch_command="$7"

    "$PYTHON_BIN" - "$logger_name" "$level" "$message" "$launch_stream" "$launch_phase" "$launch_mode" "$launch_command" <<'PY'
import sys
from utils import utils_logger

logger_name, level, message, launch_stream, launch_phase, launch_mode, launch_command = sys.argv[1:8]
utils_logger.emit_launch_wrapper_log(
    logger_name=logger_name,
    level=level,
    message=message,
    log_origin="launch_wrapper",
    launch_stream=launch_stream or None,
    launch_phase=launch_phase or None,
    launch_mode=launch_mode or None,
    launch_command=launch_command or None,
)
PY
}

run_with_wrapper() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    shift 3

    local cmd=("$@")
    local command_str="${cmd[*]}"
    local timestamp
    timestamp="$(date +%y%m%d_%H%M%S)"
    local stdout_log="$WRAPPER_LOG_DIR/${launch_phase}_${launch_mode}_${timestamp}.stdout.log"
    local stderr_log="$WRAPPER_LOG_DIR/${launch_phase}_${launch_mode}_${timestamp}.stderr.log"

    emit_wrapper_line "$logger_name" "info" "launch wrapper started: ${command_str}" "stdout" "$launch_phase" "$launch_mode" "$command_str"

    local stdout_pipe stderr_pipe
    stdout_pipe="$(mktemp -u /tmp/s-vrt-wrapper-stdout.XXXXXX)"
    stderr_pipe="$(mktemp -u /tmp/s-vrt-wrapper-stderr.XXXXXX)"
    mkfifo "$stdout_pipe" "$stderr_pipe"

    while IFS= read -r line; do
        printf '%s\n' "$line"
        printf '%s\n' "$line" >> "$stdout_log"
        emit_wrapper_line "$logger_name" "info" "$line" "stdout" "$launch_phase" "$launch_mode" "$command_str"
    done < "$stdout_pipe" &
    local stdout_reader_pid=$!

    while IFS= read -r line; do
        printf '%s\n' "$line" >&2
        printf '%s\n' "$line" >> "$stderr_log"
        emit_wrapper_line "$logger_name" "error" "$line" "stderr" "$launch_phase" "$launch_mode" "$command_str"
    done < "$stderr_pipe" &
    local stderr_reader_pid=$!

    "${cmd[@]}" > "$stdout_pipe" 2> "$stderr_pipe"
    local cmd_exit_code=$?

    wait "$stdout_reader_pid"
    wait "$stderr_reader_pid"
    rm -f "$stdout_pipe" "$stderr_pipe"

    if [[ $cmd_exit_code -eq 0 ]]; then
        emit_wrapper_line "$logger_name" "info" "launch wrapper completed successfully: ${command_str}" "stdout" "$launch_phase" "$launch_mode" "$command_str"
    else
        emit_wrapper_line "$logger_name" "error" "launch wrapper failed with exit code ${cmd_exit_code}: ${command_str}" "stderr" "$launch_phase" "$launch_mode" "$command_str"
    fi

    return "$cmd_exit_code"
}
```

- [ ] **Step 2: Initialize the main logger before any wrapped command can emit**

Still in `launch_train.sh`, before data preparation or dependency preparation, add a minimal Python logger bootstrap command:

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

Call it once after runtime config resolution:

```bash
TRAIN_LOG_DIR="$(dirname "$RUNTIME_CONFIG")"
ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH"
```

- [ ] **Step 3: Route data preparation through the wrapper**

Replace the direct invocation:

```bash
    "$PYTHON_BIN" scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS
```

with:

```bash
    run_with_wrapper "train" "prepare" "local_single" \
        "$PYTHON_BIN" scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS
```

- [ ] **Step 4: Route platform DDP and local training paths through the wrapper**

Replace the three direct execution branches with wrapped versions:

```bash
    run_with_wrapper "train" "train" "platform_ddp" \
        "$PYTHON_BIN" -u main_train_vrt.py --opt "$RUNTIME_CONFIG"
```

```bash
        run_with_wrapper "train" "train" "local_multi" \
            env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
            "$PYTHON_BIN" -m torch.distributed.run \
                --nproc_per_node="$GPU_COUNT" \
                --standalone \
                main_train_vrt.py --opt "$RUNTIME_CONFIG"
```

```bash
        run_with_wrapper "train" "train" "local_single" \
            env CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
            "$PYTHON_BIN" main_train_vrt.py --opt "$RUNTIME_CONFIG"
```

- [ ] **Step 5: Smoke-check shell syntax**

Run: `bash -n launch_train.sh`
Expected: exit code 0 and no syntax errors

- [ ] **Step 6: Verify help output still works**

Run: `./launch_train.sh --help`
Expected: usage text prints successfully and exits 0

- [ ] **Step 7: Verify a wrapped command path emits without breaking exit codes**

Run: `GPU_COUNT=1 ./launch_train.sh 1 options/gopro_rgbspike_server.json --help`
Expected: help output still prints successfully; wrapper setup does not break the shell script.

- [ ] **Step 8: Commit**

```bash
git add launch_train.sh
git commit -m "feat(launch): wrap shell output into main logger"
```

---

## Task 3: End-To-End Verification For Launch-Originated Logfire Text Events

**Files:**
- Modify: `launch_train.sh` (only if verification reveals a bug)
- Modify: `utils/utils_logger.py` (only if verification reveals a bug)

- [ ] **Step 1: Run logger unit tests again after shell integration**

Run: `python -m pytest tests/utils/test_utils_logger_logfire.py -v`
Expected: PASS

- [ ] **Step 2: Run a real wrapped training launch with a Logfire-enabled config**

Run: `MPLCONFIGDIR=/tmp/mpl TORCH_EXTENSIONS_DIR=/tmp/torch_extensions ./launch_train.sh 1 options/gopro_rgbspike_local_debug.json`
Expected: command may still fail later due to environment or dataset issues, but launch-originated lines and stderr should now be written through the main logger before failure.

- [ ] **Step 3: Inspect the generated training log for launch-wrapper lines**

Run: `rg -n "launch wrapper|log_origin|prepare|platform_ddp|local_single|local_multi" experiments -S`
Expected: matches showing wrapper-originated lines were written into the main training log files.

- [ ] **Step 4: Confirm Logfire-side distinction fields exist through the fake logger tests**

Re-run: `python -m pytest tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_uses_existing_main_logger tests/utils/test_utils_logger_logfire.py::test_emit_launch_wrapper_log_maps_stderr_to_error -v`
Expected: PASS, proving `log_origin`, `launch_stream`, `launch_phase`, `launch_mode`, and `launch_command` are forwarded into Logfire text events.

- [ ] **Step 5: Commit final verification-driven fixes if needed**

```bash
git add launch_train.sh utils/utils_logger.py tests/utils/test_utils_logger_logfire.py
git commit -m "test(logging): verify launch wrapper forwarding"
```
