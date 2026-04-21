# Unified Launch Logger Design

**Date:** 2026-04-21  
**Branch:** codex/sync  
**Scope:** `launch_train.sh`, `utils/utils_logger.py`

---

## Problem

Three separate log streams exist today with no unification:

| Stream | Location | Writer | Issue |
|---|---|---|---|
| Training logs | `experiments/{task}/log/train_*.log` | Python `logging` | Only covers training phase |
| Launch pre-logs | `options/train_*.log` (empty) | `ensure_launch_logger` | Always empty; pollutes `options/` |
| FIFO shell logs | `/tmp/s-vrt-launch-wrapper/*.log` | Shell `while read` | Separate from training logs; not in Logfire |

`launch_echo` only prints to terminal — nothing is persisted. The `options/` empty files are noise.

---

## Goals

1. All logs (launch + training) land in `experiments/{task}/log/train_*.log`
2. Eliminate `options/train_*.log` empty files
3. Shell subprocess output (data prep, torchrun) passes through Python logger
4. Launch log lines are distinguishable from training lines via prefix
5. Logfire receives both sources with structured fields for filtering

---

## Design

### Path Resolution (Pre-Training)

At the top of `launch_train.sh`, before any logging, resolve `TRAIN_LOG_DIR` by parsing the config:

```bash
TRAIN_LOG_DIR=$("$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from utils import utils_option
opt = utils_option.parse(sys.argv[1], is_train=True)
print(opt['path']['log'])
PY
)
```

`ensure_launch_logger` uses `TRAIN_LOG_DIR` instead of `PREP_LOG_DIR`. `PREP_LOG_DIR` is removed entirely.

### Python stdin Reader (replaces shell FIFO reader)

In `run_with_wrapper`, replace the two shell `while IFS= read -r line` background readers with Python inline processes:

```bash
# stdout reader
"$PYTHON_BIN" - "$logger_name" "stdout" "$launch_phase" "$launch_mode" <<'PY' < "$stdout_pipe" &
import sys
from utils import utils_logger
logger_name, stream, phase, mode = sys.argv[1:5]
for line in sys.stdin:
    utils_logger.emit_launch_wrapper_log(
        logger_name, 'info', line.rstrip('\n'),
        launch_stream=stream, launch_phase=phase, launch_mode=mode
    )
PY

# stderr reader  
"$PYTHON_BIN" - "$logger_name" "stderr" "$launch_phase" "$launch_mode" <<'PY' < "$stderr_pipe" &
import sys
from utils import utils_logger
logger_name, stream, phase, mode = sys.argv[1:5]
for line in sys.stdin:
    utils_logger.emit_launch_wrapper_log(
        logger_name, 'warning', line.rstrip('\n'),
        launch_stream=stream, launch_phase=phase, launch_mode=mode
    )
PY
```

Python reader processes are independent of the training process — no shared GIL, no GPU/CPU contention.

### `launch_echo` Routing

`launch_echo` currently only `printf`s to terminal. Replace with a call to `emit_launch_wrapper_log`. The existing `StreamHandler` on the logger handles terminal output, so no separate printf needed.

```bash
launch_echo() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local log_dir="$4"   # unused after refactor, kept for signature compat during transition
    local opt_path="$5"  # unused after refactor
    local level="$6"
    shift 6
    local message="$*"

    "$PYTHON_BIN" - "$logger_name" "$level" "$message" "$launch_phase" "$launch_mode" <<'PY'
import sys
from utils import utils_logger
logger_name, level, message, phase, mode = sys.argv[1:6]
utils_logger.emit_launch_wrapper_log(logger_name, level, message,
    launch_phase=phase, launch_mode=mode)
PY
}
```

### Log Format

**File (`train_*.log`):**
```
26-04-21 14:14:53.123 : [launch/prepare/stdout] Running data prep script...
26-04-21 14:14:53.456 : [launch/dependency/info] DCNv4 build OK
26-04-21 14:15:01.789 : Training started, iter=0, lr=2e-4
```

Prefix rule: `[launch/{phase}/{stream}]`
- `phase`: `prepare` | `dependency` | `train` | `launch`
- `stream`: `stdout` | `stderr` | `info` | `error`

Training lines have no prefix — they are distinguishable by absence.

Prefix is injected inside `emit_launch_wrapper_log` before calling `logger.info()`.

**Logfire:**
- All lines pass through `_LogfireLoggingHandler` automatically
- `extra` fields `launch_phase`, `launch_stream`, `launch_mode` are passed as structured fields
- Training lines lack these fields — natural distinction for dashboard filtering
- `log_origin='launch_wrapper'` field present on all launch lines

### `_LogfireLoggingHandler` Extra Field Passthrough

Verify that `_LogfireLoggingHandler.emit()` passes `record.__dict__` extras to logfire. If not, add explicit extraction of `launch_phase`, `launch_stream`, `launch_mode` from the log record and include them in the logfire call kwargs.

---

## Files Changed

| File | Change |
|---|---|
| `launch_train.sh` | Add path resolution block; remove `PREP_LOG_DIR`; rewrite `launch_echo`; rewrite `run_with_wrapper` readers |
| `utils/utils_logger.py` | Add prefix injection in `emit_launch_wrapper_log`; verify Logfire extra passthrough |

No new files. No changes to `utils_option.py` or `main_train_vrt.py`.

---

## Non-Goals

- SwanLab text logging (out of scope)
- Log rotation
- Changes to validation/metric logging paths
