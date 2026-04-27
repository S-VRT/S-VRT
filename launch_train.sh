#!/bin/bash

# ================================================================================
# Modern DDP Launch Script for VRT Training with Data Preparation
# ================================================================================
# 
# This script supports two scenarios:
#
# 1. Platform DDP (e.g., cloud platform with pre-injected env vars):
#    - Platform sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
#    - Each process runs the same command independently
#    - DO NOT use torchrun (it would create nested processes)
#    - Command: python -u main_train_vrt.py --opt CONFIG_PATH
#
# 2. Local/Self-managed training:
#    - Single GPU: python main_train_vrt.py --opt CONFIG_PATH
#    - Multi-GPU: torchrun --nproc_per_node=N main_train_vrt.py --opt CONFIG_PATH
#
# New Features:
# - Automatic dataset preparation (GoPro + Spike)
# - Optional LMDB generation for faster data loading
# - Dataset validation before training
#
# Usage:
#   ./launch_train.sh [GPU_COUNT] [CONFIG_PATH] [--prepare-data] [--generate-lmdb] [--force-prepare]
#
# Examples:
#   ./launch_train.sh                                  # Auto-detect available GPUs, default config
#   ./launch_train.sh 1                                # Use one available GPU, default config
#   ./launch_train.sh 1 --prepare-data                 # Single GPU, prepare data first
#   ./launch_train.sh 1 --prepare-data --generate-lmdb # Single GPU, prepare data + LMDB
#   ./launch_train.sh 4                                # 4 GPUs, default config
#   ./launch_train.sh 8 options/vrt/custom.json        # 8 GPUs, custom config
#   ./launch_train.sh 1 --prepare-data --force-prepare # Force re-prepare data
#
# ================================================================================

# Default configuration
DEFAULT_CONFIG="options/gopro_rgbspike_server.json"
DEFAULT_GOPRO_ROOT="/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large"
DEFAULT_SPIKE_ROOT="/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq"
DEFAULT_GPU_COUNT="auto"
ORIGINAL_ARGS=("$@")

if command -v uv >/dev/null 2>&1 && uv run python -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')"
elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "Python not found."
    exit 1
fi

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$(dirname "$PYTHON_BIN"):/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$PYTHON_BIN")/../lib/python3.11/site-packages/torch/lib:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

sanitize_positive_thread_env() {
    local name="$1"
    local default_value="$2"
    local current_value="${!name:-}"
    if [[ -z "$current_value" || ! "$current_value" =~ ^[1-9][0-9]*$ ]]; then
        export "$name=$default_value"
    fi
}

# External shells may inject invalid values such as OMP_NUM_THREADS=0. Keep
# per-process thread pools bounded because DataLoader workers also inherit them.
CPU_THREADS_PER_PROCESS="${CPU_THREADS_PER_PROCESS:-4}"
sanitize_positive_thread_env OMP_NUM_THREADS "$CPU_THREADS_PER_PROCESS"
sanitize_positive_thread_env MKL_NUM_THREADS "$CPU_THREADS_PER_PROCESS"
sanitize_positive_thread_env OPENBLAS_NUM_THREADS "$CPU_THREADS_PER_PROCESS"
sanitize_positive_thread_env NUMEXPR_NUM_THREADS "$CPU_THREADS_PER_PROCESS"

# Parse arguments
GPU_COUNT=""
CONFIG_PATH=""
PREPARE_DATA=false
GENERATE_LMDB=false
FORCE_PREPARE=false
DATASET_ROOT=""
OVERRIDE_GOPRO_ROOT=""
OVERRIDE_SPIKE_ROOT=""
GPU_LIST=""
TERMINAL_LOG=true
TERMINAL_MODE="attach"
AUTODL_TENSORBOARD=${AUTODL_TENSORBOARD:-true}
TENSORBOARD_LOGDIR_OVERRIDE=""

# Parse all arguments
for arg in "$@"; do
    case $arg in
        --prepare-data)
            PREPARE_DATA=true
            shift
            ;;
        --generate-lmdb)
            GENERATE_LMDB=true
            shift
            ;;
        --force-prepare)
            FORCE_PREPARE=true
            shift
            ;;
        --dataset-root=*)
            DATASET_ROOT="${arg#*=}"
            shift
            ;;
        --gopro-root=*)
            OVERRIDE_GOPRO_ROOT="${arg#*=}"
            shift
            ;;
        --spike-root=*)
            OVERRIDE_SPIKE_ROOT="${arg#*=}"
            shift
            ;;
        --gpus=*)
            GPU_LIST="${arg#*=}"
            shift
            ;;
        --foreground)
            TERMINAL_MODE="foreground"
            shift
            ;;
        --attach)
            TERMINAL_MODE="attach"
            shift
            ;;
        --detach)
            TERMINAL_MODE="detach"
            shift
            ;;
        --no-terminal-log)
            TERMINAL_LOG=false
            shift
            ;;
        --autodl-tensorboard)
            AUTODL_TENSORBOARD=true
            shift
            ;;
        --no-autodl-tensorboard)
            AUTODL_TENSORBOARD=false
            shift
            ;;
        --tensorboard-logdir=*)
            TENSORBOARD_LOGDIR_OVERRIDE="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [GPU_COUNT] [CONFIG_PATH] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  GPU_COUNT        Number of available GPUs to use (default: auto/all available)"
            echo "  CONFIG_PATH      Path to training config (default: $DEFAULT_CONFIG)"
            echo ""
            echo "Options:"
            echo "  --prepare-data   Prepare GoPro + Spike dataset before training"
            echo "  --generate-lmdb  Generate LMDB files (requires --prepare-data)"
            echo "  --force-prepare  Force re-preparation even if data exists"
            echo "  --dataset-root=/path/to/gopro_spike           Root where zip was extracted"
            echo "  --gopro-root=/path/to/GOPRO_Large             Override GoPro root"
            echo "  --spike-root=/path/to/GOPRO_Large_spike_seq   Override Spike root"
            echo "  --gpus=0,1,2     Comma-separated GPU ids for single-node DDP (overrides auto-detect)"
            echo "  --attach         Run inside screen and attach immediately (default)"
            echo "  --detach         Run inside a detached screen session and return"
            echo "  --foreground     Run in the current terminal, still recording terminal_*.log"
            echo "  --no-terminal-log Disable screen/script terminal transcript wrapping"
            echo "  --autodl-tensorboard    Ensure AutoDL TensorBoard uses the project experiments logdir (default)"
            echo "  --no-autodl-tensorboard Do not manage AutoDL TensorBoard before training"
            echo "  --tensorboard-logdir=/path Override TensorBoard logdir used by --autodl-tensorboard"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --detach"
            echo "  $0 1 --prepare-data"
            echo "  $0 4 --prepare-data --generate-lmdb"
            echo "  $0 8 options/vrt/custom.json"
            echo "  $0 4 options/gopro_rgbspike_server_debug.json --detach"
            echo "  screen -r svrt_<task>_<timestamp>"
            exit 0
            ;;
        *)
            if [[ -z "$GPU_COUNT" ]] && [[ "$arg" =~ ^[0-9]+$ ]]; then
                GPU_COUNT="$arg"
            elif [[ -z "$CONFIG_PATH" ]] && [[ -f "$arg" || "$arg" == *.json ]]; then
                CONFIG_PATH="$arg"
            fi
            ;;
    esac
done

detect_available_gpus() {
    local max_used_mb="${SVRT_GPU_FREE_MEMORY_MAX_MB:-512}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F, -v max_used_mb="$max_used_mb" '
            {
                gsub(/^[ \t]+|[ \t]+$/, "", $1)
                gsub(/^[ \t]+|[ \t]+$/, "", $2)
                if ($1 != "" && $2 + 0 <= max_used_mb) {
                    if (out != "") out = out ","
                    out = out $1
                }
            }
            END { if (out != "") print out }
        '
}

resolve_gpu_selection() {
    local requested_count="$1"
    local requested_list="$2"
    local available_list selected_list

    if [[ -n "$requested_list" ]]; then
        GPU_LIST="$requested_list"
        GPU_LIST_SOURCE="manual"
        return 0
    fi

    available_list="$(detect_available_gpus || true)"
    if [[ -z "$available_list" ]]; then
        echo "Error: no available GPU detected. Override with --gpus=0 or adjust SVRT_GPU_FREE_MEMORY_MAX_MB." >&2
        exit 1
    fi

    if [[ -z "$requested_count" || "$requested_count" == "auto" ]]; then
        GPU_LIST="$available_list"
        GPU_LIST_SOURCE="auto"
        return 0
    fi

    IFS=',' read -r -a available_ids <<< "$available_list"
    if [[ "$requested_count" -gt "${#available_ids[@]}" ]]; then
        echo "Error: requested $requested_count GPU(s), but only ${#available_ids[@]} available: $available_list" >&2
        exit 1
    fi

    selected_list=""
    local idx
    for ((idx = 0; idx < requested_count; idx++)); do
        if [[ -n "$selected_list" ]]; then
            selected_list+=","
        fi
        selected_list+="${available_ids[$idx]}"
    done

    GPU_LIST="$selected_list"
    GPU_LIST_SOURCE="auto_limited"
}

# Set defaults if not provided.
GPU_COUNT=${GPU_COUNT:-$DEFAULT_GPU_COUNT}
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG}

resolve_gpu_selection "$GPU_COUNT" "$GPU_LIST"

# Normalize GPU list (remove spaces) and derive effective GPU count.
GPU_LIST=$(echo "$GPU_LIST" | tr -d '[:space:]')
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
GPU_COUNT_FROM_LIST=${#GPU_ID_ARRAY[@]}
if [[ "$GPU_COUNT_FROM_LIST" -eq 0 ]]; then
    echo "Invalid GPU list specified via --gpus (value: '$GPU_LIST')."
    exit 1
fi

if [[ "$GPU_COUNT" != "auto" && "$GPU_COUNT" -ne "$GPU_COUNT_FROM_LIST" ]]; then
    echo "Info: GPU_COUNT ($GPU_COUNT) and GPU list size ($GPU_COUNT_FROM_LIST) differ."
    echo "      Using GPU list size for torchrun."
fi
GPU_COUNT=$GPU_COUNT_FROM_LIST

if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT=12355
fi

resolve_train_log_dir() {
    local opt_path="$1"
    "$PYTHON_BIN" - "$opt_path" <<'PY'
import contextlib
import io
import sys
from utils import utils_option
with contextlib.redirect_stdout(io.StringIO()):
    opt = utils_option.parse(sys.argv[1], is_train=True)
print(opt['path']['log'])
PY
}

resolve_tensorboard_logdir() {
    local opt_path="$1"
    if [[ -n "$TENSORBOARD_LOGDIR_OVERRIDE" ]]; then
        "$PYTHON_BIN" - "$TENSORBOARD_LOGDIR_OVERRIDE" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    path = pathlib.Path.cwd() / path
print(path.resolve())
PY
        return
    fi

    "$PYTHON_BIN" - "$opt_path" <<'PY'
import contextlib
import io
import pathlib
import sys
from utils import utils_option

with contextlib.redirect_stdout(io.StringIO()):
    opt = utils_option.parse(sys.argv[1], is_train=True)
root = pathlib.Path(opt['path'].get('root', 'experiments')).expanduser()
if not root.is_absolute():
    root = pathlib.Path.cwd() / root
print(root.resolve())
PY
}

ensure_autodl_tensorboard() {
    local logdir="$1"
    local port="${AUTODL_TENSORBOARD_PORT:-6007}"

    if [[ -z "$logdir" ]]; then
        launch_echo "train" "tensorboard" "local_single" "warning" "AutoDL TensorBoard: empty logdir; skipping setup."
        return 0
    fi

    if ! command -v tensorboard >/dev/null 2>&1; then
        launch_echo "train" "tensorboard" "local_single" "warning" "AutoDL TensorBoard: tensorboard command not found; skipping setup."
        return 0
    fi

    mkdir -p "$logdir"

    local current_pids current_logdir
    current_pids="$(pgrep -f "tensorboard.*--port[= ]${port}" || true)"
    current_logdir="$(ps -eo pid,args | "$PYTHON_BIN" -c '
import re
import sys

port = sys.argv[1]
for raw in sys.stdin:
    line = raw.strip()
    if "tensorboard" not in line:
        continue
    if not re.search(rf"--port(?:=|\s+){re.escape(port)}(?:\s|$)", line):
        continue
    match = re.search(r"--logdir(?:=|\s+)(\S+)", line)
    if match:
        print(match.group(1))
        break
' "$port")"

    if [[ -n "$current_logdir" ]]; then
        local current_abs
        current_abs="$("$PYTHON_BIN" - "$current_logdir" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser()
if not path.is_absolute():
    path = pathlib.Path.cwd() / path
print(path.resolve())
PY
)"
        if [[ "$current_abs" == "$logdir" ]]; then
            launch_echo "train" "tensorboard" "local_single" "info" "AutoDL TensorBoard already configured: port $port logdir $logdir"
            return 0
        fi
        launch_echo "train" "tensorboard" "local_single" "info" "AutoDL TensorBoard logdir mismatch: port $port currently uses $current_abs; expected $logdir"
    else
        launch_echo "train" "tensorboard" "local_single" "info" "AutoDL TensorBoard: no existing port $port process found; starting one for $logdir"
    fi

    if command -v supervisorctl >/dev/null 2>&1; then
        supervisorctl stop tensorboard >/dev/null 2>&1 || true
    fi

    if [[ -n "$current_pids" ]]; then
        kill $current_pids >/dev/null 2>&1 || true
        sleep 2
    fi

    nohup tensorboard --host 0.0.0.0 --port "$port" --logdir "$logdir" >/tmp/s-vrt-tensorboard.log 2>&1 &
    launch_echo "train" "tensorboard" "local_single" "info" "AutoDL TensorBoard started: port $port logdir $logdir"
}

quote_shell_word() {
    printf '%q' "$1"
}

build_reentry_command() {
    local command_text
    command_text="cd $(quote_shell_word "$(pwd)") && SVRT_LAUNCH_INNER=1"
    if [[ -n "${SVRT_TERMINAL_LOG:-}" ]]; then
        command_text+=" SVRT_TERMINAL_LOG=$(quote_shell_word "$SVRT_TERMINAL_LOG")"
    fi
    command_text+=" exec bash $(quote_shell_word "$0")"
    local arg
    for arg in "$@"; do
        command_text+=" $(quote_shell_word "$arg")"
    done
    printf '%s' "$command_text"
}

start_terminal_transcript_wrapper() {
    local log_dir="$1"
    shift

    local timestamp task_name session_name terminal_log info_file inner_command script_command
    timestamp="$(date +%y%m%d_%H%M%S)"
    task_name="$(basename "$log_dir")"
    task_name="${task_name//[!A-Za-z0-9_.-]/_}"
    session_name="svrt_${task_name}_${timestamp}"
    terminal_log="${log_dir}/terminal_${timestamp}.log"
    info_file="${log_dir}/screen_${timestamp}.info"

    mkdir -p "$log_dir"

    if ! command -v script >/dev/null 2>&1; then
        echo "Warning: script not found; terminal transcript logging is disabled." >&2
        return 0
    fi

    export SVRT_TERMINAL_LOG="$terminal_log"
    inner_command="$(build_reentry_command "$@")"
    script_command="script -q -f -e -c $(quote_shell_word "$inner_command") $(quote_shell_word "$terminal_log")"

    {
        echo "screen_session=$session_name"
        echo "terminal_log=$terminal_log"
        echo "launch_mode=$TERMINAL_MODE"
        echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
        echo "reattach_command=screen -r $session_name"
        echo "tail_command=tail -f $terminal_log"
    } > "$info_file"

    if [[ "$TERMINAL_MODE" == "foreground" ]]; then
        echo "Terminal transcript: $terminal_log"
        exec script -q -f -e -c "$inner_command" "$terminal_log"
    fi

    if ! command -v screen >/dev/null 2>&1; then
        echo "Warning: screen not found; falling back to foreground script logging." >&2
        echo "Terminal transcript: $terminal_log"
        exec script -q -f -e -c "$inner_command" "$terminal_log"
    fi

    echo "Screen session: $session_name"
    echo "Terminal transcript: $terminal_log"
    echo "Session info: $info_file"

    if [[ "$TERMINAL_MODE" == "detach" || ! -t 0 ]]; then
        screen -dmS "$session_name" bash -lc "$script_command"
        echo "Detached. Reattach with: screen -r $session_name"
        exit 0
    fi

    exec screen -S "$session_name" bash -lc "$script_command"
}

if [[ "$TERMINAL_LOG" == true && "${SVRT_LAUNCH_INNER:-0}" != "1" ]]; then
    TRAIN_LOG_DIR_FOR_TRANSCRIPT="$(resolve_train_log_dir "$CONFIG_PATH")"
    if [[ -n "$TRAIN_LOG_DIR_FOR_TRANSCRIPT" ]]; then
        start_terminal_transcript_wrapper "$TRAIN_LOG_DIR_FOR_TRANSCRIPT" "${ORIGINAL_ARGS[@]}"
    else
        echo "Warning: could not resolve log dir from $CONFIG_PATH; continuing without terminal transcript wrapper." >&2
    fi
fi

# Resolve effective dataset roots
EFFECTIVE_GOPRO_ROOT="$DEFAULT_GOPRO_ROOT"
EFFECTIVE_SPIKE_ROOT="$DEFAULT_SPIKE_ROOT"

if [[ -n "$DATASET_ROOT" ]]; then
    if [[ -d "$DATASET_ROOT/GOPRO_Large" ]]; then
        EFFECTIVE_GOPRO_ROOT="$DATASET_ROOT/GOPRO_Large"
    fi
    if [[ -d "$DATASET_ROOT/GOPRO_Large_spike_seq" ]]; then
        EFFECTIVE_SPIKE_ROOT="$DATASET_ROOT/GOPRO_Large_spike_seq"
    fi
fi

if [[ -n "$OVERRIDE_GOPRO_ROOT" ]]; then
    EFFECTIVE_GOPRO_ROOT="$OVERRIDE_GOPRO_ROOT"
fi
if [[ -n "$OVERRIDE_SPIKE_ROOT" ]]; then
    EFFECTIVE_SPIKE_ROOT="$OVERRIDE_SPIKE_ROOT"
fi

# Recommended NCCL environment variables for stability
export NCCL_ASYNC_ERROR_HANDLING=1
# Uncomment if no InfiniBand available:
# export NCCL_IB_DISABLE=1
# Uncomment for improved stability with some models:
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# Function to handle errors gracefully
# Shows error message and waits for user confirmation, then exits with code 0
# This prevents terminal windows from closing on error, allowing users to
# see the error and retry in the same terminal session
handle_error() {
    local exit_code=$1
    local message="${2:-}"
    
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        if [[ -n "$message" ]]; then
            echo "$message"
        fi
        echo ""
        echo "错误信息已显示在上方。"
        echo "您可以在修复问题后，在同一终端会话中重新运行此脚本。"
        echo ""
        # Check if running in an interactive terminal
        if [[ -t 1 ]]; then
            echo "按 Enter 键继续（脚本将结束，但终端保持打开）..."
            read -r _
        fi
        # Exit with code 0 to prevent terminal from closing
        # The actual error code is already displayed in the message
        exit 0
    fi
}

# Ensure optional runtime deps required by selected configs are available.
ensure_python_package() {
    local import_name="$1"
    local pip_name="$2"

    if "$PYTHON_BIN" -c "import ${import_name}" >/dev/null 2>&1; then
        echo "Dependency check: ${pip_name} already installed."
        return 0
    fi

    echo "Dependency check: ${pip_name} missing, installing..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON_BIN" "${pip_name}"
    else
        "$PYTHON_BIN" -m ensurepip --upgrade
        "$PYTHON_BIN" -m pip install "${pip_name}"
    fi
}

ensure_python_package_version() {
    local import_name="$1"
    local version_expr="$2"
    local pip_name="${3:-${import_name}${version_expr}}"

    if "$PYTHON_BIN" - "$import_name" "$version_expr" >/dev/null 2>&1 <<'PY'
import importlib
import sys
from packaging.version import Version

module_name, required = sys.argv[1:3]
mod = importlib.import_module(module_name)
installed = getattr(mod, "__version__", None)
if installed is None:
    raise SystemExit(1)
required = required.strip()
if required.startswith(">="):
    ok = Version(installed) >= Version(required[2:])
elif required.startswith("=="):
    ok = Version(installed) == Version(required[2:])
else:
    raise SystemExit(1)
raise SystemExit(0 if ok else 1)
PY
    then
        echo "Dependency check: ${pip_name} already compatible."
        return 0
    fi

    echo "Dependency check: ${pip_name} incompatible or missing, installing..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON_BIN" "${pip_name}"
    else
        "$PYTHON_BIN" -m ensurepip --upgrade
        "$PYTHON_BIN" -m pip install "${pip_name}"
    fi
}

ensure_dcnv4_module() {
    local import_output=""
    import_output=$("$PYTHON_BIN" - <<'PY' 2>&1
import glob
import os

print("DCNv4 import probe:")
for path in sorted(glob.glob("DCNv4/ext*.so")):
    print(f"  artifact: {os.path.abspath(path)}")

from models.op.dcnv4 import DCNv4  # noqa: F401
from DCNv4 import ext

print(f"  import ok: {getattr(ext, '__file__', '<unknown>')}")
PY
)
    if [[ $? -eq 0 ]]; then
        echo "$import_output"
        echo "Dependency check: DCNv4 module already available."
        return 0
    fi
    echo "$import_output"
    echo "Dependency check: DCNv4 import probe failed, rebuilding..."

    if [[ ! -d "models/op/dcnv4" ]]; then
        echo "Warning: models/op/dcnv4 not found, skip DCNv4 build."
        return 0
    fi

    echo "Dependency check: DCNv4 module missing, building with setup.py develop..."
    # setup.py builds extension name "DCNv4.ext" and copies it into a top-level
    # "DCNv4/" directory. Ensure this destination exists to avoid:
    # "could not create 'DCNv4/ext....so': No such file or directory".
    mkdir -p DCNv4
    if [[ ! -f "DCNv4/__init__.py" ]]; then
        cat > DCNv4/__init__.py <<'PYINIT'
from . import ext
PYINIT
    fi

    # Important: setup.py declares packages as models.op.dcnv4.*, so it must
    # be executed from repo root instead of models/op/dcnv4 directory.
    "$PYTHON_BIN" models/op/dcnv4/setup.py develop
    local dcn_exit_code=$?
    if [[ $dcn_exit_code -ne 0 ]]; then
        handle_error $dcn_exit_code "DCNv4 构建失败，请检查上面的错误信息。"
    fi

    # Post-build sanity check: ensure extension can be imported in current env.
    if ! "$PYTHON_BIN" -c "from models.op.dcnv4 import DCNv4; from DCNv4 import ext" >/dev/null 2>&1; then
        handle_error 1 "DCNv4 扩展导入失败（DCNv4.ext 不可用），请检查编译环境与 Python 环境是否一致。"
    fi
}

launch_echo() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local level="$4"
    shift 4
    local message="$*"

    launch_echo_lines "$logger_name" "$launch_phase" "$launch_mode" "$level" "$message"
}

launch_emit_record() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local level="$4"
    local launch_stream="$5"
    shift 5
    local message="$*"

    printf '%s\t%s\t%s\t%s\t%s\n' "$level" "$launch_phase" "$launch_mode" "$launch_stream" "$message" >&5
}

launch_echo_lines() {
    local logger_name="$1"
    local launch_phase="$2"
    local launch_mode="$3"
    local level="$4"
    shift 4

    local message
    for message in "$@"; do
        launch_emit_record "$logger_name" "$launch_phase" "$launch_mode" "$level" "-" "$message"
    done
}

start_launch_logger() {
    local logger_name="$1"
    local log_file="$2"
    local opt_path="$3"
    local daemon_code='
import contextlib
import io
import sys
from utils import utils_option, utils_logger

logger_name, log_file, opt_path = sys.argv[1:4]
with contextlib.redirect_stdout(io.StringIO()):
    opt = utils_option.parse(opt_path, is_train=True)
utils_logger.logger_info(logger_name, log_file, opt=opt, verbose=False)
for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue
    try:
        level, phase, mode, stream, message = raw.split("\t", 4)
    except ValueError:
        level, phase, mode, stream, message = "warning", "launch", None, "stderr", raw
    utils_logger.emit_launch_wrapper_log(
        logger_name, level, message,
        launch_stream=None if stream == "-" else stream,
        launch_phase=phase or None,
        launch_mode=mode or None,
    )
'

    LAUNCH_LOG_PIPE="$(mktemp -u /tmp/s-vrt-launch-logger.XXXXXX)"
    mkfifo "$LAUNCH_LOG_PIPE"
    "$PYTHON_BIN" -c "$daemon_code" "$logger_name" "$log_file" "$opt_path" < "$LAUNCH_LOG_PIPE" &
    LAUNCH_LOGGER_PID=$!
    exec 5>"$LAUNCH_LOG_PIPE"
}

cleanup_launch_logger() {
    local status=$?
    trap - EXIT
    if [[ -n "${LAUNCH_LOGGER_PID:-}" ]]; then
        exec 5>&- 2>/dev/null || true
        wait "$LAUNCH_LOGGER_PID" 2>/dev/null || true
    fi
    if [[ -n "${LAUNCH_LOG_PIPE:-}" ]]; then
        rm -f "$LAUNCH_LOG_PIPE"
    fi
    return "$status"
}

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

    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            launch_emit_record "$logger_name" "$launch_phase" "$launch_mode" "info" "stdout" "$line"
        fi
    done < "$stdout_pipe" &
    local stdout_reader_pid=$!

    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            launch_emit_record "$logger_name" "$launch_phase" "$launch_mode" "warning" "stderr" "$line"
        fi
    done < "$stderr_pipe" &
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

ensure_launch_logger() {
    local logger_name="$1"
    local log_dir="$2"
    local opt_path="$3"

    mkdir -p "$log_dir"
    "$PYTHON_BIN" - "$logger_name" "$log_dir" "$opt_path" <<'PY'
import sys, logging
from utils import utils_logger

logger_name, log_dir, opt_path = sys.argv[1:4]
utils_logger.logger_info(logger_name, f"{log_dir}/{logger_name}.log", opt=None,
    add_stream_handler=False, verbose=False)
log = logging.getLogger(logger_name)
for h in log.handlers:
    if hasattr(h, 'baseFilename'):
        print(h.baseFilename)
        break
PY
}

run_dependency_preparation() {
    ensure_python_package "snntorch" "snntorch"
    ensure_python_package "cv2" "opencv-python-headless"
    ensure_python_package_version "google.protobuf" ">=6.32.1" "protobuf>=6.32.1"
    ensure_python_package "wandb" "wandb"
    ensure_python_package "swanlab" "swanlab"
    ensure_dcnv4_module
}

# Resolve the training log directory from the config before any logging begins.
TRAIN_LOG_DIR="$(resolve_train_log_dir "$CONFIG_PATH")"
if [[ -z "$TRAIN_LOG_DIR" ]]; then
    echo "Error: could not resolve log dir from $CONFIG_PATH" >&2
    exit 1
fi
LAUNCH_LOG_FILE="$(ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH")"
start_launch_logger "train" "$LAUNCH_LOG_FILE" "$CONFIG_PATH"
trap cleanup_launch_logger EXIT
TENSORBOARD_LOGDIR="$(resolve_tensorboard_logdir "$CONFIG_PATH")"

launch_echo_lines "train" "launch" "local_single" "info" \
    "==========================================" \
    "VRT Training Launch Script" \
    "==========================================" \
    "Config: $CONFIG_PATH" \
    "Requested GPUs: $GPU_COUNT" \
    "GPU List: $GPU_LIST" \
    "GPU List Source: $GPU_LIST_SOURCE" \
    "Prepare Data: $PREPARE_DATA" \
    "Generate LMDB: $GENERATE_LMDB" \
    "Dataset Root: ${DATASET_ROOT:-<none>}" \
    "GoPro Root: $EFFECTIVE_GOPRO_ROOT" \
    "Spike Root: $EFFECTIVE_SPIKE_ROOT" \
    "Python: $PYTHON_BIN" \
    "AutoDL TensorBoard: $AUTODL_TENSORBOARD" \
    "TensorBoard Logdir: ${TENSORBOARD_LOGDIR:-<unresolved>}" \
    ""

if [[ "$AUTODL_TENSORBOARD" == true ]]; then
    ensure_autodl_tensorboard "$TENSORBOARD_LOGDIR"
fi

# ================================================================================
# Data Preparation (if requested)
# ================================================================================
if [[ "$PREPARE_DATA" == true ]]; then
    launch_echo_lines "train" "prepare" "local_single" "info" \
        "==========================================" \
        "Data Preparation Phase" \
        "==========================================" \
        "GoPro Root: $EFFECTIVE_GOPRO_ROOT" \
        "Spike Root: $EFFECTIVE_SPIKE_ROOT" \
        ""
    
    PREP_ARGS="--gopro_root $EFFECTIVE_GOPRO_ROOT --spike_root $EFFECTIVE_SPIKE_ROOT"
    if [[ -n "$DATASET_ROOT" ]]; then
        PREP_ARGS="$PREP_ARGS --dataset_root $DATASET_ROOT"
    fi
    
    if [[ "$GENERATE_LMDB" == true ]]; then
        PREP_ARGS="$PREP_ARGS --generate_lmdb"
        launch_echo "train" "prepare" "local_single" "info" "LMDB generation: ENABLED"
    else
        launch_echo "train" "prepare" "local_single" "info" "LMDB generation: DISABLED (use --generate-lmdb to enable)"
    fi
    
    if [[ "$FORCE_PREPARE" == true ]]; then
        PREP_ARGS="$PREP_ARGS --force"
        launch_echo "train" "prepare" "local_single" "info" "Force preparation: ENABLED"
    fi
    
    launch_echo_lines "train" "prepare" "local_single" "info" \
        "" \
        "Running data preparation script..." \
        "Command: python scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS" \
        ""
    
    run_with_wrapper "train" "prepare" "local_single" \
        "$PYTHON_BIN" scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS

    PREP_EXIT_CODE=$?
    launch_echo "train" "prepare" "local_single" "info" ""
    
    if [[ $PREP_EXIT_CODE -ne 0 ]]; then
        launch_echo_lines "train" "prepare" "local_single" "error" \
            "==========================================" \
            "Data preparation FAILED (exit code: $PREP_EXIT_CODE)" \
            "==========================================" \
            "" \
            "Please fix the data preparation issues before training." \
            "Check the error messages above for details."
        handle_error $PREP_EXIT_CODE "数据准备失败，请检查上面的错误信息。"
        # handle_error will exit with code 0 to keep terminal open
    fi
    
    launch_echo_lines "train" "prepare" "local_single" "info" \
        "==========================================" \
        "Data preparation completed successfully!" \
        "==========================================" \
        ""
    
    # Brief pause to let user see the summary
    sleep 2
fi

# ================================================================================
# Dependency Preparation
# ================================================================================
launch_echo_lines "train" "dependency" "local_single" "info" \
    "==========================================" \
    "Dependency Preparation" \
    "=========================================="
run_with_wrapper "train" "dependency" "local_single" \
    /bin/bash -lc "PYTHON_BIN='$PYTHON_BIN'; export CUDA_HOME='$CUDA_HOME'; export PATH='$PATH'; export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'; export TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST'; export PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF'; $(declare -f ensure_python_package); $(declare -f ensure_python_package_version); $(declare -f ensure_dcnv4_module); $(declare -f handle_error); run_dependency_preparation() { $(declare -f run_dependency_preparation | tail -n +2); }; run_dependency_preparation"
launch_echo "train" "dependency" "local_single" "info" ""

# ================================================================================
# Training Phase
# ================================================================================
launch_echo_lines "train" "train" "local_single" "info" \
    "==========================================" \
    "Training Phase" \
    "==========================================" \
    ""

# Create a temporary config with runtime-resolved dataset paths
RUNTIME_CONFIG="$CONFIG_PATH"
TMP_CONFIG=""
if [[ -n "$PYTHON_BIN" ]]; then
    TMP_CONFIG="$(mktemp /tmp/vrt_config.XXXXXX.json)"
    "$PYTHON_BIN" - "$CONFIG_PATH" "$TMP_CONFIG" "$EFFECTIVE_GOPRO_ROOT" "$EFFECTIVE_SPIKE_ROOT" <<'PYUPDATE'
import json, re, sys, pathlib
src, dst, gopro_root, spike_root = sys.argv[1:5]

def strip_json_comments(text: str) -> str:
    return re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)

with open(src, 'r') as f:
    cfg = json.loads(strip_json_comments(f.read()))
# Defensive: walk expected keys if present
ds = cfg.get('datasets', {})
train = ds.get('train', {})
test = ds.get('test', {})
if isinstance(train, dict):
    train['dataroot_gt'] = str(pathlib.Path(gopro_root) / 'train_GT')
    train['dataroot_lq'] = str(pathlib.Path(gopro_root) / 'train_GT_blurred')
    if 'dataroot_spike' in train or True:
        train['dataroot_spike'] = str(pathlib.Path(spike_root) / 'train')
if isinstance(test, dict):
    test['dataroot_gt'] = str(pathlib.Path(gopro_root) / 'test_GT')
    test['dataroot_lq'] = str(pathlib.Path(gopro_root) / 'test_GT_blurred')
with open(dst, 'w') as f:
    json.dump(cfg, f, indent=2)
print(dst)
PYUPDATE
    if [[ -s "$TMP_CONFIG" ]]; then
        RUNTIME_CONFIG="$TMP_CONFIG"
        launch_echo "train" "train" "local_single" "info" "Using runtime config: $RUNTIME_CONFIG"
    else
        launch_echo "train" "train" "local_single" "warning" "Warning: Failed to materialize runtime config; falling back to original."
        rm -f "$TMP_CONFIG"
        TMP_CONFIG=""
    fi
else
    launch_echo "train" "train" "local_single" "warning" "Warning: Python not found to rewrite config paths; using original config."
fi

# Check if we're in platform DDP mode
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE:-0}" -gt 1 ]]; then
    # ========================================
    # Platform DDP Mode
    # ========================================
    launch_echo_lines "train" "train" "platform_ddp" "info" \
        "Platform DDP detected:" \
        "  RANK=$RANK" \
        "  LOCAL_RANK=$LOCAL_RANK" \
        "  WORLD_SIZE=$WORLD_SIZE" \
        "  MASTER_ADDR=$MASTER_ADDR" \
        "  MASTER_PORT=$MASTER_PORT" \
        "" \
        "Running: $PYTHON_BIN -u main_train_vrt.py --opt $RUNTIME_CONFIG" \
        "=========================================="
    
    # The training process owns the train logger; do not wrap it again here.
    "$PYTHON_BIN" -u main_train_vrt.py --opt "$RUNTIME_CONFIG"

else
    # ========================================
    # Local/Self-managed Mode
    # ========================================
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        # Multi-GPU: use torchrun
        launch_echo_lines "train" "train" "local_multi" "info" \
            "Local training mode" \
            "Multi-GPU training with torchrun" \
            "  GPUs: $GPU_COUNT" \
            "  CUDA_VISIBLE_DEVICES: $GPU_LIST" \
            "" \
            "Running: $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$GPU_COUNT main_train_vrt.py --opt $RUNTIME_CONFIG" \
            "=========================================="
        
        env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
            "$PYTHON_BIN" -m torch.distributed.run \
            --nproc_per_node="$GPU_COUNT" \
            --standalone \
            main_train_vrt.py --opt "$RUNTIME_CONFIG"
    else
        # Single GPU: plain python
        SINGLE_GPU_ID="${GPU_ID_ARRAY[0]}"
        launch_echo_lines "train" "train" "local_single" "info" \
            "Local training mode" \
            "Single GPU training" \
            "  CUDA_VISIBLE_DEVICES: $SINGLE_GPU_ID" \
            "" \
            "Running: $PYTHON_BIN main_train_vrt.py --opt $RUNTIME_CONFIG" \
            "=========================================="

        env CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
            "$PYTHON_BIN" main_train_vrt.py --opt "$RUNTIME_CONFIG"
    fi
fi

EXIT_CODE=$?
launch_echo_lines "train" "launch" "local_single" "info" "" "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    launch_echo_lines "train" "launch" "local_single" "info" \
        "Training completed successfully" \
        "=========================================="
    # Success - exit normally
    exit 0
else
    launch_echo_lines "train" "launch" "local_single" "error" \
        "Training exited with code: $EXIT_CODE" \
        "=========================================="
    # Use handle_error to show error and keep terminal open
    handle_error $EXIT_CODE "训练失败，请检查上面的错误信息。"
    # handle_error will exit with code 0 to keep terminal open
fi
