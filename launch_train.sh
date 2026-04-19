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
#   ./launch_train.sh 1                                # Single GPU, default config
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
DEFAULT_GPU_COUNT=1

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
        --help|-h)
            echo "Usage: $0 [GPU_COUNT] [CONFIG_PATH] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  GPU_COUNT        Number of GPUs to use (default: 1)"
            echo "  CONFIG_PATH      Path to training config (default: $DEFAULT_CONFIG)"
            echo ""
            echo "Options:"
            echo "  --prepare-data   Prepare GoPro + Spike dataset before training"
            echo "  --generate-lmdb  Generate LMDB files (requires --prepare-data)"
            echo "  --force-prepare  Force re-preparation even if data exists"
            echo "  --dataset-root=/path/to/gopro_spike           Root where zip was extracted"
            echo "  --gopro-root=/path/to/GOPRO_Large             Override GoPro root"
            echo "  --spike-root=/path/to/GOPRO_Large_spike_seq   Override Spike root"
            echo "  --gpus=0,1,2     Comma-separated GPU ids for single-node DDP (default: 0,1,2)"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 1 --prepare-data"
            echo "  $0 4 --prepare-data --generate-lmdb"
            echo "  $0 8 options/vrt/custom.json"
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

# Set defaults if not provided (single-node 3-GPU by default)
GPU_COUNT=${GPU_COUNT:-$DEFAULT_GPU_COUNT}
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG}
if [[ -z "$GPU_LIST" ]]; then
    if [[ "$GPU_COUNT" -le 1 ]]; then
        GPU_LIST="0"
    else
        # Generate comma-separated list 0,1,...,GPU_COUNT-1
        GPU_LIST=$(seq -s, 0 $((GPU_COUNT - 1)))
    fi
fi

# Normalize GPU list (remove spaces) and derive effective GPU count
GPU_LIST=$(echo "$GPU_LIST" | tr -d '[:space:]')
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
GPU_COUNT_FROM_LIST=${#GPU_ID_ARRAY[@]}
if [[ "$GPU_COUNT_FROM_LIST" -eq 0 ]]; then
    echo "Invalid GPU list specified via --gpus (value: '$GPU_LIST')."
    exit 1
fi

if [[ "$GPU_COUNT" -ne "$GPU_COUNT_FROM_LIST" ]]; then
    echo "Info: GPU_COUNT ($GPU_COUNT) and GPU list size ($GPU_COUNT_FROM_LIST) differ."
    echo "      Using GPU list size for torchrun."
    GPU_COUNT=$GPU_COUNT_FROM_LIST
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT=12355
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

echo "=========================================="
echo "VRT Training Launch Script"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Requested GPUs: $GPU_COUNT"
echo "GPU List: $GPU_LIST"
echo "Prepare Data: $PREPARE_DATA"
echo "Generate LMDB: $GENERATE_LMDB"
echo "Dataset Root: ${DATASET_ROOT:-<none>}"
echo "GoPro Root: $EFFECTIVE_GOPRO_ROOT"
echo "Spike Root: $EFFECTIVE_SPIKE_ROOT"
echo "Python: $PYTHON_BIN"
echo ""

# ================================================================================
# Data Preparation (if requested)
# ================================================================================
if [[ "$PREPARE_DATA" == true ]]; then
    echo "=========================================="
    echo "Data Preparation Phase"
    echo "=========================================="
    echo "GoPro Root: $EFFECTIVE_GOPRO_ROOT"
    echo "Spike Root: $EFFECTIVE_SPIKE_ROOT"
    echo ""
    
    PREP_ARGS="--gopro_root $EFFECTIVE_GOPRO_ROOT --spike_root $EFFECTIVE_SPIKE_ROOT"
    if [[ -n "$DATASET_ROOT" ]]; then
        PREP_ARGS="$PREP_ARGS --dataset_root $DATASET_ROOT"
    fi
    
    if [[ "$GENERATE_LMDB" == true ]]; then
        PREP_ARGS="$PREP_ARGS --generate_lmdb"
        echo "LMDB generation: ENABLED"
    else
        echo "LMDB generation: DISABLED (use --generate-lmdb to enable)"
    fi
    
    if [[ "$FORCE_PREPARE" == true ]]; then
        PREP_ARGS="$PREP_ARGS --force"
        echo "Force preparation: ENABLED"
    fi
    
    echo ""
    echo "Running data preparation script..."
    echo "Command: python scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS"
    echo ""
    
    run_with_wrapper "train" "prepare" "local_single" \
        "$PYTHON_BIN" scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS

    PREP_EXIT_CODE=$?
    echo ""
    
    if [[ $PREP_EXIT_CODE -ne 0 ]]; then
        echo "=========================================="
        echo "Data preparation FAILED (exit code: $PREP_EXIT_CODE)"
        echo "=========================================="
        echo ""
        echo "Please fix the data preparation issues before training."
        echo "Check the error messages above for details."
        handle_error $PREP_EXIT_CODE "数据准备失败，请检查上面的错误信息。"
        # handle_error will exit with code 0 to keep terminal open
    fi
    
    echo "=========================================="
    echo "Data preparation completed successfully!"
    echo "=========================================="
    echo ""
    
    # Brief pause to let user see the summary
    sleep 2
fi

# ================================================================================
# Dependency Preparation
# ================================================================================
echo "=========================================="
echo "Dependency Preparation"
echo "=========================================="
ensure_python_package "snntorch" "snntorch"
ensure_python_package "cv2" "opencv-python-headless"
ensure_python_package_version "google.protobuf" ">=6.32.1" "protobuf>=6.32.1"
ensure_python_package "wandb" "wandb"
ensure_python_package "swanlab" "swanlab"
ensure_dcnv4_module
echo ""

# ================================================================================
# Training Phase
# ================================================================================
echo "=========================================="
echo "Training Phase"
echo "=========================================="
echo ""

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
        echo "Using runtime config: $RUNTIME_CONFIG"
    else
        echo "Warning: Failed to materialize runtime config; falling back to original."
        rm -f "$TMP_CONFIG"
        TMP_CONFIG=""
    fi
else
    echo "Warning: Python not found to rewrite config paths; using original config."
fi

TRAIN_LOG_DIR="$(dirname "$RUNTIME_CONFIG")"
ensure_launch_logger "train" "$TRAIN_LOG_DIR" "$CONFIG_PATH"

# Check if we're in platform DDP mode
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE:-0}" -gt 1 ]]; then
    # ========================================
    # Platform DDP Mode
    # ========================================
    echo "Platform DDP detected:"
    echo "  RANK=$RANK"
    echo "  LOCAL_RANK=$LOCAL_RANK"
    echo "  WORLD_SIZE=$WORLD_SIZE"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo ""
    echo "Running: $PYTHON_BIN -u main_train_vrt.py --opt $RUNTIME_CONFIG"
    echo "=========================================="
    
    # Platform has already set up environment, just run python directly
    run_with_wrapper "train" "train" "platform_ddp" \
        "$PYTHON_BIN" -u main_train_vrt.py --opt "$RUNTIME_CONFIG"

else
    # ========================================
    # Local/Self-managed Mode
    # ========================================
    echo "Local training mode"
    
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        # Multi-GPU: use torchrun
        echo "Multi-GPU training with torchrun"
        echo "  GPUs: $GPU_COUNT"
        echo "  CUDA_VISIBLE_DEVICES: $GPU_LIST"
        echo ""
        echo "Running: $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$GPU_COUNT main_train_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="
        
        run_with_wrapper "train" "train" "local_multi" \
            env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
            "$PYTHON_BIN" -m torch.distributed.run \
                --nproc_per_node="$GPU_COUNT" \
                --standalone \
                main_train_vrt.py --opt "$RUNTIME_CONFIG"
    else
        # Single GPU: plain python
        echo "Single GPU training"
        SINGLE_GPU_ID="${GPU_ID_ARRAY[0]}"
        echo "  CUDA_VISIBLE_DEVICES: $SINGLE_GPU_ID"
        echo ""
        echo "Running: $PYTHON_BIN main_train_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="

        run_with_wrapper "train" "train" "local_single" \
            env CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" \
            "$PYTHON_BIN" main_train_vrt.py --opt "$RUNTIME_CONFIG"
    fi
fi

EXIT_CODE=$?
echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Training completed successfully"
    echo "=========================================="
    # Success - exit normally
    exit 0
else
    echo "Training exited with code: $EXIT_CODE"
    echo "=========================================="
    # Use handle_error to show error and keep terminal open
    handle_error $EXIT_CODE "训练失败，请检查上面的错误信息。"
    # handle_error will exit with code 0 to keep terminal open
fi
