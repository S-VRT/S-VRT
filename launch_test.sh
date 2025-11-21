#!/bin/bash

# ================================================================================
# Modern Test Launch Script for VRT/S-VRT Inference
# ================================================================================
# This script mirrors the training launcher to provide a consistent entrypoint for
# running `main_test_vrt.py` with locally stored datasets and optional data
# preparation helpers.
# ================================================================================

# Default configuration (shared with training)
DEFAULT_CONFIG="options/vrt/gopro_rgbspike_local.json"

# Parsed arguments
GPU_COUNT=""
CONFIG_PATH=""
PREPARE_DATA=false
GENERATE_LMDB=false
FORCE_PREPARE=false
DATASET_ROOT=""
OVERRIDE_GOPRO_ROOT=""
OVERRIDE_SPIKE_ROOT=""
GPU_LIST=""

usage() {
    cat <<EOF
Usage: $0 GPU_COUNT [CONFIG_PATH] [options]

Positional arguments:
  GPU_COUNT        Number of GPUs requested (required unless --gpus provided)
  CONFIG_PATH      Optional path to training/test JSON config (default: $DEFAULT_CONFIG)

Options:
  --prepare-data                 Run GoPro+Spike data preparation helper
  --generate-lmdb                Generate LMDB during preparation (requires --prepare-data)
  --force-prepare                Force re-running preparation
  --dataset-root PATH            Root folder that contains GOPRO_Large and GOPRO_Large_spike_seq
  --gopro-root PATH              Override GoPro root explicitly
  --spike-root PATH              Override Spike root explicitly
  --gpus 0,1,2                   Comma separated GPU ids (default mirrors GPU_COUNT)
  --help | -h                    Show this message
EOF
}

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
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
            DATASET_ROOT="${1#*=}"
            shift
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --gopro-root=*)
            OVERRIDE_GOPRO_ROOT="${1#*=}"
            shift
            ;;
        --gopro-root)
            OVERRIDE_GOPRO_ROOT="$2"
            shift 2
            ;;
        --spike-root=*)
            OVERRIDE_SPIKE_ROOT="${1#*=}"
            shift
            ;;
        --spike-root)
            OVERRIDE_SPIKE_ROOT="$2"
            shift 2
            ;;
        --gpus=*)
            GPU_LIST="${1#*=}"
            shift
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$GPU_COUNT" && "$1" =~ ^[0-9]+$ ]]; then
                GPU_COUNT="$1"
                shift
            elif [[ -z "$CONFIG_PATH" && ( -f "$1" || "$1" == *.json ) ]]; then
                CONFIG_PATH="$1"
                shift
            else
                echo "Unknown argument: $1"
                usage
                exit 1
            fi
            ;;
    esac
done

# Defaults
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG}

if [[ -z "$GPU_COUNT" && -z "$GPU_LIST" ]]; then
    echo "Error: GPU_COUNT positional argument or --gpus must be provided."
    exit 1
fi

if [[ -n "$GPU_COUNT" && -z "$GPU_LIST" ]]; then
    if [[ "$GPU_COUNT" -le 0 ]]; then
        echo "Error: GPU_COUNT must be a positive integer."
        exit 1
    fi
    if [[ "$GPU_COUNT" -eq 1 ]]; then
        GPU_LIST="0"
    else
        GPU_LIST=$(seq -s, 0 $((GPU_COUNT - 1)))
    fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

CONFIG_VARS=$(python - "$CONFIG_PATH" <<'PY'
import json, pathlib, shlex, sys
from main_test_vrt import _strip_json_comments

cfg_path = pathlib.Path(sys.argv[1])
text = cfg_path.read_text(encoding='utf-8')
cfg = json.loads(_strip_json_comments(text))

datasets = cfg.get('datasets', {}) or {}
test_ds = datasets.get('test', {}) or {}
val_cfg = cfg.get('val', {}) or {}

task = val_cfg.get('task_name') or cfg.get('task') or ''
test_lq = test_ds.get('dataroot_lq') or ''
test_gt = test_ds.get('dataroot_gt') or ''

def get_parent(path_str):
    if not path_str:
        return ''
    return str(pathlib.Path(path_str).expanduser().absolute().parent)

defaults = {
    "TASK_FROM_CONFIG": task,
    "TEST_LQ_FROM_CONFIG": test_lq,
    "TEST_GT_FROM_CONFIG": test_gt,
    "CONFIG_GOPRO_ROOT": get_parent(test_gt) if test_gt else (get_parent(test_lq) if test_lq else ''),
    "CONFIG_SPIKE_ROOT": get_parent(test_ds.get('dataroot_spike') or ''),
}

for key, value in defaults.items():
    if value is None:
        value = ''
    print(f"{key}={shlex.quote(str(value))}")
PY
)

if [[ $? -ne 0 ]]; then
    echo "Failed to parse config: $CONFIG_PATH"
    exit 1
fi

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    eval "$line"
done <<< "$CONFIG_VARS"

if [[ -z "$GPU_LIST" ]]; then
    if [[ "$GPU_COUNT" -le 1 ]]; then
        GPU_LIST="0"
    else
        GPU_LIST=$(seq -s, 0 $((GPU_COUNT - 1)))
    fi
fi

GPU_LIST=$(echo "$GPU_LIST" | tr -d '[:space:]')
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_LIST"
GPU_COUNT_FROM_LIST=${#GPU_ID_ARRAY[@]}
if [[ "$GPU_COUNT_FROM_LIST" -eq 0 ]]; then
    echo "Invalid GPU list specified via --gpus (value: '$GPU_LIST')."
    exit 1
fi

if [[ -z "$GPU_COUNT" ]]; then
    GPU_COUNT=$GPU_COUNT_FROM_LIST
elif [[ "$GPU_COUNT" -ne "$GPU_COUNT_FROM_LIST" ]]; then
    echo "Info: GPU_COUNT ($GPU_COUNT) and GPU list size ($GPU_COUNT_FROM_LIST) differ."
    echo "      Using GPU list size for execution."
    GPU_COUNT=$GPU_COUNT_FROM_LIST
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
    export MASTER_PORT=12355
fi

# Resolve dataset roots
EFFECTIVE_GOPRO_ROOT="${CONFIG_GOPRO_ROOT:-}"
EFFECTIVE_SPIKE_ROOT="${CONFIG_SPIKE_ROOT:-}"

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


# NCCL / CUDA stability vars
export NCCL_ASYNC_ERROR_HANDLING=1

# Helper to keep terminal open on error
handle_error() {
    local exit_code=$1
    local message="${2:-}"

    if [[ $exit_code -ne 0 ]]; then
        echo ""
        [[ -n "$message" ]] && echo "$message" && echo ""
        echo "错误信息已显示在上方。"
        if [[ -t 1 ]]; then
            echo "按 Enter 键继续（脚本将结束，但终端保持打开）..."
            read -r _
        fi
        exit 0
    fi
}

echo "=========================================="
echo "VRT Testing Launch Script"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Task (from config): ${TASK_FROM_CONFIG:-<unknown>}"
echo "Requested GPUs: $GPU_COUNT"
echo "GPU List: $GPU_LIST"
echo "Prepare Data: $PREPARE_DATA"
echo "Generate LMDB: $GENERATE_LMDB"
echo "Dataset Root Override: ${DATASET_ROOT:-<none>}"
echo "GoPro Root: $EFFECTIVE_GOPRO_ROOT"
echo "Spike Root: $EFFECTIVE_SPIKE_ROOT"
echo "Configured LQ Folder: ${TEST_LQ_FROM_CONFIG:-<none>}"
echo "Configured GT Folder: ${TEST_GT_FROM_CONFIG:-<none>}"
echo ""

# ================================================================================
# Data Preparation (optional)
# ================================================================================
if [[ "$PREPARE_DATA" == true ]]; then
    if [[ -z "$EFFECTIVE_GOPRO_ROOT" || -z "$EFFECTIVE_SPIKE_ROOT" ]]; then
        echo "Error: Data preparation requires GoPro and Spike roots via config or overrides."
        exit 1
    fi
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

    python scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS
    PREP_EXIT_CODE=$?
    echo ""

    if [[ $PREP_EXIT_CODE -ne 0 ]]; then
        echo "=========================================="
        echo "Data preparation FAILED (exit code: $PREP_EXIT_CODE)"
        echo "=========================================="
        handle_error $PREP_EXIT_CODE "数据准备失败，请检查上面的错误信息。"
    fi

    echo "=========================================="
    echo "Data preparation completed successfully!"
    echo "=========================================="
    echo ""
    sleep 2
fi

# ================================================================================
# Testing Phase
# ================================================================================
echo "=========================================="
echo "Testing Phase"
echo "=========================================="
echo ""

# Create a runtime config with overridden dataset paths (if needed)
RUNTIME_CONFIG="$CONFIG_PATH"
TMP_CONFIG=""
if command -v python >/dev/null 2>&1 && [[ -n "$EFFECTIVE_GOPRO_ROOT" || -n "$EFFECTIVE_SPIKE_ROOT" ]]; then
    TMP_CONFIG="$(mktemp /tmp/vrt_test_config.XXXXXX.json)"
    python - "$CONFIG_PATH" "$TMP_CONFIG" "$EFFECTIVE_GOPRO_ROOT" "$EFFECTIVE_SPIKE_ROOT" <<'PYUPDATE'
import json, sys, pathlib
from main_test_vrt import _strip_json_comments
src, dst, gopro_root, spike_root = sys.argv[1:5]
with open(src, 'r', encoding='utf-8') as f:
    cfg = json.loads(_strip_json_comments(f.read()))
ds = cfg.get('datasets', {})
train = ds.get('train', {})
test = ds.get('test', {})
if isinstance(train, dict) and gopro_root:
    base = pathlib.Path(gopro_root)
    train['dataroot_gt'] = str(base / 'train_GT')
    train['dataroot_lq'] = str(base / 'train_GT_blurred')
if isinstance(train, dict) and spike_root:
    train['dataroot_spike'] = str(pathlib.Path(spike_root) / 'train')
if isinstance(test, dict) and gopro_root:
    base = pathlib.Path(gopro_root)
    test['dataroot_gt'] = str(base / 'test_GT')
    test['dataroot_lq'] = str(base / 'test_GT_blurred')
if isinstance(test, dict) and spike_root:
    test['dataroot_spike'] = str(pathlib.Path(spike_root) / 'test')
with open(dst, 'w', encoding='utf-8') as f:
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

# ================================================================================
# Execution (DDP-aware, mirroring launch_train.sh)
# ================================================================================
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE:-0}" -gt 1 ]]; then
    echo "Platform DDP detected:"
    echo "  RANK=${RANK:-<unset>}"
    echo "  LOCAL_RANK=${LOCAL_RANK:-<unset>}"
    echo "  WORLD_SIZE=$WORLD_SIZE"
    echo "  MASTER_ADDR=${MASTER_ADDR:-<unset>}"
    echo "  MASTER_PORT=${MASTER_PORT:-<unset>}"
    echo ""
    echo "Running: python -u main_test_vrt.py --opt $RUNTIME_CONFIG"
    echo "=========================================="

    python -u main_test_vrt.py --opt "$RUNTIME_CONFIG"
else
    echo "Local testing mode"
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        echo "Multi-GPU inference with torchrun"
        echo "  GPUs: $GPU_COUNT"
        echo "  CUDA_VISIBLE_DEVICES: $GPU_LIST"
        echo ""
        echo "Running: torchrun --nproc_per_node=$GPU_COUNT main_test_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES="$GPU_LIST" \
        torchrun \
            --nproc_per_node="$GPU_COUNT" \
            --standalone \
            main_test_vrt.py --opt "$RUNTIME_CONFIG"
    else
        echo "Single GPU inference"
        SINGLE_GPU_ID="${GPU_ID_ARRAY[0]}"
        echo "  CUDA_VISIBLE_DEVICES: $SINGLE_GPU_ID"
        echo ""
        echo "Running: python main_test_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES="$SINGLE_GPU_ID" python main_test_vrt.py --opt "$RUNTIME_CONFIG"
    fi
fi

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Testing completed successfully"
    echo "=========================================="
    exit 0
else
    echo "Testing exited with code: $EXIT_CODE"
    echo "=========================================="
    handle_error $EXIT_CODE "测试失败，请检查上面的错误信息。"
fi
