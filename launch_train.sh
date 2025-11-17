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
DEFAULT_CONFIG="options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike_local.json"
DEFAULT_GOPRO_ROOT="/media/mallm/hd4t/modelrepostore/datasets/gopro_small/GOPRO_Large"
DEFAULT_SPIKE_ROOT="/media/mallm/hd4t/modelrepostore/datasets/gopro_small/GOPRO_Large_spike_seq"

# Parse arguments
GPU_COUNT=""
CONFIG_PATH=""
PREPARE_DATA=false
GENERATE_LMDB=false
FORCE_PREPARE=false
DATASET_ROOT=""
OVERRIDE_GOPRO_ROOT=""
OVERRIDE_SPIKE_ROOT=""

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

# Set defaults if not provided
GPU_COUNT=${GPU_COUNT:-1}
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG}

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

echo "=========================================="
echo "VRT Training Launch Script"
echo "=========================================="
echo "Config: $CONFIG_PATH"
echo "Requested GPUs: $GPU_COUNT"
echo "Prepare Data: $PREPARE_DATA"
echo "Generate LMDB: $GENERATE_LMDB"
echo "Dataset Root: ${DATASET_ROOT:-<none>}"
echo "GoPro Root: $EFFECTIVE_GOPRO_ROOT"
echo "Spike Root: $EFFECTIVE_SPIKE_ROOT"
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
    
    python scripts/data_preparation/prepare_gopro_spike_dataset.py $PREP_ARGS
    
    PREP_EXIT_CODE=$?
    echo ""
    
    if [[ $PREP_EXIT_CODE -ne 0 ]]; then
        echo "=========================================="
        echo "Data preparation FAILED (exit code: $PREP_EXIT_CODE)"
        echo "=========================================="
        echo ""
        echo "Please fix the data preparation issues before training."
        echo "Check the error messages above for details."
        exit $PREP_EXIT_CODE
    fi
    
    echo "=========================================="
    echo "Data preparation completed successfully!"
    echo "=========================================="
    echo ""
    
    # Brief pause to let user see the summary
    sleep 2
fi

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
if command -v python >/dev/null 2>&1; then
    TMP_CONFIG="$(mktemp /tmp/vrt_config.XXXXXX.json)"
    python - "$CONFIG_PATH" "$TMP_CONFIG" "$EFFECTIVE_GOPRO_ROOT" "$EFFECTIVE_SPIKE_ROOT" <<'PYUPDATE'
import json, sys, pathlib
src, dst, gopro_root, spike_root = sys.argv[1:5]
p = pathlib.Path(src)
with open(src, 'r') as f:
    cfg = json.load(f)
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
    echo "Running: python -u main_train_vrt.py --opt $RUNTIME_CONFIG"
    echo "=========================================="
    
    # Platform has already set up environment, just run python directly
    python -u main_train_vrt.py --opt "$RUNTIME_CONFIG"

else
    # ========================================
    # Local/Self-managed Mode
    # ========================================
    echo "Local training mode"
    
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        # Multi-GPU: use torchrun
        echo "Multi-GPU training with torchrun"
        echo "  GPUs: $GPU_COUNT"
        echo ""
        echo "Running: torchrun --nproc_per_node=$GPU_COUNT main_train_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="
        
        torchrun \
            --nproc_per_node="$GPU_COUNT" \
            --standalone \
            main_train_vrt.py --opt "$RUNTIME_CONFIG"
    else
        # Single GPU: plain python
        echo "Single GPU training"
        echo ""
        echo "Running: python main_train_vrt.py --opt $RUNTIME_CONFIG"
        echo "=========================================="
        
        python main_train_vrt.py --opt "$RUNTIME_CONFIG"
    fi
fi

EXIT_CODE=$?
echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Training completed successfully"
else
    echo "Training exited with code: $EXIT_CODE"
    if [[ -t 1 ]]; then
        echo ""
        echo "Training failed. Press Enter to close this window..."
        read -r _
    fi
fi
echo "=========================================="

exit $EXIT_CODE
