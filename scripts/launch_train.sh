#!/bin/bash
set -e

# Activate conda environment
conda activate vrtspike

# Set PYTHONPATH to project root (CRITICAL for imports)
# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# NCCL environment variables for better multi-GPU performance
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Async error handling
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available (set to 1 to disable)
export NCCL_DEBUG=WARN  # Set to INFO for verbose output, WARN for production
export NCCL_NVML_DISABLE=1  # Disable NVML to avoid driver/library version mismatch

# OpenMP threading for CPU operations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0

# PyTorch CUDA memory allocator configuration
# expandable_segments:True reduces memory fragmentation by allowing PyTorch to expand allocated segments
# This is critical for preventing OOM during validation with large images
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Number of GPUs to use
NUM_GPUS=3

# Parse arguments for --config and --exp_dir
CONFIG="configs/deblur/vrt_spike_baseline.yaml"  # Default
EXP_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --exp_dir)
            EXP_DIR="$2"
            shift 2
            ;;
        *)
            # Collect other args
            OTHER_ARGS="$OTHER_ARGS $1"
            shift
            ;;
    esac
done

# Restore other args
set -- $OTHER_ARGS

# Determine exp_dir
if [ -n "$EXP_DIR" ]; then
    EXP_DIR_ABS="$(realpath "$EXP_DIR")"
    echo "Resuming from provided exp_dir: $EXP_DIR_ABS"
else
    CONFIG_NAME="$(basename "$CONFIG" .yaml)"
    RUN_NAME="$(date +%Y%m%d_%H%M%S)"
    BASE_DIR="$PROJECT_ROOT/outputs/$CONFIG_NAME"
    EXP_DIR_ABS="$BASE_DIR/$RUN_NAME"
    mkdir -p "$EXP_DIR_ABS"
    # CRITICAL FIX: Set EXP_DIR to EXP_DIR_ABS so it gets passed to train.py
    EXP_DIR="$EXP_DIR_ABS"
    echo "Created new experiment directory: $EXP_DIR_ABS"
fi

# Log file in exp_dir/logs (use the same RUN_NAME, don't call date again)
mkdir -p "$EXP_DIR_ABS/logs"
LOG_FILE="$EXP_DIR_ABS/logs/train_$(basename "$EXP_DIR_ABS").log"

echo "Starting training with $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"

# Launch with torchrun, pass --exp_dir if set
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    src/train.py \
    --config "$CONFIG" \
    ${EXP_DIR:+"--exp_dir" "$EXP_DIR"} \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo "Training completed. Log saved to $LOG_FILE"

