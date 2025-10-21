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
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available (set to 1 to disable)
export NCCL_DEBUG=INFO  # Set to WARN for less verbose output

# OpenMP threading for CPU operations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0

# Number of GPUs to use
NUM_GPUS=3

# Log file with timestamp
LOG_FILE="outputs/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training with $NUM_GPUS GPUs..."
echo "Log file: $LOG_FILE"

# Use torchrun for distributed training (replaces python -m torch.distributed.launch)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    src/train.py \
    --config configs/deblur/vrt_spike_baseline.yaml \
    2>&1 | tee "$LOG_FILE"

echo "Training completed. Log saved to $LOG_FILE"

