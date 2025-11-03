#!/bin/bash
set -e

# Activate conda environment
conda activate vrtspike

# Set PYTHONPATH to project root (CRITICAL for imports)
# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# NCCL environment variables (optional for testing, but included for consistency)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Async error handling
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available (set to 1 to disable)
export NCCL_DEBUG=WARN  # Set to INFO for verbose output, WARN for production
export NCCL_NVML_DISABLE=1  # Disable NVML to avoid driver/library version mismatch

# OpenMP threading for CPU operations
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0

# Number of GPUs to use (testing typically uses 1)
NUM_GPUS=1

# Log file with timestamp
LOG_FILE="outputs/logs/test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting testing with $NUM_GPUS GPU(s)..."
echo "Log file: $LOG_FILE"

# For testing, use python directly (no distributed needed)
# Pass any additional arguments to the script
python src/test.py \
    --config configs/deblur/vrt_spike_baseline.yaml \
    "$@" \
    2>&1 | tee "$LOG_FILE"

echo "Testing completed. Log saved to $LOG_FILE"
