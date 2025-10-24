#!/bin/bash
# CUDA Environment Diagnostic Script

echo "=== CUDA Environment Diagnostic ==="
echo ""

echo "1. NVIDIA Driver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "  ❌ nvidia-smi not available"
echo ""

echo "2. CUDA Driver Version (from nvidia-smi):"
nvidia-smi | grep "CUDA Version" || echo "  ❌ Could not detect CUDA version"
echo ""

echo "3. Conda Environment:"
echo "  Active environment: $CONDA_DEFAULT_ENV"
conda list | grep -E "(pytorch|cuda|nccl)" | head -20
echo ""

echo "4. PyTorch CUDA Information:"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version (PyTorch): {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
echo ""

echo "5. NCCL Test:"
python -c "
import torch
import torch.distributed as dist
print('  Testing NCCL availability...')
try:
    if dist.is_nccl_available():
        print('  ✓ NCCL is available')
    else:
        print('  ❌ NCCL is not available')
except Exception as e:
    print(f'  ❌ NCCL test failed: {e}')
" 2>&1
echo ""

echo "6. Recommended Actions:"
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -z "$DRIVER_VERSION" ]; then
    echo "  ❌ Cannot detect NVIDIA driver. Please install NVIDIA drivers."
else
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    echo "  Current driver: $DRIVER_VERSION (major: $DRIVER_MAJOR)"
    
    if [ "$DRIVER_MAJOR" -lt "520" ]; then
        echo "  ⚠️  Driver version is old. Recommend updating to 520+ for CUDA 12.x"
        echo "     Run: sudo apt update && sudo apt install nvidia-driver-535"
    else
        echo "  ✓ Driver version should be compatible with CUDA 12.x"
    fi
fi
echo ""

echo "7. Current NCCL Environment Variables:"
env | grep -E "(NCCL|CUDA)" | sort
echo ""

echo "=== Diagnostic Complete ==="



