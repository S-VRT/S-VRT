#!/bin/bash
# Resume training from a saved checkpoint
# 从保存的检查点恢复训练

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrtspike

# Setup environment
source setup_env.sh

# Configuration
CONFIG_FILE="configs/deblur/vrt_spike_baseline.yaml"
CHECKPOINT_PATH="outputs/ckpts/last.pth"  # 默认使用最新检查点

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --best)
            CHECKPOINT_PATH="outputs/ckpts/best.pth"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config CONFIG_FILE] [--checkpoint CHECKPOINT_PATH] [--best]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "恢复训练 (Resume Training)"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "检查点: $CHECKPOINT_PATH"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误: 检查点文件不存在: $CHECKPOINT_PATH"
    echo ""
    echo "可用的检查点:"
    ls -lh outputs/ckpts/*.pth 2>/dev/null || echo "  (没有找到检查点)"
    exit 1
fi

# Show checkpoint info
echo ""
echo "检查点信息:"
python -c "
import torch
ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=False)
print(f'  Step: {ckpt.get(\"step\", \"N/A\")}')
print(f'  Epoch: {ckpt.get(\"epoch\", \"N/A\")}')
print(f'  Best PSNR: {ckpt.get(\"best_psnr\", \"N/A\")}')
print(f'  包含优化器状态: {\"optimizer\" in ckpt}')
print(f'  包含调度器状态: {\"scheduler\" in ckpt}')
"
echo ""

# Check GPU status
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s (%.1f/%.1f GB)\n", $1, $2, $3/1024, $4/1024}'
echo ""

# Confirm before proceeding
read -p "确认继续训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始恢复训练..."
echo ""

# Run training with resume
python src/train.py \
    --config "$CONFIG_FILE" \
    --resume "$CHECKPOINT_PATH" \
    --device cuda

echo ""
echo "训练完成!"

