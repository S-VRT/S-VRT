#!/bin/bash
# NVIDIA 驱动模块热重载脚本（无需重启）

set -e

echo "=== NVIDIA 驱动版本不匹配修复工具 ==="
echo ""

# 检查是否以 root 权限运行
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 错误: 此脚本需要 root 权限"
    echo "请使用: sudo bash $0"
    exit 1
fi

echo "步骤 1: 检查当前状态"
echo "----------------------------------------"
nvidia-smi 2>&1 | head -5 || echo "nvidia-smi 当前无法使用（预期）"
echo ""

echo "步骤 2: 查找使用 GPU 的进程"
echo "----------------------------------------"
GPU_PROCESSES=$(lsof /dev/nvidia* 2>/dev/null | grep -v "COMMAND" | awk '{print $2}' | sort -u)

if [ -n "$GPU_PROCESSES" ]; then
    echo "⚠️  警告: 检测到以下进程正在使用 GPU:"
    ps -p $GPU_PROCESSES -o pid,user,cmd 2>/dev/null || echo "$GPU_PROCESSES"
    echo ""
    read -p "是否终止这些进程并继续? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在终止 GPU 进程..."
        kill -9 $GPU_PROCESSES 2>/dev/null || true
        sleep 2
    else
        echo "❌ 取消操作。请手动停止 GPU 进程后重试。"
        exit 1
    fi
else
    echo "✓ 没有检测到正在使用 GPU 的进程"
fi
echo ""

echo "步骤 3: 卸载 NVIDIA 内核模块"
echo "----------------------------------------"
# 按正确的顺序卸载模块
MODULES=(nvidia_uvm nvidia_drm nvidia_modeset nvidia)
for module in "${MODULES[@]}"; do
    if lsmod | grep -q "^$module "; then
        echo "卸载模块: $module"
        rmmod $module 2>/dev/null || {
            echo "⚠️  警告: 无法卸载 $module，尝试强制卸载..."
            rmmod -f $module 2>/dev/null || echo "  跳过 $module"
        }
    else
        echo "模块 $module 未加载，跳过"
    fi
done
echo ""

echo "步骤 4: 重新加载 NVIDIA 内核模块"
echo "----------------------------------------"
# 按正确的顺序加载模块
MODULES_LOAD=(nvidia nvidia_modeset nvidia_drm nvidia_uvm)
for module in "${MODULES_LOAD[@]}"; do
    echo "加载模块: $module"
    modprobe $module || {
        echo "❌ 错误: 无法加载 $module"
        exit 1
    }
done
echo ""

echo "步骤 5: 验证修复结果"
echo "----------------------------------------"
sleep 2

# 检查 nvidia-smi
if nvidia-smi &>/dev/null; then
    echo "✓ nvidia-smi 正常工作"
    echo ""
    nvidia-smi
    echo ""
    echo "=== 修复成功! ==="
    echo ""
    echo "驱动版本信息:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    echo ""
    echo "您现在可以继续使用 GPU 进行训练。"
else
    echo "❌ 修复失败，nvidia-smi 仍然无法工作"
    echo "建议重启系统: sudo reboot"
    exit 1
fi



