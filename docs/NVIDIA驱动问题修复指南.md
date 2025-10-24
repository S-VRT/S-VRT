# NVIDIA 驱动问题修复指南

## 快速诊断

### 常见问题

**问题1：驱动/库版本不匹配**
```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.95
```

**问题2：NCCL 训练错误**
```
NCCL WARN nvmlInit_v2() failed: Driver/library version mismatch
torch.distributed.DistBackendError: NCCL error
```

**问题3：nvidia-smi 无法工作**
```
nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
```

### 根本原因

所有这些问题通常源于同一个根本原因：

```
内核模块版本: 580.82.09 (旧版本，在内存中运行)
NVML 库版本:  580.95     (新版本，已安装)
结果:         版本不匹配 → 无法通信
```

**发生场景**：
1. 通过 `apt upgrade` 更新了 NVIDIA 驱动
2. 更新后没有重启系统
3. 旧的内核模块仍在内存中运行
4. 新的用户空间库已安装
5. 两者版本不匹配导致通信失败

---

## 解决方案（按推荐顺序）

### 方案 1：重启系统（最简单、最可靠）✅

```bash
sudo reboot
```

**优点**：
- ✓ 100% 成功率
- ✓ 最简单安全
- ✓ 自动加载正确的驱动版本
- ✓ 无需手动操作

**缺点**：
- ✗ 需要停机时间
- ✗ 中断所有运行中的任务

**适用场景**：
- 服务器可以重启
- 没有紧急运行的任务
- 首次遇到此问题

---

### 方案 2：自动化脚本修复（无需重启）

如果您的系统已有修复脚本：

```bash
# 1. 停止所有使用 GPU 的进程
# 检查是否有进程在使用 GPU
lsof /dev/nvidia* 2>/dev/null

# 2. 运行修复脚本
sudo bash /home/mallm/henry/Deblur/scripts/fix_nvidia_driver.sh
```

**脚本功能**：
- ✓ 自动检查 GPU 使用进程
- ✓ 按正确顺序卸载旧模块
- ✓ 加载新版本模块
- ✓ 验证修复结果

---

### 方案 3：手动重新加载内核模块

**步骤 1：检查 GPU 使用情况**

```bash
# 查看哪些进程在使用 GPU
sudo lsof /dev/nvidia* 2>/dev/null

# 或者查看模块依赖
lsmod | grep nvidia
```

**步骤 2：停止 GPU 进程**

```bash
# 停止 Python 训练程序
pkill -f python

# 停止 Jupyter
pkill -f jupyter

# 如果使用了 nvidia-persistenced
sudo systemctl stop nvidia-persistenced
```

**步骤 3：卸载 NVIDIA 模块**

⚠️ **顺序很重要！必须按以下顺序卸载：**

```bash
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
```

**步骤 4：重新加载模块**

```bash
sudo modprobe nvidia
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm
```

**步骤 5：验证**

```bash
nvidia-smi
```

预期输出应显示正确的驱动版本（如 580.95）。

---

### 常见错误与解决

#### 错误 1：模块正在使用中

```bash
rmmod: ERROR: Module nvidia is in use
```

**解决方法**：

```bash
# 查看哪个模块依赖 nvidia
lsmod | grep nvidia

# 查看哪些进程在使用
sudo fuser -v /dev/nvidia*

# 必须先停止所有 GPU 进程：
# - Python 训练脚本
# - Jupyter notebook
# - CUDA 应用程序
# - X11 服务器（如果使用桌面环境）
```

#### 错误 2：无法卸载（桌面环境占用）

如果使用图形界面，X11 服务器可能占用 GPU：

```bash
# 临时切换到文本模式
sudo systemctl isolate multi-user.target

# 执行模块重载
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# 恢复图形界面
sudo systemctl isolate graphical.target
```

#### 错误 3：nvidia-persistenced 阻止

```bash
# 停止持久化守护进程
sudo systemctl stop nvidia-persistenced

# 重新加载模块
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# 重新启动守护进程
sudo systemctl start nvidia-persistenced
```

---

## NCCL 训练相关问题

### 环境变量临时解决方案

如果无法立即修复驱动问题，可以通过环境变量临时绕过：

在 `scripts/launch_train.sh` 中添加：

```bash
# 禁用 NVML，避免驱动/库版本不匹配
export NCCL_NVML_DISABLE=1

# 更新过时的环境变量
# 旧版: export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

**优点**：
- 无需重启系统
- 无需更改驱动或 CUDA 版本
- 仍可正常使用多 GPU 分布式训练

**缺点**：
- NCCL 无法通过 NVML 获取某些 GPU 监控信息（不影响训练功能）
- 治标不治本，建议仍然修复驱动问题

---

## 验证修复结果

### 1. 检查 nvidia-smi

```bash
nvidia-smi
```

**预期输出**：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 580.95       Driver Version: 580.95       CUDA Version: 13.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|...
```

### 2. 运行 CUDA 环境检查脚本

```bash
cd /home/mallm/henry/Deblur
bash scripts/check_cuda_env.sh
```

**预期输出**：
```
1. NVIDIA Driver Version:
580.95

2. CUDA Driver Version (from nvidia-smi):
CUDA Version: 12.8

4. PyTorch CUDA Information:
  PyTorch version: 2.8.0+cu128
  CUDA available: True
  CUDA version (PyTorch): 12.8
  Number of GPUs: 3
    GPU 0: NVIDIA RTX A6000
    GPU 1: NVIDIA RTX A6000
    GPU 2: NVIDIA RTX A6000

5. NCCL Test:
  ✓ NCCL is available
```

### 3. 测试分布式训练初始化

```bash
python -c "import torch; torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0); print('✓ NCCL 初始化成功')"
```

### 4. 测试训练

```bash
# 测试预处理模块
python scripts/test_preprocessing.py

# 启动训练
bash scripts/launch_train.sh
```

---

## 版本兼容性说明

### CUDA 版本关系

您的系统可能有多个 CUDA 版本：

```
NVIDIA 驱动:    580.95.05  (支持 CUDA 13.0)
系统 nvcc:      CUDA 12.2
PyTorch CUDA:   CUDA 12.8
```

这是**正常的**！因为：

1. **驱动支持的 CUDA 版本**：向后兼容，只要驱动版本足够新即可
2. **系统 nvcc**：用于编译 CUDA 程序
3. **PyTorch CUDA**：PyTorch 编译时使用的 CUDA 版本

只要：`驱动支持的 CUDA 版本 ≥ PyTorch CUDA 版本`，就可以正常工作。

### 不需要重新安装 PyTorch

修复驱动后，**无需重新安装 PyTorch**。CUDA 运行时库向后兼容。

---

## 推荐执行流程

### 情况 1：服务器可以重启

```bash
# 最简单的方案
sudo reboot
```

重启后验证：
```bash
nvidia-smi
bash scripts/check_cuda_env.sh
bash scripts/launch_train.sh
```

### 情况 2：服务器不能重启

**第一步：快速尝试**（1分钟）
```bash
# 检查是否有 GPU 进程
sudo lsof /dev/nvidia* 2>/dev/null

# 如果没有，直接重新加载模块
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
nvidia-smi
```

**第二步：如果失败，清理 GPU 进程**
```bash
# 停止所有 Python 进程
pkill -f python

# 停止 Jupyter
pkill -f jupyter

# 停止 nvidia-persistenced
sudo systemctl stop nvidia-persistenced

# 重试模块重新加载
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm
```

**第三步：如果还是失败，使用环境变量临时方案**
```bash
# 编辑 scripts/launch_train.sh
# 添加: export NCCL_NVML_DISABLE=1

# 然后继续训练
bash scripts/launch_train.sh
```

---

## 常见问题 FAQ

### Q1：为什么会出现这个问题？

**A**：系统通过包管理器（`apt`）更新了 NVIDIA 驱动，但内核模块仍然是旧版本在内存中运行。新的用户空间库（NVML）与旧的内核模块版本不匹配，无法通信。

### Q2：这个问题影响训练吗？

**A**：是的。虽然 PyTorch 可以检测到 GPU，但分布式训练初始化时（`dist.barrier()`）会失败，NVML 警告也会影响性能监控。

### Q3：重新加载模块安全吗？

**A**：非常安全，前提是没有程序正在使用 GPU。这就像"热插拔"驱动，不会损坏硬件或数据。

### Q4：如果我正在运行训练怎么办？

**A**：必须先停止训练。保存检查点，停止进程，修复驱动，然后从检查点恢复训练。

### Q5：修复后需要重新安装 PyTorch 吗？

**A**：不需要。PyTorch 使用的是 CUDA 运行时库，与驱动版本向后兼容。只要驱动支持的 CUDA 版本 ≥ PyTorch 需要的版本即可。

### Q6：训练报错是预处理代码的问题吗？

**A**：**不是**。错误发生在 `dist.barrier()`（分布式训练同步点），这在任何预处理代码执行**之前**。预处理实现是正确的。

---

## 系统状态摘要

**典型配置**：
- 硬件：3 × NVIDIA RTX A6000
- PyTorch：2.8.0+cu128
- CUDA (PyTorch)：12.8
- NCCL：可用
- **驱动问题**：内核模块（旧版） ≠ NVML 库（新版）

**修复目标**：
- 内核模块版本 = NVML 库版本
- `nvidia-smi` 正常工作
- 分布式训练正常初始化

---

## 快速参考卡片

### 一键修复（推荐）

```bash
# 方法 1：重启（最可靠）
sudo reboot

# 方法 2：重新加载模块（无需重启）
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia nvidia_modeset nvidia_drm nvidia_uvm

# 验证
nvidia-smi
bash scripts/check_cuda_env.sh
bash scripts/launch_train.sh
```

### 紧急临时方案

```bash
# 在 scripts/launch_train.sh 中添加
export NCCL_NVML_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
```

---

## 相关资源

- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [PyTorch Distributed Troubleshooting](https://pytorch.org/docs/stable/distributed.html)
- [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)

---

**需要帮助？** 如果遇到任何问题，请提供：
1. `nvidia-smi` 的完整输出
2. `lsmod | grep nvidia` 的输出
3. 具体的错误信息

