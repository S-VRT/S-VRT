# 项目工具脚本说明

本目录包含用于项目配置、分析和优化的实用工具。

## 🎯 核心工具

### 1. 硬件资源与训练速度分析 (推荐首次使用)

**文件**: `analyze_hardware_training.py`

**用途**: 在克隆项目后，首次运行训练前，分析您的硬件环境并生成优化建议。

**功能**:
- ✅ 自动检测GPU、CPU、内存等硬件资源
- ✅ 运行轻量级性能基准测试（约30秒）
- ✅ 分析当前配置的可行性
- ✅ 估算训练速度和显存占用
- ✅ 提供针对性的配置优化建议
- ✅ 自动生成优化后的配置文件

**使用方法**:
```bash
# 分析默认配置
python scripts/analyze_hardware_training.py

# 分析指定配置
python scripts/analyze_hardware_training.py configs/deblur/vrt_spike_baseline.yaml
```

**示例输出**:
```
硬件资源检测
================================================================================
✅ 检测到 3 个GPU:
  GPU 0: NVIDIA RTX A6000
    显存: 48.0 GB
    计算能力: 8.6
    ...

GPU性能基准测试
================================================================================
⏱️  正在运行轻量级基准测试（约30秒）...
[完成] FP16加速比: 2.3x

训练资源需求分析
================================================================================
📋 当前配置:
  Batch Size (per GPU): 1
  Gradient Accumulation: 6
  预计总需求 (FP32): ~18.5 GB
  预计总需求 (FP16): ~10.2 GB

配置优化建议
================================================================================
🟠 [HIGH] 启用混合精度训练 (AMP)
   原因: GPU的FP16性能是FP32的2.3倍
   建议: 在配置中添加: TRAIN.USE_AMP: true
   预期效果: 训练速度提升约30%, 显存减少45%
...

✅ 已生成优化配置: configs/deblur/vrt_spike_baseline_optimized.yaml
```

**适用场景**:
- 🆕 首次克隆项目
- 🔄 更换硬件环境
- ⚡ 训练速度优化
- 💾 显存不足问题排查
- ⚙️ 配置参数调优

---

## 📊 辅助分析工具

### 2. 训练性能分析

**文件**: `analyze_performance.py`

**用途**: 分析训练日志，找出模型各阶段的性能瓶颈。

**使用方法**:
```bash
# 分析最新的训练日志
python scripts/analyze_performance.py

# 分析指定日志文件
python scripts/analyze_performance.py outputs/logs/train_20241021_120000.log
```

**输出内容**:
- 各Stage耗时统计
- 性能瓶颈排名
- 优化建议

**适用场景**:
- 已经开始训练，需要分析性能瓶颈
- 优化模型架构
- 调试计时信息

---

### 3. GPU显存分析

**文件**: `analyze_gpu_memory.py`

**用途**: 详细分析GPU显存占用情况，帮助理解显存使用。

**使用方法**:
```bash
python scripts/analyze_gpu_memory.py
```

**输出内容**:
- GPU显存使用情况
- PyTorch内存追踪
- 显存占用分解
- 优化建议

**适用场景**:
- 遇到OOM（Out of Memory）错误
- 想了解显存分配细节
- 调试显存泄漏问题

---

## 🔄 使用流程建议

### 首次使用项目

```bash
# 1. 运行硬件分析（必须）
python scripts/analyze_hardware_training.py

# 2. 根据建议修改配置或使用生成的优化配置

# 3. 运行系统就绪测试
python tests/integration/training/test_system_readiness.py

# 4. 开始训练
python src/train.py --config configs/deblur/vrt_spike_baseline_optimized.yaml
```

### 训练过程中遇到问题

**场景1: 训练速度慢**
```bash
# 1. 查看训练日志中的性能分析
python scripts/analyze_performance.py

# 2. 根据瓶颈优化模型配置
```

**场景2: 显存不足 (OOM)**
```bash
# 1. 分析显存使用
python scripts/analyze_gpu_memory.py

# 2. 重新运行硬件分析，获取优化建议
python scripts/analyze_hardware_training.py

# 3. 应用建议（减小batch size、启用AMP等）
```

**场景3: 更换硬件环境**
```bash
# 重新运行硬件分析
python scripts/analyze_hardware_training.py configs/deblur/vrt_spike_baseline.yaml
```

---

## 🛠️ 其他工具

### 数据准备

**文件**: `prepare_gopro_spike_structure.py`

**用途**: 准备GoPro+Spike数据集

**使用方法**:
```bash
python scripts/prepare_gopro_spike_structure.py
```

### 训练监控

**文件**: `monitor_training.sh`

**用途**: 实时监控训练进程

**使用方法**:
```bash
bash scripts/monitor_training.sh
```

---

## 💡 常见问题

### Q1: 首次运行应该用哪个工具？

**A**: 使用 `analyze_hardware_training.py`。它会自动检测硬件并生成适合你环境的配置。

### Q2: 工具运行需要多长时间？

**A**: 
- `analyze_hardware_training.py`: 约1-2分钟（含基准测试）
- `analyze_performance.py`: 几秒钟
- `analyze_gpu_memory.py`: 几秒钟

### Q3: 基准测试会影响系统吗？

**A**: 基准测试是轻量级的，只运行30秒左右，不会对系统造成负面影响。建议在空闲时运行。

### Q4: 生成的优化配置安全吗？

**A**: 是的。工具只会应用"CRITICAL"和"HIGH"优先级的建议，这些都是保守的优化。你可以手动检查生成的配置文件。

### Q5: 我应该多久运行一次硬件分析？

**A**: 
- 首次克隆项目: 必须运行
- 更换硬件: 必须运行
- 训练配置变化: 建议运行
- 遇到性能问题: 建议运行
- 常规训练: 不需要

---

## 📖 更多信息

详细的配置说明和优化指南，请参考：
- [配置指南](../docs/CONFIG_GUIDE.md)
- [性能优化指南](../docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)
- [快速开始](../docs/QUICK_START.md)

---

**最后更新**: 2025-10-21

