# 模型计时调试指南

## 概述

本项目的模型各个模块已添加详细的计时调试信息，可以帮助你了解每个step中各个模块的执行时长。

## 添加的调试功能

### 1. 主模型 (VRTWithSpike)

位置: `src/models/integrate_vrt.py`

打印信息包括：
- 总体前向传播时间
- 三个主要步骤的耗时及占比：
  - Spike编码器
  - Spike时间自注意力
  - VRT处理与融合

### 2. Spike编码器 (SpikeEncoder3D)

位置: `src/models/spike_encoder3d.py`

打印信息包括：
- 输入维度转换时间
- 每个尺度的处理时间（投影、残差、下采样）
- 每个尺度的输出shape
- 总耗时

### 3. Spike时间自注意力 (SpikeTemporalSA)

位置: `src/models/spike_temporal_sa.py`

打印信息包括：
- 每个尺度的处理时间
- 每个尺度内部的详细计时：
  - 维度转换
  - Self-Attention（总时间和每块平均时间）
  - FFN前馈网络（总时间和每块平均时间）
  - 块数量统计
- 总耗时

### 4. 跨模态融合 (MultiScaleTemporalCrossAttnFuse)

位置: `src/models/fusion/cross_attn_temporal.py`

打印信息包括：
- 每个尺度的融合时间
- 每个融合块的详细计时：
  - 维度转换
  - LayerNorm
  - Cross-Attention（总时间和每块平均时间）
  - FFN前馈网络
  - 块数量统计
- 总耗时

### 5. VRT各阶段

位置: `src/models/integrate_vrt.py` (在monkey-patch的forward_features中)

打印信息包括：
- 编码阶段（Stage 1-4）每个stage的时间
- 每个stage后的融合时间
- 瓶颈层（Stage 5）时间
- 解码阶段（Stage 6-7）时间
- 重建层（Stage 8）时间
- LayerNorm时间

## 输出示例

```
================================================================================
开始前向传播 - VRTWithSpike
================================================================================

[主模型] 步骤1: Spike编码器
[主模型] 输入shape - RGB: torch.Size([1, 4, 3, 256, 256]), Spike: torch.Size([1, 4, 5, 256, 256])
  [SpikeEncoder3D] 输入维度转换: 0.15ms
  [SpikeEncoder3D] 尺度0 (输入投影+残差): 45.23ms, 输出shape: torch.Size([1, 96, 4, 256, 256])
  [SpikeEncoder3D] 尺度1 (下采样+残差): 38.67ms, 输出shape: torch.Size([1, 96, 4, 128, 128])
  [SpikeEncoder3D] 尺度2 (下采样+残差): 25.43ms, 输出shape: torch.Size([1, 96, 4, 64, 64])
  [SpikeEncoder3D] 尺度3 (下采样+残差): 18.92ms, 输出shape: torch.Size([1, 96, 4, 32, 32])
  [SpikeEncoder3D] 总耗时: 128.40ms

[主模型] Spike编码器总耗时: 128.40ms

[主模型] 步骤2: Spike时间自注意力
  [SpikeTemporalSA] 开始处理4个尺度的特征
  [SpikeTemporalSA] 尺度0: 输入shape=torch.Size([1, 96, 4, 256, 256])
    [SpikeTemporalSelfAttention] 维度转换: 0.12ms
    [SpikeTemporalSelfAttention] 处理了256个块
    [SpikeTemporalSelfAttention] Self-Attention总耗时: 156.78ms (平均0.61ms/块)
    [SpikeTemporalSelfAttention] FFN总耗时: 89.34ms (平均0.35ms/块)
    [SpikeTemporalSelfAttention] 块处理总耗时: 246.12ms
    [SpikeTemporalSelfAttention] 总耗时: 246.24ms
  [SpikeTemporalSA] 尺度0耗时: 246.45ms

  ...（其他尺度类似）

[主模型] Spike时间自注意力总耗时: 567.89ms

[主模型] 步骤3: VRT处理与融合
  [VRT] 开始编码阶段（4个Stage，每个Stage后融合）
  [VRT] Stage 1 (1x分辨率)耗时: 234.56ms
    [VRT融合] Stage 1 开始融合
    [TemporalCrossAttnFuseBlock] 维度转换: 0.23ms
    [TemporalCrossAttnFuseBlock] LayerNorm: 12.34ms
    [TemporalCrossAttnFuseBlock] Cross-Attention: 178.90ms
      [TemporalCrossAttention] 处理了256个块
      [TemporalCrossAttention] Cross-Attention总耗时: 176.54ms (平均0.69ms/块)
      [TemporalCrossAttention] 总耗时: 178.90ms
    [TemporalCrossAttnFuseBlock] FFN: 56.78ms
    [TemporalCrossAttnFuseBlock] 总耗时: 248.25ms
    [VRT融合] Stage 1 总耗时: 248.48ms (转换:0.23ms, 融合:248.25ms)

  ...（其他stages类似）

[主模型] VRT处理总耗时: 1234.56ms

================================================================================
前向传播总耗时: 1930.85ms
  - Spike编码器: 128.40ms (6.7%)
  - Spike时间自注意力: 567.89ms (29.4%)
  - VRT处理与融合: 1234.56ms (63.9%)
================================================================================
```

## 使用方法

### 1. 运行测试脚本

```bash
cd /home/mallm/henry/Deblur
python test_timing_debug.py
```

这将创建一个小型模型并运行前向传播，展示所有的计时信息。

### 2. 在训练/测试中查看

在正常的训练或测试过程中，这些调试信息会自动打印。如果输出太多，你可以：

**方法1: 重定向到文件**
```bash
python src/train.py > timing_debug.log 2>&1
```

**方法2: 使用grep过滤关键信息**
```bash
python src/train.py 2>&1 | grep "总耗时"
```

**方法3: 临时禁用调试信息**

如果你想临时禁用这些打印，可以在代码中添加一个全局开关：

```python
# 在模型文件开头添加
ENABLE_TIMING_DEBUG = False  # 设为False禁用调试输出

# 然后将所有 print() 改为:
if ENABLE_TIMING_DEBUG:
    print(...)
```

## 性能优化建议

根据输出的计时信息，你可以识别性能瓶颈：

1. **如果Spike编码器很慢**：考虑减少残差块数量或优化3D卷积
2. **如果时间自注意力很慢**：调整chunk_cfg参数增大块大小
3. **如果跨模态融合很慢**：可能需要优化attention head数量或chunk大小
4. **如果VRT某个stage特别慢**：可能需要检查该stage的配置

## 注意事项

1. **GPU同步**：当前的计时使用Python的`time.time()`，在GPU上运行时可能不够精确。如果需要更精确的GPU计时，可以使用`torch.cuda.synchronize()`和`torch.cuda.Event()`。

2. **第一次运行**：第一次前向传播可能会更慢（CUDA初始化、内核编译等），建议warm-up后再看计时。

3. **内存占用**：如果发现某个模块特别慢，也要检查是否是内存不足导致的。

4. **批量处理**：批大小会影响各个模块的相对耗时，建议在实际使用的批大小下测试。

## 进一步改进

如果需要更详细的profiling，可以考虑：

1. **使用PyTorch Profiler**：
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    model(rgb, spike)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. **使用TensorBoard**：可视化性能数据

3. **添加GPU内存使用统计**：在关键位置记录显存占用

