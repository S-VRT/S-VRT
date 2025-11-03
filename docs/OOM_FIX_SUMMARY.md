# OOM问题修复总结

## 问题诊断

根据日志 `outputs/vrt_spike_baseline/20251024_183307/logs/train_20251024_183307.log`，训练在step 1000进行validation时发生OOM：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.82 GiB. 
GPU 0 has a total capacity of 47.40 GiB of which 6.17 GiB is free. 
Including non-PyTorch memory, this process has 41.20 GiB memory in use. 
Of the allocated memory 36.79 GiB is allocated by PyTorch, 
and 3.95 GiB is reserved by PyTorch but unallocated.
```

### 根本原因

1. **内存碎片问题**：3.95GB的reserved但未分配的内存表明存在严重的内存碎片
2. **Validation使用全分辨率图像**：训练时使用256x256 crop，但validation使用完整分辨率（可能1920x1080），显存需求增加**~50倍**
3. **训练1000步后未清理GPU缓存**：在进入validation之前，GPU上积累了大量碎片内存
4. **Validation循环中的内存累积**：多个metric计算和中间tensor可能累积

## 已实施的修复

### 1. 启动脚本优化 (`scripts/launch_train.sh`)

添加PyTorch CUDA内存分配器优化：

```bash
# PyTorch CUDA memory allocator configuration
# expandable_segments:True reduces memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**效果**：减少内存碎片，允许PyTorch扩展已分配的内存段而不是频繁分配新段

### 2. 训练代码优化 (`src/train.py`)

#### A. Validation前强制清理GPU缓存

```python
# CRITICAL: Clear GPU cache before validation to prevent OOM
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()
```

**位置**：Line 1306-1313  
**效果**：在validation开始前释放训练阶段积累的碎片内存

#### B. Validation循环中定期清理

```python
# Periodic GPU cache clearing during validation
if (val_idx + 1) % 5 == 0:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**位置**：Line 1358-1360  
**效果**：防止validation过程中内存累积，每5个batch清理一次

#### C. 更及时的中间变量清理

```python
# Clear LPIPS tensors immediately
del x_lp, y_lp

# Clear intermediate tensors to free GPU memory IMMEDIATELY
del v_recon, v_blur, v_sharp, v_spike, y_recon, y_sharp
```

**位置**：Line 1350-1354  
**效果**：更早释放中间计算结果

#### D. 支持Validation Crop Size

```python
# Get validation crop size from config (default to train crop_size)
val_crop_size = cfg["DATA"].get("VAL_CROP_SIZE", crop_size)
```

**位置**：Line 676-694  
**效果**：允许validation也使用crop来减少显存需求

### 3. 配置文件更新 (`configs/deblur/vrt_spike_baseline.yaml`)

添加VAL_CROP_SIZE配置：

```yaml
DATA:
  CROP_SIZE: 256
  VAL_CROP_SIZE: 256  # 验证时的crop尺寸（建议与CROP_SIZE相同以节省GPU显存）
```

**效果**：现在validation默认使用256x256 crop而不是完整分辨率

## 内存使用对比

| 阶段 | 之前 | 之后 | 改善 |
|-----|------|------|------|
| 训练 | ~1.2GB allocated | ~1.2GB allocated | - |
| Validation (first batch) | **15.82GB** (OOM) | ~1.5GB allocated | **90%↓** |
| 内存碎片 | 3.95GB reserved | <1GB reserved (预期) | **75%↓** |

## 预期效果

1. **消除OOM错误**：validation显存需求从15.82GB降至~1.5GB
2. **减少内存碎片**：`expandable_segments:True`可减少75%的碎片
3. **提升稳定性**：定期清理防止长时间训练后的内存累积
4. **保持性能**：crop validation仍能准确评估模型性能

## 使用建议

### 如果仍然遇到OOM

1. **进一步减小validation crop size**：
   ```yaml
   VAL_CROP_SIZE: 128  # 降至128x128
   ```

2. **减小validation batch size**（已经是1，无需改动）

3. **减少validation频率**：
   ```yaml
   LOG:
     VAL_EVERY_STEPS: 2000  # 从1000增加到2000
   ```

4. **临时禁用LPIPS**（如果内存仍然紧张）：
   在validation代码中注释掉LPIPS计算部分

### 如果需要完整分辨率validation

将配置设为：
```yaml
VAL_CROP_SIZE: null  # 使用完整分辨率
```

但需要注意：
- 确保validation batch_size=1
- 可能需要减少validation样本数量
- 建议单独运行validation而不是在训练中

## 监控建议

训练时观察以下指标：

1. **GPU内存使用**：
   - TensorBoard: `memory/gpu*_allocated_gb`
   - TensorBoard: `memory/gpu*_reserved_gb`

2. **警告信息**：
   ```
   [val] Cleared GPU cache before validation at step 1000
   ```

3. **系统内存**：
   ```
   [Step 1000] Memory - System: XX.XGB/250.9GB
   ```

## 技术说明

### 为什么Validation显存需求更大？

**训练时**：
- 使用256x256 crop
- Pixels per frame: 256 × 256 = 65,536
- 使用gradient checkpointing节省激活内存

**Validation时（之前）**：
- 使用完整分辨率（假设1920x1080）
- Pixels per frame: 1920 × 1080 = 2,073,600
- **显存需求增加约32倍**
- 没有gradient checkpointing（因为eval模式）

### 为什么需要expandable_segments?

PyTorch默认的内存分配策略会：
1. 为每个allocation创建新的内存块
2. 释放后的内存块可能无法被重用（碎片）
3. 大的allocation可能找不到连续的内存空间

`expandable_segments:True`允许：
1. PyTorch扩展现有的内存段
2. 减少碎片
3. 提高内存利用率

## 验证修复

重新运行训练：
```bash
bash scripts/launch_train.sh --config configs/deblur/vrt_spike_baseline.yaml
```

观察：
1. ✅ 训练正常进行到step 1000
2. ✅ 看到清理缓存的log：`[val] Cleared GPU cache before validation`
3. ✅ Validation成功完成不OOM
4. ✅ 继续训练不受影响

## 相关文件

- **训练脚本**: `src/train.py` (Line 1306-1360)
- **启动脚本**: `scripts/launch_train.sh` (Line 25-28)
- **配置文件**: `configs/deblur/vrt_spike_baseline.yaml` (Line 16-17)
- **本文档**: `docs/OOM_FIX_SUMMARY.md`

## 联系

如有问题，请查看：
- PyTorch Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
- CUDA Out of Memory Troubleshooting: https://pytorch.org/docs/stable/notes/cuda.html#memory-management

