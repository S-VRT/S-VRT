# Training Issues and Solutions

## Issue 1: torch.compile Incompatibility with VRT ❌

### Problem
When `COMPILE_MODEL: true`, training crashes with:
```
AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer

from user code:
   File "third_party/VRT/models/network_vrt.py", line 185, in drop_path
     if drop_prob == 0. or not training:
```

### Root Cause
`torch.compile` (torch.dynamo) has known issues with:
1. **Gradient checkpointing** (`checkpoint.checkpoint`) - VRT uses this extensively for memory efficiency
2. **Dynamic control flow** - The `drop_path` function has runtime-dependent branching
3. **Nested subgraphs** - VRT's architecture creates deeply nested computation graphs

### Solution
**Disable torch.compile** in the config:
```yaml
TRAIN:
  COMPILE_MODEL: false  # Disabled: incompatible with VRT's checkpoint.checkpoint
```

### Performance Impact
- torch.compile typically provides 10-30% speedup
- However, VRT already uses gradient checkpointing for memory efficiency
- The trade-off is acceptable given the stability gain

### Alternative Approaches (Not Recommended)
1. **Remove gradient checkpointing** - Would cause OOM on most GPUs
2. **Use `torch._dynamo.config.suppress_errors = True`** - Silently falls back to eager mode, defeating the purpose
3. **Rewrite VRT without checkpointing** - Major refactoring, not worth the effort

---

## Issue 2: RAM Cache Not Loading Data ⚠️

### Problem
When `USE_RAM_CACHE: true`, the cache shows:
```
[SpikeDeblurDataset] Loading 0 blur, 0 sharp, 0 voxel files...
[SpikeDeblurDataset] Cache preloaded: 0 items, 0.0GB / 30.0GB
```

### Root Cause
The `_preload_cache()` method is called in `__init__`, but `self._samples` might be empty if:
1. **Wrong split name** - e.g., using `train` when data is in `Train`
2. **Missing data directories** - `blur/`, `sharp/`, or `spike_vox/` not found
3. **Insufficient frames** - Sequences with fewer than `CLIP_LEN` frames are skipped

### Solution
Added debug logging and safety check:
```python
# Debug: print number of samples found
print(f"[SpikeDeblurDataset] Found {len(self._samples)} samples in split '{split}'")

# Preload data into cache if enabled
if use_ram_cache and len(self._samples) > 0:
    self._preload_cache()
elif use_ram_cache and len(self._samples) == 0:
    print(f"[SpikeDeblurDataset] Warning: RAM cache enabled but no samples found!")
```

### Debugging Steps
1. **Check split directory exists:**
   ```bash
   ls -la data/processed/gopro_spike_unified/train/
   ```

2. **Verify data structure:**
   ```
   train/
   ├── GOPR0372_07_00/
   │   ├── blur/
   │   │   ├── 00000000.png
   │   │   └── ...
   │   ├── sharp/
   │   │   ├── 00000000.png
   │   │   └── ...
   │   └── spike_vox/  (if USE_PRECOMPUTED_VOXELS: true)
   │       ├── 00000000.pt
   │       └── ...
   ```

3. **Check frame count:**
   ```bash
   # Should have at least CLIP_LEN frames (default: 5)
   ls data/processed/gopro_spike_unified/train/*/blur/ | head -1 | xargs ls | wc -l
   ```

---

## Issue 3: channels_last Memory Format Error ✅ (Fixed)

### Problem
```
RuntimeError: Tensor must have 4 dimensions, but got 5 dimensions
```

### Root Cause
`channels_last` memory format only works with 4D tensors (NCHW), but VRT uses 5D tensors (BTCHW) for video.

### Solution
Only apply `channels_last` to Conv2D layers:
```python
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        module = module.to(memory_format=torch.channels_last)
        channels_last_count += 1
```

---

## Issue 4: Hardcoded GPU Memory Estimation ✅ (Fixed)

### Problem
Memory estimation was hardcoded to 10GB, causing warnings even with small models.

### Solution
Dynamic calculation based on model size:
```python
model_params = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_gb = model_params / (1024**3)
optimizer_size_gb = model_size_gb * 2  # params + gradients
activation_size_gb = 10  # Rough estimate for activations
total_gpu_memory_gb = model_size_gb + optimizer_size_gb + activation_size_gb
```

---

## Current Configuration Status

### ✅ Working Optimizations
- [x] RAM Cache (when data is found)
- [x] Reduced workers (4 instead of 12)
- [x] channels_last for Conv2D only
- [x] Dynamic GPU memory estimation
- [x] Gradient accumulation (12 steps)
- [x] Mixed precision (AMP)

### ❌ Disabled Optimizations
- [ ] torch.compile (incompatible with VRT)

### 📊 Expected Performance
- **Data loading:** ~2-5s per batch (with RAM cache)
- **Forward pass:** ~1-2s per batch
- **Backward pass:** ~2-3s per batch
- **Total:** ~5-10s per batch (with grad accumulation)

---

## Recommendations

### For Future Work
1. **Consider alternative architectures** that are torch.compile-friendly
2. **Profile memory usage** to optimize cache size per worker
3. **Experiment with different worker counts** based on available RAM
4. **Use precomputed voxels** (`USE_PRECOMPUTED_VOXELS: true`) for faster loading

### For Debugging
1. Always check logs for:
   - Number of samples found
   - Cache statistics
   - GPU memory usage
2. Use `python scripts/verify_optimizations.py` before training
3. Monitor RAM usage: `watch -n 1 free -h`
4. Monitor GPU usage: `watch -n 1 nvidia-smi`

---

---

## Issue 5: NaN Values in TensorBoard Metrics 🔴

### Problem
TensorBoard显示的loss曲线在某个epoch后变成平线，鼠标悬停显示：
```
Name: .
Smoothed: NaN
Value: NaN
Step: 30
```

### Root Cause
`NaN`（Not a Number）代表浮点计算中出现了无效结果，常见原因：
1. **学习率过大** - 特别是在衰减前的早期阶段
2. **梯度爆炸** - 网络过深或特征值过大
3. **Loss函数数值不稳定** - 输入范围不当或计算过程溢出
4. **混合精度训练问题** - fp16下的数值下溢/溢出
5. **非法数学操作** - log(0)、sqrt(负数)、除以0等

### Symptoms
- Loss曲线突然变平
- TensorBoard显示NaN值
- 学习率调度器仍正常工作
- 训练吞吐量正常，说明不是崩溃

### Debugging Steps

#### 1️⃣ 快速检测NaN来源
在训练代码中添加检测：
```python
if torch.isnan(loss) or torch.isinf(loss):
    print(f"[Step {step}] NaN/Inf detected in loss!")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f" -> NaN/Inf in parameter: {name}")
    break
```

#### 2️⃣ 降低学习率
```yaml
TRAIN:
  OPTIM:
    LR: 0.0001  # 从 0.0002 降低到 0.0001
```

#### 3️⃣ 启用梯度裁剪
```yaml
TRAIN:
  MAX_GRAD_NORM: 1.0  # 裁剪梯度范数
```

#### 4️⃣ 检查Loss计算
```python
# 在loss计算前添加clamp
pred = torch.clamp(pred, 0, 1)
target = torch.clamp(target, 0, 1)
```

#### 5️⃣ 验证混合精度设置
如果使用AMP (`torch.cuda.amp.autocast()`):
```yaml
TRAIN:
  MIXED_PRECISION: true  # 确保GradScaler已启用
```

或暂时禁用AMP进行测试：
```yaml
TRAIN:
  MIXED_PRECISION: false  # 排查是否是AMP问题
```

### Solution Checklist
- [ ] 降低学习率 (例如从2e-4降到1e-4)
- [ ] 启用梯度裁剪 (`MAX_GRAD_NORM: 1.0`)
- [ ] 在loss计算前clamp输入范围
- [ ] 检查VGG Perceptual Loss的输入归一化
- [ ] 验证Charbonnier Loss的epsilon值 (默认1e-3)
- [ ] 如使用AMP，检查GradScaler配置

### Prevention
```yaml
# 推荐的稳定配置
TRAIN:
  OPTIM:
    LR: 0.0001  # 保守的学习率
  MAX_GRAD_NORM: 1.0  # 梯度裁剪
  MIXED_PRECISION: true  # 启用AMP但配合GradScaler
  
LOSS:
  CHARBONNIER:
    DELTA: 0.001  # 足够的epsilon值
  VGG_PERCEPTUAL:
    WEIGHT: 0.1  # 适中的权重，避免过大
```

---

## References
- [PyTorch torch.compile Limitations](https://pytorch.org/docs/stable/dynamo/index.html#limitations)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
- [Memory Formats](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
- [Numerical Stability in Deep Learning](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)


