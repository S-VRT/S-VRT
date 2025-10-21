# VRT-Spike 性能优化指南

> **基于实测数据的针对性优化方案**  
> 分析日期: 2025-10-17  
> 样本数: 111次前向传播  
> 平均耗时: 1789.72ms

---

## 📊 执行摘要

根据性能分析，当前模型存在**两个主要瓶颈**：

1. **🔴 Stage 8 (重建层)** - 占总耗时的 **32.8%** (586.90ms)
2. **🔴 Stage 2 (1/2x分辨率)** - 占总耗时的 **15.2%** (271.33ms)

**优化这两个阶段可获得最大性能提升（理论上可减少约50%的计算时间）**

---

## 🎯 优化优先级排序

### 第一优先级：Stage 8 重建层优化 (32.8% 耗时)

**问题诊断：**
- Stage 8 是最终的重建/上采样层，包含大量的卷积和反卷积操作
- 在全分辨率 (256×256) 上操作，计算量巨大
- 可能包含复杂的时空融合和重建模块

**优化方案：**

#### 1. 启用混合精度训练（立即实施）✅
```yaml
# 在 configs/deblur/vrt_spike_baseline.yaml 中添加：
TRAIN:
  AMP_ENABLED: true        # 自动混合精度
  AMP_OPT_LEVEL: "O1"      # O1: 保持大部分操作在FP16
```

**预期效果：** 减少 30-40% 的计算时间，降低显存使用

#### 2. 使用更轻量级的重建模块
```python
# 当前 VRT Stage 8 可能使用多个RSTB块
# 建议减少深度参数：

# 修改 src/train.py 中的 VRT 初始化：
vrt = VRT(
    upscale=1,
    in_chans=3,
    out_chans=3,
    img_size=img_size_cfg,
    window_size=window_size_cfg,
    embed_dims=embed_dims_cfg,
    depths=[8, 8, 8, 8, 4, 4, 4, 4],  # ← 减少 Stage 8 的深度
    #      ↑ Stage 1-7 保持不变   ↑ Stage 8 从8降到4
    use_checkpoint_attn=True,
    use_checkpoint_ffn=True,
)
```

**预期效果：** 减少 15-25% Stage 8 耗时

#### 3. 优化窗口大小
```yaml
# 在配置中调整窗口大小以平衡精度和性能：
MODEL:
  WINDOW_SIZE: 6  # 从 8 降到 6（减少约 44% 的注意力计算量）
```

**预期效果：** 减少 20-30% Stage 8 耗时，精度损失通常<0.5dB

---

### 第二优先级：Stage 2 优化 (15.2% 耗时)

**问题诊断：**
- Stage 2 在 1/2x 分辨率 (128×128) 上操作
- 通常包含较多的 Transformer 块（depth 参数较大）
- 可能存在冗余的计算

**优化方案：**

#### 1. 减少 Stage 2 的 Transformer 块数量
```python
# 修改 src/train.py：
vrt = VRT(
    depths=[8, 6, 8, 8, 4, 4, 4, 4],  # Stage 2 从8降到6
    #         ↑ 减少2个块
    # ...
)
```

**预期效果：** 减少 25% Stage 2 耗时

#### 2. 启用 Flash Attention（如果可用）
```python
# 在 VRT 初始化时添加：
vrt = VRT(
    # ...
    attn_type="flash",  # 如果 VRT 支持
    # ...
)
```

**预期效果：** 减少 30-50% 注意力计算时间

---

### 第三优先级：Stage 1 融合优化 (8.6% 耗时)

**问题诊断：**
- Stage 1 的 Cross-Attention 融合耗时 154.02ms
- 在全分辨率上进行融合，计算量大

**优化方案：**

#### 1. 增大融合模块的 chunk_size
```yaml
# 在 configs/deblur/vrt_spike_baseline.yaml 中：
MODEL:
  FUSE:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536  # 从 49152 增加到 65536
    CHUNK_SIZE: 96           # 从 64 增加到 96
    CHUNK_SHAPE: "square"
```

**预期效果：** 减少 20-30% 融合耗时

#### 2. 考虑在低分辨率进行融合
```python
# 修改融合策略：仅在 Stage 2-4 进行融合，Stage 1 直接跳过
# 在 src/models/integrate_vrt.py 中修改 _fuse_after_stage 逻辑
```

**预期效果：** 完全消除 Stage 1 的融合开销（8.6%）

---

## 🚀 快速优化方案（推荐首先尝试）

### 方案 A：保守优化（精度优先）

**配置更改：**
```yaml
# configs/deblur/vrt_spike_baseline_opt.yaml
MODEL:
  WINDOW_SIZE: 7  # 从 8 降到 7
  FUSE:
    CHUNK_SIZE: 80  # 从 64 增加到 80
  
TRAIN:
  AMP_ENABLED: true
  AMP_OPT_LEVEL: "O1"
```

**VRT 深度调整：**
```python
depths=[8, 7, 8, 8, 4, 4, 4, 6]  # Stage 2 从8→7, Stage 8 从8→6
```

**预期提升：** 25-35% 性能提升，<0.3dB 精度损失

---

### 方案 B：激进优化（性能优先）

**配置更改：**
```yaml
# configs/deblur/vrt_spike_fast.yaml
MODEL:
  WINDOW_SIZE: 6  # 从 8 降到 6
  FUSE:
    CHUNK_SIZE: 96
    MAX_BATCH_TOKENS: 65536
  SPIKE_TSA:
    CHUNK_SIZE: 96
    MAX_BATCH_TOKENS: 65536

TRAIN:
  AMP_ENABLED: true
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 2  # 混合精度下可增大 batch size
```

**VRT 深度调整：**
```python
depths=[8, 6, 6, 8, 4, 4, 4, 4]  # Stage 2 从8→6, Stage 8 从8→4
```

**预期提升：** 45-60% 性能提升，0.5-1.0dB 精度损失

---

## 📈 实施步骤

### Step 1: 创建优化配置文件

```bash
# 创建保守优化版本
cp configs/deblur/vrt_spike_baseline.yaml \
   configs/deblur/vrt_spike_opt.yaml

# 创建激进优化版本
cp configs/deblur/vrt_spike_baseline.yaml \
   configs/deblur/vrt_spike_fast.yaml
```

### Step 2: 修改 `src/train.py` 的 VRT 初始化

在 `create_model()` 函数中添加深度参数配置：

```python
def create_model(cfg: dict, device: torch.device) -> nn.Module:
    # ... 现有代码 ...
    
    # 从配置读取 VRT depths（如果未指定则使用默认值）
    vrt_depths = cfg.get("MODEL", {}).get("VRT_DEPTHS", [8, 8, 8, 8, 4, 4, 4, 8])
    
    vrt = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=img_size_cfg,
        window_size=window_size_cfg,
        embed_dims=embed_dims_cfg,
        depths=vrt_depths,  # ← 添加这一行
        use_checkpoint_attn=True,
        use_checkpoint_ffn=True,
    )
    # ... 其余代码 ...
```

### Step 3: 更新配置文件

**编辑 `configs/deblur/vrt_spike_opt.yaml`：**
```yaml
MODEL:
  VRT_DEPTHS: [8, 7, 8, 8, 4, 4, 4, 6]  # 保守优化
  WINDOW_SIZE: 7
  FUSE:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536
    CHUNK_SIZE: 80
    CHUNK_SHAPE: "square"

TRAIN:
  AMP_ENABLED: true  # 启用混合精度
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 1  # 保持不变
```

**编辑 `configs/deblur/vrt_spike_fast.yaml`：**
```yaml
MODEL:
  VRT_DEPTHS: [8, 6, 6, 8, 4, 4, 4, 4]  # 激进优化
  WINDOW_SIZE: 6
  FUSE:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536
    CHUNK_SIZE: 96
    CHUNK_SHAPE: "square"
  SPIKE_TSA:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536
    CHUNK_SIZE: 96
    CHUNK_SHAPE: "square"

TRAIN:
  AMP_ENABLED: true
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 2  # 混合精度下可增大
```

### Step 4: 启用混合精度训练

在 `src/train.py` 中添加 AMP 支持：

```python
# 在 train() 函数开始处添加：
from torch.cuda.amp import autocast, GradScaler

def train(...):
    # ... 现有代码 ...
    
    # 初始化 GradScaler（如果启用 AMP）
    amp_enabled = cfg.get("TRAIN", {}).get("AMP_ENABLED", False)
    scaler = GradScaler() if amp_enabled else None
    
    # 在训练循环中修改：
    for step, batch in enumerate(train_loader):
        # ... 数据加载 ...
        
        optimizer.zero_grad()
        
        # 使用自动混合精度
        with autocast(enabled=amp_enabled):
            out = model(rgb, spike)
            loss_dict = criterion(out, y_gt)
            loss = sum(loss_dict.values())
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # ... 其余代码 ...
```

### Step 5: 运行基准测试

```bash
# 测试保守优化版本
python src/train.py --config configs/deblur/vrt_spike_opt.yaml

# 测试激进优化版本
python src/train.py --config configs/deblur/vrt_spike_fast.yaml

# 运行性能分析
python analyze_performance.py outputs/logs/train_<timestamp>.log
```

---

## 🔬 高级优化技术

### 1. 使用 torch.compile() (PyTorch 2.0+)

```python
# 在模型创建后添加：
if torch.__version__ >= "2.0.0" and cfg.get("TRAIN", {}).get("COMPILE_MODEL", False):
    model = torch.compile(model, mode="max-autotune")
```

**注意：** 需要确保 VRT 的 `checkpoint.checkpoint` 兼容性

### 2. 渐进式训练策略

```yaml
# 先用小分辨率训练，再微调大分辨率
DATA:
  CROP_SIZE: 128  # 前40 epochs
  # CROP_SIZE: 256  # 后40 epochs
```

### 3. 知识蒸馏

训练一个更小的学生模型，使用当前模型作为教师：

```python
# 创建轻量级模型
vrt_student = VRT(
    depths=[4, 4, 4, 4, 2, 2, 2, 2],  # 深度减半
    # ...
)
```

---

## 📊 预期性能对比

| 配置 | 预期耗时 | 相对提升 | 精度影响 |
|------|---------|---------|---------|
| 当前基线 | 1789ms | - | - |
| 保守优化 | 1200-1300ms | +30-35% | <0.3dB |
| 激进优化 | 800-1000ms | +45-60% | 0.5-1.0dB |
| 混合精度 | 1100-1200ms | +35-40% | <0.1dB |
| 综合优化 | 700-900ms | +50-70% | 0.5-1.0dB |

---

## ✅ 验证清单

训练完成后，请验证：

- [ ] 训练损失曲线是否稳定收敛
- [ ] 验证集 PSNR/SSIM 是否在可接受范围内
- [ ] GPU 显存使用是否降低
- [ ] 训练速度（samples/sec）是否提升
- [ ] 使用 `analyze_performance.py` 确认瓶颈是否得到缓解

---

## 🐛 故障排除

### 问题 1: OOM (内存不足)

**解决方案：**
```yaml
TRAIN:
  BATCH_SIZE: 1  # 降低 batch size
  GRADIENT_ACCUMULATION_STEPS: 12  # 增加梯度累积
```

### 问题 2: 混合精度训练不稳定

**解决方案：**
```python
# 使用更保守的 O1 级别，或添加损失缩放：
scaler = GradScaler(init_scale=2.**10)  # 降低初始缩放
```

### 问题 3: 精度显著下降

**解决方案：**
- 回退到保守优化方案
- 增加训练 epochs 以补偿容量损失
- 使用知识蒸馏保持精度

---

## 📚 参考资源

- [VRT 官方仓库](https://github.com/JingyunLiang/VRT)
- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [性能分析工具使用](./analyze_performance.py)

---

## 📝 下一步行动

1. ✅ **立即实施：** 启用混合精度训练（最简单，收益最大）
2. ⏭️ **短期目标：** 创建并测试保守优化配置
3. 🎯 **中期目标：** 测试激进优化配置，找到精度-性能最佳平衡点
4. 🚀 **长期目标：** 探索模型架构改进和知识蒸馏

---

**祝优化顺利！如有问题，请查看 `analyze_performance.py` 生成的详细报告。** 🎉

