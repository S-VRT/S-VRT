# 配置文件说明

## 完整的 YAML 配置结构

本文档说明训练配置文件的各个部分。

---

## 0. 配置文件快速选择指南

项目提供三个预配置的YAML文件，适用于不同的硬件环境：

### 🎯 三种配置对比

| 配置文件 | 适用场景 | GPU内存需求 | 训练速度 | 关键特性 |
|---------|---------|------------|---------|---------|
| **vrt_spike_baseline.yaml** | 标准训练（推荐） | 30-35GB/GPU | 基准速度 | 梯度检查点 + 标准优化器 |
| **vrt_spike_fast.yaml** | 高性能训练 | 40GB+/GPU | 快约2-3倍 | 无检查点 + 标准优化器 |
| **vrt_spike_low_memory.yaml** | 低显存环境 | 24GB+/GPU | 慢约50-60% | 梯度检查点 + 8-bit优化器 |

### 🚀 快速使用

```bash
# 标准训练（推荐配置，平衡性能和内存）
python src/train.py --config configs/deblur/vrt_spike_baseline.yaml

# 高性能训练（需要大显存，如3×A6000 48GB）
python src/train.py --config configs/deblur/vrt_spike_fast.yaml

# 低显存训练（24GB GPU也可训练）
pip install bitsandbytes  # 首次需要安装
python src/train.py --config configs/deblur/vrt_spike_low_memory.yaml
```

### 📊 内存和速度权衡

**内存分配示例（单GPU）：**
- 模型参数：~5-8GB
- 优化器状态（标准AdamW）：~10-16GB (模型参数×2)
- 优化器状态（8-bit AdamW）：~2-4GB (节省75%)
- 激活值（有检查点）：~5-10GB
- 激活值（无检查点）：~20-30GB
- VGG/LPIPS损失模型：~2-3GB

**配置选择建议：**
- **48GB GPU** → 选择 `fast` 配置（速度优先）
- **32-40GB GPU** → 选择 `baseline` 配置（平衡）
- **24GB GPU** → 选择 `low_memory` 配置（节省内存）

---

## 1. DATALOADER 配置（数据加载性能优化）

```yaml
DATALOADER:
  # CPU Worker配置
  TOTAL_WORKERS: "auto"          # 总worker数，推荐使用 "auto"
  
  # 可选的细粒度配置
  TRAIN_WORKERS: "auto"          # 训练worker数（可选）
  VAL_WORKERS: "auto"            # 验证worker数（可选）
  
  # Prefetch配置
  TRAIN_PREFETCH_FACTOR: 4       # 训练预取批次数
  VAL_PREFETCH_FACTOR: 2         # 验证预取批次数
  
  # 内存配置
  PIN_MEMORY: true               # 启用pin memory
  PERSISTENT_WORKERS: true       # 保持workers活跃
```

### TOTAL_WORKERS 选项说明

1. **`"auto"`（推荐）**
   - 自动使用 80% 的 CPU 核心
   - 例如：40核系统 → 32 workers
   - 保留 20% CPU 给系统和训练进程

2. **`"cpu*0.9"`（自定义百分比）**
   - 使用指定百分比的 CPU 核心
   - 例如：`"cpu*0.9"` 在 40核系统 → 36 workers

3. **整数（直接指定）**
   - 例如：`TOTAL_WORKERS: 24`
   - 会在所有 GPU 间平均分配

### Worker 分配示例

**3 GPU 系统，40 CPU 核心：**

```yaml
TOTAL_WORKERS: "auto"  # 40 * 0.8 = 32 workers
# 结果：
# - 每个 GPU: 32/3 ≈ 10 个训练workers
# - 每个 GPU: 10/2 = 5 个验证workers
# - 总计: 3×10 = 30 个workers并行加载
```

---

## 2. LOG 配置（日志和输出）

```yaml
LOG:
  # 基础配置
  TENSORBOARD: true              # 启用TensorBoard
  SAVE_DIR: outputs              # 输出目录
  VAL_EVERY_STEPS: 1000          # 验证间隔
  
  # Timing Logger 配置（新增）
  ENABLE_TIMING_LOG: true        # 是否启用耗时日志
  TIMING_CONSOLE: true           # 终端显示
  TIMING_FILE: true              # 文件记录
  TIMING_CONSOLE_INTERVAL: 10    # 终端更新间隔（每N个step）
  TIMING_FILE_INTERVAL: 50       # 文件刷新间隔（每N个step）
```

### Timing Logger 说明

- **ENABLE_TIMING_LOG**: 主开关，设为 `false` 可完全禁用
- **TIMING_CONSOLE**: 终端原地更新显示（彩色进度条）
- **TIMING_FILE**: 详细日志文件（保存到 `{SAVE_DIR}/logs/timing_*.log`）
- **TIMING_CONSOLE_INTERVAL**: 更新频率，越大越省资源
- **TIMING_FILE_INTERVAL**: 文件刷新频率

---

## 2.5. MODEL 配置（模型和内存优化）

```yaml
MODEL:
  USE_SPIKE: true
  VRT_CFG: third_party/VRT/options/deblur/vrt_base.yaml
  CHANNELS_PER_SCALE: [96, 96, 96, 96]
  LAYERS: 4
  
  # VRT Gradient Checkpointing Settings（梯度检查点设置）
  VRT:
    USE_CHECKPOINT_ATTN: true   # 检查点注意力层（推荐: true）
    USE_CHECKPOINT_FFN: true    # 检查点前馈层（推荐: true）
  
  SPIKE_TSA:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 49152
  
  FUSE:
    TYPE: TemporalCrossAttn
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 49152
```

### VRT Gradient Checkpointing 说明

**什么是梯度检查点？**
- 在前向传播时不保存中间激活值，在反向传播时重新计算
- 用计算时间换取内存空间（典型trade-off: 2-3倍计算时间，节省90%激活值内存）

**两个独立开关：**
1. **USE_CHECKPOINT_ATTN**: 控制Self-Attention层的检查点
   - `true`: 节省大量内存（注意力计算的激活值占比最大）
   - `false`: 更快的反向传播，但内存占用高

2. **USE_CHECKPOINT_FFN**: 控制Feed-Forward Network层的检查点
   - `true`: 进一步节省内存（FFN激活值也较大）
   - `false`: 更快的反向传播

**推荐配置：**
- **低显存环境（24-32GB）**: 都设为 `true`（默认）
- **充足显存环境（40GB+）**: 都设为 `false`（见 `vrt_spike_fast.yaml`）
- **中等显存环境（32-40GB）**: 可尝试只checkpoint注意力层

**内存节省估算：**
- 仅checkpoint注意力：节省约60-70%激活值内存
- 同时checkpoint注意力和FFN：节省约85-90%激活值内存

---

## 3. TRAIN 配置（训练参数）

```yaml
TRAIN:
  EPOCHS: 40
  BATCH_SIZE: 1                  # 每GPU的batch size
  VAL_BATCH_SIZE: 1              # 验证batch size
  
  # 梯度累积和优化
  GRADIENT_ACCUMULATION_STEPS: 6 # 梯度累积步数
  MAX_GRAD_NORM: 1.0             # 梯度裁剪
  COMPILE_MODEL: false           # 是否编译模型（需要PyTorch 2.0+）
  
  # 优化器配置
  OPTIM:
    TYPE: adamw                  # 优化器类型: "adamw" (默认) 或 "adamw8bit" (节省约75%优化器内存)
    LR: 0.0002
    BETAS: [0.9, 0.99]
    WEIGHT_DECAY: 0.0001
  
  # 学习率调度器
  SCHED:
    TYPE: cosine
    WARMUP_STEPS: 256
```

### 优化器选项说明

#### 标准 AdamW (`TYPE: adamw`)
- **优点**: 最佳数值精度，训练速度快
- **缺点**: 内存占用高（模型参数的2倍内存用于优化器状态）
- **推荐场景**: GPU内存充足（如3×A6000 48GB）

#### 8-bit AdamW (`TYPE: adamw8bit`)
- **优点**: 显著降低内存占用（节省约75%优化器内存）
- **缺点**: 需要安装 `bitsandbytes` 库
- **推荐场景**: GPU内存受限（单GPU 24GB或更少）
- **安装方法**: `pip install bitsandbytes`
- **数学等价**: 收敛精度与标准AdamW基本一致

### 注意事项

- **旧的 NUM_WORKERS 配置已废弃**，请使用 `DATALOADER` 部分的配置
- 向后兼容：如果没有 `DATALOADER` 配置，会使用 `TRAIN.NUM_WORKERS`
- **8-bit优化器**: 如果选择 `adamw8bit` 但未安装 `bitsandbytes`，会自动回退到标准 AdamW

---

## 4. 完整配置示例

以下是一个推荐的完整配置：

```yaml
# 基础设置
SEED: 123

# 数据配置
DATA:
  ROOT: data/processed/gopro_spike_unified
  TRAIN_SPLIT: train
  VAL_SPLIT: test
  CROP_SIZE: 256
  CLIP_LEN: 5
  K: 32
  NUM_VOXEL_BINS: 32
  USE_PRECOMPUTED_VOXELS: true
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0

# 模型配置
MODEL:
  USE_SPIKE: true
  VRT_CFG: third_party/VRT/options/deblur/vrt_base.yaml
  CHANNELS_PER_SCALE: [96, 96, 96, 96]
  LAYERS: 4
  SPIKE_TSA:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 49152
  FUSE:
    TYPE: TemporalCrossAttn
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 49152

# 数据加载器配置（推荐配置）
DATALOADER:
  TOTAL_WORKERS: "auto"          # 自动使用80%的CPU核心
  TRAIN_PREFETCH_FACTOR: 4       # 每个worker预取4个批次
  VAL_PREFETCH_FACTOR: 2         # 验证预取2个批次
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true

# 训练配置
TRAIN:
  EPOCHS: 40
  BATCH_SIZE: 1
  VAL_BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 6
  MAX_GRAD_NORM: 1.0
  COMPILE_MODEL: false
  OPTIM:
    LR: 0.0002
    BETAS: [0.9, 0.99]
    WEIGHT_DECAY: 0.0001
  SCHED:
    TYPE: cosine
    WARMUP_STEPS: 256

# 损失函数
LOSS:
  CHARBONNIER:
    DELTA: 0.001
    WEIGHT: 1.0
  VGG_PERCEPTUAL:
    LAYERS: [relu3_3]
    WEIGHT: 0.1

# 日志配置
LOG:
  TENSORBOARD: true
  SAVE_DIR: outputs
  VAL_EVERY_STEPS: 1000
  # Timing Logger
  ENABLE_TIMING_LOG: true
  TIMING_CONSOLE: true
  TIMING_FILE: true
  TIMING_CONSOLE_INTERVAL: 10
  TIMING_FILE_INTERVAL: 50

# 测试配置
TEST:
  BATCH_SIZE: 1
  NUM_WORKERS: 2
```

---

## 5. 性能调优建议

### 对于高端服务器（如 3×A6000, 40核）

```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"          # 32 workers (40*0.8)
  TRAIN_PREFETCH_FACTOR: 4       # 预取4个批次
  PERSISTENT_WORKERS: true       # 必须开启，提升性能

TRAIN:
  BATCH_SIZE: 2                  # 如果显存足够，增大batch size
  GRADIENT_ACCUMULATION_STEPS: 3  # 相应减少累积步数

LOG:
  TIMING_CONSOLE_INTERVAL: 10    # 每10步更新终端
  TIMING_FILE_INTERVAL: 50       # 每50步写文件
```

### 对于显存有限的系统

```yaml
DATALOADER:
  TOTAL_WORKERS: "cpu*0.6"       # 使用60%的CPU
  TRAIN_PREFETCH_FACTOR: 2       # 减少预取，节省内存

TRAIN:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 12  # 增大累积步数补偿

DATA:
  USE_RAM_CACHE: false           # 关闭缓存
  # 或者
  CACHE_SIZE_GB: 2.0             # 减小缓存大小
```

### 对于调试和开发

```yaml
DATALOADER:
  TOTAL_WORKERS: 4               # 固定较小的worker数
  TRAIN_PREFETCH_FACTOR: 2
  PERSISTENT_WORKERS: false      # 关闭以便快速重启

TRAIN:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 1  # 不累积，快速看到结果

LOG:
  ENABLE_TIMING_LOG: true
  TIMING_CONSOLE: true
  TIMING_CONSOLE_INTERVAL: 1     # 每步都显示
  TIMING_FILE_INTERVAL: 10
```

---

## 6. 常见问题

### Q1: LOG 部分不能重复定义

❌ **错误示例**：
```yaml
LOG:
  TENSORBOARD: true
  SAVE_DIR: outputs

TEST:
  BATCH_SIZE: 1

LOG:  # 重复定义！会覆盖前面的配置
  ENABLE_TIMING_LOG: true
```

✅ **正确示例**：
```yaml
LOG:
  TENSORBOARD: true
  SAVE_DIR: outputs
  ENABLE_TIMING_LOG: true  # 所有LOG配置放在一起

TEST:
  BATCH_SIZE: 1
```

### Q2: 如何禁用 Timing Logger

```yaml
LOG:
  ENABLE_TIMING_LOG: false  # 完全禁用
```

或者只禁用终端显示，保留文件日志：

```yaml
LOG:
  ENABLE_TIMING_LOG: true
  TIMING_CONSOLE: false      # 不在终端显示
  TIMING_FILE: true          # 仍然记录到文件
```

### Q3: DataLoader worker 数量如何选择

**经验法则**：
- 每个 GPU 配置 4-12 个 workers
- 总 workers ≤ CPU 核心数的 80%
- 如果数据加载是瓶颈（GPU利用率低），增加 workers
- 如果 CPU 利用率过高，减少 workers

**自动模式推荐**：
```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"  # 让系统自动决定
```

### Q4: 如何查看当前配置的效果

训练启动时会打印：

```
[train] ========== 数据加载配置 ==========
[train] CPU核心数: 40
[train] 总worker数: 32 (建议值，用于所有3个GPU)
[train] 训练DataLoader:
[train]   - batch_size: 1 (每GPU)
[train]   - num_workers: 10 (每GPU)
[train]   - prefetch_factor: 4
[train] 验证DataLoader:
[train]   - num_workers: 5 (每GPU)
[train]   - prefetch_factor: 2
[train] 总计: 30 个训练workers 利用CPU资源
[train] ======================================
```

---

## 7. 配置文件验证

使用 Python 验证配置：

```python
import yaml

with open("configs/deblur/vrt_spike_baseline.yaml") as f:
    config = yaml.safe_load(f)

# 检查必要的字段
assert "LOG" in config
assert "SAVE_DIR" in config["LOG"]
assert "DATALOADER" in config
print("✓ 配置文件格式正确")
```

---

## 参考文档

- [Timing Logger 使用指南](TIMING_LOG.md)
- [性能分析文档](../PERFORMANCE_ANALYSIS.md)

