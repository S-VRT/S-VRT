# VRT-Spike 性能优化完整指南

> **统一的性能和内存优化方案**  
> 整合自: OPTIMIZATION_GUIDE.md + MEMORY_OPTIMIZATION.md  
> 最后更新: 2025-10-21

---

## 📋 目录

1. [快速诊断](#快速诊断)
2. [内存优化](#内存优化)
3. [模型性能优化](#模型性能优化)
4. [DataLoader优化](#dataloader优化)
5. [综合优化方案](#综合优化方案)
6. [监控和调试](#监控和调试)
7. [故障排除](#故障排除)

---

## 🚀 快速诊断

### 1. 识别性能瓶颈

**症状检查清单**:

| 症状 | 可能原因 | 跳转章节 |
|------|---------|---------|
| 内存快速增长至OOM | DataLoader内存爆炸 | [内存优化](#内存优化) |
| GPU利用率低 (<50%) | 数据加载瓶颈 | [DataLoader优化](#dataloader优化) |
| 前向传播慢 (>1500ms) | 模型计算瓶颈 | [模型性能优化](#模型性能优化) |
| 训练速度慢 (<1 sample/s) | 综合问题 | [综合优化方案](#综合优化方案) |

### 2. 性能分析工具

```bash
# 运行性能分析
python analyze_performance.py outputs/logs/train_<timestamp>.log

# 内存诊断
python docs/diagnose_memory.py
```

---

## 💾 内存优化

### 问题诊断

#### 为什么会出现内存爆炸？

**根本原因：**

1. **数据集规模大**
   - GoPro+Spike数据集约2,222帧
   - 每帧: blur(720×1280×3×4) + sharp(720×1280×3×4) + voxel(32×720×1280×4) ≈ 133.59 MB
   - 总数据量: ~290 GB

2. **DataLoader worker内存复制**
   - 每个worker进程fork时复制父进程内存
   - 36 workers × 3 GPUs = 108个进程
   - 理论最坏情况: 290GB × 108 ≈ 31TB！

3. **无限制缓存策略**
   - 简单字典缓存试图加载所有数据到RAM
   - 没有内存限制，导致爆炸性增长

### 解决方案：LRU缓存

#### 1. LRU (Least Recently Used) 缓存

**原理：**
- 设置内存上限（默认50GB）
- 缓存最近使用的数据
- 达到上限时自动淘汰最久未使用的数据
- 保持高命中率同时防止内存爆炸

**实现：**
```python
class LRUCache:
    def __init__(self, max_memory_gb: float = 50.0):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_memory = 0
    
    def get(self, key: str) -> torch.Tensor | None:
        if key in self.cache:
            self.cache.move_to_end(key)  # 标记为最近使用
            return self.cache[key].clone()
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        # 淘汰LRU项直到有足够空间
        while self.current_memory + item_size > self.max_memory_bytes:
            self.cache.popitem(last=False)  # 删除最旧项
        self.cache[key] = value
```

#### 2. 配置参数

```yaml
DATA:
  USE_RAM_CACHE: true           # 启用LRU缓存（推荐）
  CACHE_SIZE_GB: 50.0           # 缓存大小限制（GB）
  
DATALOADER:
  TOTAL_WORKERS: 12             # DataLoader workers数量
  TRAIN_PREFETCH_FACTOR: 4      # 预取批次数
```

#### 3. 参数调优建议

**CACHE_SIZE_GB：**
- 256GB系统: 50-80GB（留足空间给模型和系统）
- 128GB系统: 20-40GB
- 64GB系统: 10-20GB

**TOTAL_WORKERS：**
- 单GPU: 4-8
- 多GPU: 每GPU 4个，总计 `num_gpus × 4`
- 最大不超过CPU核心数的50%

#### 4. 性能对比

| 配置 | 内存占用 | 训练速度 | 风险 |
|------|---------|---------|------|
| 无缓存 | ~10GB | 0.4 samples/s | I/O瓶颈 |
| 全量缓存 | ~290GB × workers | 理论最快 | OOM崩溃 |
| **LRU缓存(50GB)** | **~50GB稳定** | **1.5-2.5 samples/s** | **✅ 最佳** |

---

## ⚡ 模型性能优化

### 性能瓶颈分析

**基于实测数据（111次前向传播，平均1789.72ms）：**

1. **🔴 Stage 8 (重建层)** - 占总耗时 **32.8%** (586.90ms)
2. **🔴 Stage 2 (1/2x分辨率)** - 占总耗时 **15.2%** (271.33ms)
3. **🟡 Stage 1 融合** - 占总耗时 **8.6%** (154.02ms)

**优化这三个阶段可减少约50%的计算时间**

---

### 优化策略

#### 第一优先级：混合精度训练（立即实施）✅

**配置：**
```yaml
TRAIN:
  AMP_ENABLED: true        # 自动混合精度
  AMP_OPT_LEVEL: "O1"      # O1: 保持大部分操作在FP16
```

**代码修改 (src/train.py)：**
```python
from torch.cuda.amp import autocast, GradScaler

def train(...):
    # 初始化 GradScaler
    amp_enabled = cfg.get("TRAIN", {}).get("AMP_ENABLED", False)
    scaler = GradScaler() if amp_enabled else None
    
    for step, batch in enumerate(train_loader):
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
```

**预期效果：** 减少 30-40% 计算时间，降低显存使用

---

#### 第二优先级：优化Stage 8重建层

**问题：** Stage 8 在全分辨率上操作，包含大量卷积和重建模块

**方案1：减少Transformer块深度**

```python
# 修改 src/train.py 中的 VRT 初始化：
vrt = VRT(
    upscale=1,
    in_chans=3,
    out_chans=3,
    img_size=img_size_cfg,
    window_size=window_size_cfg,
    embed_dims=embed_dims_cfg,
    depths=[8, 8, 8, 8, 4, 4, 4, 4],  # Stage 8 从8降到4
    #      ↑ Stage 1-7 保持不变   ↑ 减半
    use_checkpoint_attn=True,
    use_checkpoint_ffn=True,
)
```

**预期效果：** 减少 15-25% Stage 8 耗时

**方案2：优化窗口大小**

```yaml
MODEL:
  WINDOW_SIZE: 6  # 从 8 降到 6（减少约 44% 的注意力计算量）
```

**预期效果：** 减少 20-30% Stage 8 耗时，精度损失通常<0.5dB

---

#### 第三优先级：优化Stage 2

**问题：** Stage 2 在 1/2x 分辨率 (128×128) 上操作，Transformer块较多

**方案：减少Stage 2的块数量**

```python
vrt = VRT(
    depths=[8, 6, 8, 8, 4, 4, 4, 4],  # Stage 2 从8降到6
    #         ↑ 减少2个块
    # ...
)
```

**预期效果：** 减少 25% Stage 2 耗时

---

#### 第四优先级：优化融合模块

**方案：增大融合chunk_size**

```yaml
MODEL:
  FUSE:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536  # 从 49152 增加
    CHUNK_SIZE: 96           # 从 64 增加
    CHUNK_SHAPE: "square"
```

**预期效果：** 减少 20-30% 融合耗时

---

### 高级优化技术

#### 1. torch.compile() (PyTorch 2.0+)

```python
# 在模型创建后添加：
if torch.__version__ >= "2.0.0" and cfg.get("TRAIN", {}).get("COMPILE_MODEL", False):
    model = torch.compile(model, mode="max-autotune")
```

#### 2. 渐进式训练策略

```yaml
# 先用小分辨率训练，再微调大分辨率
DATA:
  TRAIN_CROP_SIZE: 128  # 前40 epochs
  # TRAIN_CROP_SIZE: 256  # 后40 epochs
```

#### 3. 知识蒸馏

```python
# 创建轻量级学生模型
vrt_student = VRT(
    depths=[4, 4, 4, 4, 2, 2, 2, 2],  # 深度减半
    # ...
)
```

---

## 🔄 DataLoader优化

### 核心配置

```yaml
DATA:
  USE_PRECOMPUTED_VOXELS: false  # 实时模式，节省磁盘
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0

DATALOADER:
  TOTAL_WORKERS: "auto"          # 或指定数字，如 32
  TRAIN_PREFETCH_FACTOR: 4       # 每worker预取4个batch
  VAL_PREFETCH_FACTOR: 2
  PIN_MEMORY: true               # 使用锁页内存
  PERSISTENT_WORKERS: true       # 保持worker进程
  DROP_LAST: true
```

### Worker数量自动计算

```python
# TOTAL_WORKERS: "auto" 的计算逻辑
if TOTAL_WORKERS == "auto":
    total_workers = int(cpu_count() * 0.8)  # 使用80%的CPU核心
    workers_per_gpu = total_workers // num_gpus
    workers_per_gpu = max(4, workers_per_gpu)  # 至少4个
```

**示例：**
- 40核CPU, 3 GPUs: `32 workers` → 每GPU 10 workers
- 64核CPU, 4 GPUs: `51 workers` → 每GPU 12 workers

### 性能调优建议

#### CPU密集型系统
```yaml
DATALOADER:
  TOTAL_WORKERS: 32           # cpu_count * 0.8
  TRAIN_PREFETCH_FACTOR: 4
```

#### 内存受限系统
```yaml
DATALOADER:
  TOTAL_WORKERS: 16           # cpu_count * 0.5
  TRAIN_PREFETCH_FACTOR: 2
DATA:
  CACHE_SIZE_GB: 2.0          # 减小缓存
```

---

## 🎯 综合优化方案

### 方案 A：保守优化（精度优先）

**推荐场景：** 首次优化，重视精度稳定性

**配置文件 (configs/deblur/vrt_spike_opt.yaml)：**
```yaml
MODEL:
  VRT_DEPTHS: [8, 7, 8, 8, 4, 4, 4, 6]  # Stage 2: 8→7, Stage 8: 8→6
  WINDOW_SIZE: 7                         # 从 8 降到 7
  FUSE:
    HEADS: 4
    ADAPTIVE_CHUNK: true
    MAX_BATCH_TOKENS: 65536
    CHUNK_SIZE: 80                       # 从 64 增加
    CHUNK_SHAPE: "square"

DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 50.0

DATALOADER:
  TOTAL_WORKERS: "auto"
  TRAIN_PREFETCH_FACTOR: 4
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true

TRAIN:
  AMP_ENABLED: true                      # 启用混合精度
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 1
```

**预期提升：** 
- 性能: +25-35%
- 耗时: 1789ms → 1200-1300ms
- 精度影响: <0.3dB

---

### 方案 B：激进优化（性能优先）

**推荐场景：** 需要快速迭代，可接受适度精度损失

**配置文件 (configs/deblur/vrt_spike_fast.yaml)：**
```yaml
MODEL:
  VRT_DEPTHS: [8, 6, 6, 8, 4, 4, 4, 4]  # Stage 2: 8→6, Stage 8: 8→4
  WINDOW_SIZE: 6                         # 从 8 降到 6
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

DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 50.0
  TRAIN_CROP_SIZE: 128                   # 先用小尺寸训练

DATALOADER:
  TOTAL_WORKERS: "auto"
  TRAIN_PREFETCH_FACTOR: 6               # 增大预取
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true

TRAIN:
  AMP_ENABLED: true
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 2                          # 混合精度下可增大
```

**预期提升：**
- 性能: +45-60%
- 耗时: 1789ms → 800-1000ms
- 精度影响: 0.5-1.0dB

---

### 方案 C：内存受限优化

**推荐场景：** 64-128GB内存系统，或共享服务器

**配置文件 (configs/deblur/vrt_spike_lowmem.yaml)：**
```yaml
MODEL:
  VRT_DEPTHS: [8, 7, 8, 8, 4, 4, 4, 6]
  WINDOW_SIZE: 7

DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 20.0                    # 减小缓存

DATALOADER:
  TOTAL_WORKERS: 8                       # 减少workers
  TRAIN_PREFETCH_FACTOR: 2               # 减小预取
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true

TRAIN:
  AMP_ENABLED: true
  AMP_OPT_LEVEL: "O1"
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 4         # 梯度累积补偿小batch
```

---

### 实施步骤

#### Step 1: 创建优化配置文件

```bash
# 创建保守优化版本
cp configs/deblur/vrt_spike_baseline.yaml \
   configs/deblur/vrt_spike_opt.yaml

# 创建激进优化版本
cp configs/deblur/vrt_spike_baseline.yaml \
   configs/deblur/vrt_spike_fast.yaml
```

#### Step 2: 修改 VRT 初始化代码

在 `src/train.py` 的 `create_model()` 函数中：

```python
def create_model(cfg: dict, device: torch.device) -> nn.Module:
    # 从配置读取 VRT depths（如果未指定则使用默认值）
    vrt_depths = cfg.get("MODEL", {}).get("VRT_DEPTHS", [8, 8, 8, 8, 4, 4, 4, 8])
    
    vrt = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=img_size_cfg,
        window_size=window_size_cfg,
        embed_dims=embed_dims_cfg,
        depths=vrt_depths,  # ← 使用配置中的depths
        use_checkpoint_attn=True,
        use_checkpoint_ffn=True,
    )
    # ... 其余代码 ...
```

#### Step 3: 运行优化配置

```bash
# 测试保守优化版本
python src/train.py --config configs/deblur/vrt_spike_opt.yaml

# 测试激进优化版本
python src/train.py --config configs/deblur/vrt_spike_fast.yaml

# 运行性能分析
python analyze_performance.py outputs/logs/train_<timestamp>.log
```

---

## 📊 监控和调试

### 1. 内存监控

#### 自动监控

训练时自动输出内存统计（每100步）：

```
[After dataset creation] Memory - Process: 12345.6MB (4.8%), System: 48.3/256.0GB (18.9%)
[Step 100] Memory - Process: 15678.9MB (6.1%), System: 125.4/256.0GB (49.0%)
⚠️  WARNING: System memory usage is critically high (91.2%)!
   Consider reducing CACHE_SIZE_GB or TOTAL_WORKERS in config.
```

#### 缓存统计

```python
# 在数据集对象上调用
train_set.print_cache_stats()
# 输出：
# [SpikeDeblurDataset] Cache Stats:
#   - Size: 374 items
#   - Memory: 49876.3 / 51200.0 MB
#   - Hit Rate: 82.45% (8245 hits / 1755 misses)
```

#### 系统级监控

```bash
# 实时查看内存和GPU
watch -n 1 'free -h && nvidia-smi'

# 查看进程内存排序
ps aux --sort=-rss | head -20

# 磁盘I/O监控
iostat -x 1
```

### 2. 性能监控

#### 数据加载时间

```python
# 添加到 train.py
import time

data_time = 0
start = time.time()

for batch in train_loader:
    data_time += time.time() - start
    
    # ... training code ...
    
    if step % 50 == 0:
        print(f"Avg data time: {data_time/50*1000:.1f}ms")
        data_time = 0
    
    start = time.time()
```

**目标：** data_time < 10ms

#### GPU利用率监控

```bash
# 持续监控
nvidia-smi dmon -s u

# 详细监控
watch -n 1 nvidia-smi
```

**目标：** GPU利用率 > 90%

---

## 🔍 故障排除

### 问题1：内存不足 (OOM)

**症状：**
```
RuntimeError: Out of memory
或
系统内存使用100%，进程被kill
```

**可能原因：**
1. CACHE_SIZE_GB设置过大
2. TOTAL_WORKERS过多
3. batch_size过大

**解决方案：**
```yaml
DATA:
  CACHE_SIZE_GB: 30.0  # 降低缓存大小

DATALOADER:
  TOTAL_WORKERS: 8     # 减少workers

TRAIN:
  BATCH_SIZE: 1        # 降低batch size
  GRADIENT_ACCUMULATION_STEPS: 4  # 梯度累积补偿
```

---

### 问题2：GPU利用率低 (<50%)

**症状：**
```
GPU Util: 30-50%
samples/s: <1.0
```

**可能原因：**
1. Worker数量不足
2. Prefetch factor太小
3. 磁盘I/O瓶颈

**解决方案：**
```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"          # 增加workers
  TRAIN_PREFETCH_FACTOR: 4       # 增大prefetch
  PIN_MEMORY: true               # 启用pin memory
  PERSISTENT_WORKERS: true       # 保持workers

DATA:
  USE_RAM_CACHE: true            # 启用缓存
  CACHE_SIZE_GB: 50.0
```

---

### 问题3：训练速度慢但GPU满载

**症状：**
```
GPU Util: >90%
samples/s: <1.5
前向传播时间: >1500ms
```

**解决方案：** 参考[模型性能优化](#模型性能优化)

1. **启用混合精度** (最优先)
   ```yaml
   TRAIN:
     AMP_ENABLED: true
   ```

2. **优化模型结构**
   - 减少depths
   - 降低window_size
   - 增大chunk_size

---

### 问题4：缓存命中率低 (<60%)

**症状：**
```
Cache hit rate: 45.2% (4520 hits / 5480 misses)
```

**原因：**
- 缓存大小不足
- 数据访问模式随机性强

**解决方案：**
```yaml
DATA:
  CACHE_SIZE_GB: 80.0  # 增加缓存（如有余量）

# 或调整采样策略（在代码中）
# - 按序列分组采样
# - 减少shuffle的随机性
```

---

### 问题5：混合精度训练不稳定

**症状：**
```
Loss突然变成NaN
或梯度爆炸/消失
```

**解决方案：**
```python
# 使用更保守的scaler初始化
scaler = GradScaler(init_scale=2.**10)  # 降低初始缩放

# 或使用O2级别
TRAIN:
  AMP_OPT_LEVEL: "O2"  # 更激进，但可能更不稳定
```

---

## 📈 预期性能对比

| 配置 | 前向传播耗时 | 训练速度 | 相对提升 | 精度影响 | 内存使用 |
|------|-------------|---------|---------|---------|---------|
| **基线** | 1789ms | 1.2 samples/s | - | - | ~60GB |
| **仅混合精度** | 1100-1200ms | 1.8 samples/s | +35-40% | <0.1dB | ~45GB |
| **保守优化** | 1200-1300ms | 2.0 samples/s | +30-35% | <0.3dB | ~50GB |
| **激进优化** | 800-1000ms | 2.5 samples/s | +45-60% | 0.5-1.0dB | ~50GB |
| **内存受限优化** | 1300-1400ms | 1.8 samples/s | +25-30% | <0.3dB | ~30GB |

---

## ✅ 验证清单

训练完成后验证：

### 功能验证
- [ ] 训练损失曲线稳定收敛
- [ ] 验证集 PSNR/SSIM 在可接受范围内
- [ ] 无NaN或Inf值出现
- [ ] checkpoint正确保存和加载

### 性能验证
- [ ] 训练速度（samples/s）达到目标
- [ ] GPU利用率 > 90%
- [ ] data_time < 10ms
- [ ] 缓存命中率 > 75%

### 资源验证
- [ ] GPU显存使用降低或可接受
- [ ] 系统内存使用稳定（无泄漏）
- [ ] CPU利用率 < 85%
- [ ] 磁盘I/O无明显瓶颈

---

## 🎯 最佳实践总结

### ✅ 推荐做法

1. **优先启用混合精度训练** - 最简单、收益最大
2. **使用LRU缓存** - 平衡内存和性能
3. **自动计算worker数量** - 适应不同硬件
4. **启用persistent workers** - 避免重复初始化
5. **定期监控性能** - 及时发现问题
6. **渐进式优化** - 先保守再激进

### ❌ 避免做法

1. 设置CACHE_SIZE_GB > 物理内存的50%
2. TOTAL_WORKERS > CPU核心数
3. 同时运行多个训练任务（在内存受限系统）
4. 禁用内存监控
5. 过度优化导致精度严重下降

---

## 📚 相关资源

- **[配置指南](CONFIG_GUIDE.md)** - 完整配置参数说明
- **[数据加载指南](DATA_GUIDE.md)** - DataLoader详细配置
- **[训练指南](QUICK_START.md)** - 训练流程和命令
- **[架构文档](ARCHITECTURE.md)** - 模型架构和数据流

### 外部资源

- [VRT 官方仓库](https://github.com/JingyunLiang/VRT)
- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- [PyTorch DataLoader最佳实践](https://pytorch.org/docs/stable/data.html#multi-process-data-loading)

---

## 📝 下一步行动

1. ✅ **立即实施：** 启用混合精度训练（最简单，收益最大）
2. ⏭️ **短期目标：** 创建并测试保守优化配置
3. 🎯 **中期目标：** 测试激进优化配置，找到精度-性能最佳平衡点
4. 🚀 **长期目标：** 探索模型架构改进和知识蒸馏

---

**最后更新**: 2025-10-21  
**适用版本**: VRT+Spike v1.0+  
**维护者**: VRT-Spike Team


