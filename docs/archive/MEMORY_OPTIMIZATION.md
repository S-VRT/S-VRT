# Memory Optimization Guide

## 问题分析：为什么会出现内存爆炸？

### 原始问题

训练时系统内存使用快速增长至256GB物理内存上限，导致OOM错误。

**根本原因：**

1. **数据集规模大**：GoPro+Spike统一数据集约2,222帧
   - 每帧数据：blur(720×1280×3×4) + sharp(720×1280×3×4) + voxel(32×720×1280×4) ≈ 133.59 MB
   - 总数据量：~290 GB

2. **DataLoader worker内存复制**：
   - 每个worker进程fork时复制父进程内存
   - 36 workers × 3 GPUs = 108个进程
   - 理论最坏情况：290GB × 108 ≈ 31TB！

3. **无限制缓存策略**：
   - 旧版本使用简单的字典缓存，试图加载所有数据到RAM
   - 没有内存限制，导致爆炸性增长

## 解决方案：LRU缓存

### 1. LRU (Least Recently Used) 缓存

**原理：**
- 设置内存上限（默认50GB）
- 缓存最近使用的数据
- 当达到上限时，自动淘汰最久未使用的数据
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
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key].clone()
        return None
    
    def put(self, key: str, value: torch.Tensor) -> None:
        # Evict LRU items until we have enough space
        while self.current_memory + item_size > self.max_memory_bytes:
            self.cache.popitem(last=False)  # Remove oldest
        self.cache[key] = value
```

### 2. Worker数量优化

**问题：**
- 36 workers太多，导致：
  - 内存复制过多
  - CPU调度开销大
  - 不利于缓存命中

**解决：**
- 降低到12 workers
- 对于A6000 × 3的配置，每个GPU 4个workers即可
- 配合persistent_workers和prefetch_factor实现高效加载

### 3. 内存监控

**功能：**
- 启动时和每100步记录内存使用
- CPU内存：进程和系统级别
- GPU内存：已分配和保留
- 超过90%时发出警告

**示例输出：**
```
[After dataset creation] Memory - Process: 12345.6MB (4.8%), System: 48.3/256.0GB (18.9%), GPU: 2048.0MB allocated, 2560.0MB reserved
[Step 100] Memory - Process: 15678.9MB (6.1%), System: 125.4/256.0GB (49.0%), GPU: 8192.0MB allocated, 9216.0MB reserved
⚠️  WARNING: System memory usage is critically high (91.2%)!
   Consider reducing CACHE_SIZE_GB or NUM_WORKERS in config.
```

## 配置参数

### 关键配置项

```yaml
DATA:
  USE_RAM_CACHE: true           # 启用LRU缓存（推荐）
  CACHE_SIZE_GB: 50.0           # 缓存大小限制（GB）
  
TRAIN:
  NUM_WORKERS: 12               # DataLoader workers数量
  PREFETCH_FACTOR: 4            # 预取批次数
```

### 参数调优建议

**CACHE_SIZE_GB：**
- 256GB系统：50-80GB（留足空间给模型和系统）
- 128GB系统：20-40GB
- 64GB系统：10-20GB

**NUM_WORKERS：**
- 单GPU：4-8
- 多GPU：每GPU 4个，总计 num_gpus × 4
- 最大不超过CPU核心数的50%

## 性能对比

### 内存使用

| 配置 | 内存占用 | 风险 |
|------|---------|------|
| 无缓存 | ~10GB | I/O瓶颈，训练慢 |
| 全量缓存 | ~290GB × workers | OOM崩溃 |
| **LRU缓存(50GB)** | **~50GB稳定** | **✅ 最佳平衡** |

### 训练速度预期

- **无缓存**：0.4 samples/s（I/O受限）
- **LRU缓存**：1.5-2.5 samples/s（目标达成）
- **全量缓存**：理论最快，但会OOM

## 监控和调试

### 1. 查看缓存统计

训练时会自动输出缓存命中率：

```python
# 在数据集对象上调用
train_set.print_cache_stats()
# 输出：
# [SpikeDeblurDataset] Cache Stats:
#   - Size: 374 items
#   - Memory: 49876.3 / 51200.0 MB
#   - Hit Rate: 82.45% (8245 hits / 1755 misses)
```

### 2. 实时内存监控

```bash
# 安装psutil（如未安装）
pip install psutil

# 训练时自动监控（每100步）
python src/train.py --config configs/deblur/vrt_spike_baseline.yaml
```

### 3. 系统级监控

```bash
# 实时查看内存
watch -n 1 'free -h && nvidia-smi'

# 查看进程内存
ps aux --sort=-rss | head -20
```

## 最佳实践

### ✅ 推荐

1. **启用LRU缓存**：USE_RAM_CACHE=true + CACHE_SIZE_GB=50
2. **合理设置workers**：NUM_WORKERS=12（3 GPUs × 4）
3. **持久化workers**：persistent_workers=True
4. **异步拷贝**：pin_memory=True + non_blocking=True
5. **定期监控**：观察内存增长趋势

### ❌ 避免

1. 设置CACHE_SIZE_GB > 物理内存的50%
2. NUM_WORKERS过多（>CPU核心数）
3. 禁用内存监控（需要psutil）
4. 同时运行多个训练任务

## 故障排除

### 问题：仍然OOM

**可能原因：**
1. CACHE_SIZE_GB设置过大
2. NUM_WORKERS过多
3. 模型batch_size过大

**解决：**
```yaml
DATA:
  CACHE_SIZE_GB: 30.0  # 降低缓存大小

TRAIN:
  NUM_WORKERS: 8       # 减少workers
  BATCH_SIZE: 1        # 如需要，降低batch size
```

### 问题：训练速度慢

**检查：**
1. 缓存命中率是否过低？调用`print_cache_stats()`查看
2. 是否禁用了缓存？确认USE_RAM_CACHE=true
3. workers数量是否过少？推荐每GPU 4个

**优化：**
```yaml
DATA:
  CACHE_SIZE_GB: 80.0  # 如有余量，增加缓存

TRAIN:
  NUM_WORKERS: 16      # 适当增加workers
  PREFETCH_FACTOR: 6   # 增加预取
```

### 问题：缓存命中率低

**原因：**
- 数据访问模式随机性强
- 缓存大小不足以覆盖一个epoch的常用数据

**解决：**
1. 增加CACHE_SIZE_GB
2. 检查shuffle策略（过度shuffle降低局部性）
3. 考虑按序列分组采样

## 参考资料

- [PyTorch DataLoader最佳实践](https://pytorch.org/docs/stable/data.html#multi-process-data-loading)
- [内存管理优化技巧](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- 项目issue: #123 "Memory explosion during training"

---

**更新日期：** 2025-10-14  
**相关文件：**
- `src/data/datasets/spike_deblur_dataset.py` - LRU缓存实现
- `configs/deblur/vrt_spike_baseline.yaml` - 配置示例
- `src/train.py` - 内存监控集成

