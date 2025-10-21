# 数据加载与数据集完整指南

> **统一的数据准备、加载和优化指南**  
> 整合自: DATALOADER_GUIDE.md + 数据集接入与使用指南.md

---

## 📋 目录

1. [快速开始](#快速开始)
2. [数据集准备](#数据集准备)
3. [DataLoader配置](#dataloader配置)
4. [实时Spike Voxel加载](#实时spike-voxel加载)
5. [性能优化](#性能优化)
6. [故障排查](#故障排查)

---

## 🚀 快速开始

### 推荐配置（40核CPU，3×GPU）

```yaml
DATA:
  ROOT: data/processed/gopro_spike_unified
  USE_PRECOMPUTED_VOXELS: false  # 实时模式，节省磁盘空间
  NUM_VOXEL_BINS: 5
  SPIKE_DIR: spike
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0

DATALOADER:
  TOTAL_WORKERS: "auto"      # 自动使用32 workers (40*0.8)
  TRAIN_PREFETCH_FACTOR: 4
  VAL_PREFETCH_FACTOR: 2
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true
```

**效果**: 
- 每GPU使用10个workers
- 总共30个workers并行加载
- CPU利用率75%
- 无需预生成voxel缓存，节省大量磁盘空间

---

## 📁 数据集准备

### 1. 数据集结构

#### GoPro-Spike合成数据集

```
data/
├── raw/
│   └── gopro_spike/                    # 原始数据
│       ├── train/
│       │   ├── GOPR0372_07_00/
│       │   │   ├── blur/              # 模糊帧
│       │   │   │   ├── 000001.png
│       │   │   │   └── ...
│       │   │   ├── sharp/             # 清晰帧
│       │   │   │   ├── 000001.png
│       │   │   │   └── ...
│       │   │   └── spike/             # Spike数据
│       │   │       ├── 000001.dat     # 二进制spike流
│       │   │       └── ...
│       │   └── ...
│       └── test/
│           └── ...
│
└── processed/
    └── gopro_spike/
        └── spike_vox/                  # 预计算的voxel (可选)
            ├── train/
            └── test/
```

### 2. 数据格式说明

#### Spike数据格式 (.dat文件)

```python
# .dat 文件格式
spike_data = np.fromfile('000001.dat', dtype=np.uint8)
# shape: (10, 396, 640) = (T, H, W)
# T=10: 10毫秒采样窗口
# H=396, W=640: Spike相机分辨率
```

#### 关键约束

1. **曝光时间窗严格对齐**
   - 模糊帧由33帧等权积分/平均而来
   - FPS=1000（1ms per frame）
   - 曝光窗口 = 33ms

2. **分辨率对齐**
   - Spike分辨率: 396×640
   - RGB分辨率: 720×1280
   - 需要空间对齐后再体素化

3. **体素化与标准化**
   ```python
   # 体素化: spike (T, H, W) → voxel (K, H, W)
   voxel = voxelize(spike, num_bins=K)
   
   # log变换
   voxel = np.log1p(voxel)
   
   # 标准化 (使用训练集统计)
   voxel = (voxel - mean) / std
   ```

4. **DataLoader输出格式**
   ```python
   batch = {
       'blur': [B, T, 3, H, W],      # 模糊序列
       'sharp': [B, T, 3, H, W],     # 清晰序列
       'spike_vox': [B, T, K, H, W]  # Spike体素序列
   }
   ```

### 3. 数据预处理脚本

使用 `scripts/prepare_data.py` 进行数据预处理：

```bash
# 扫描数据并生成索引
python scripts/prepare_data.py \
    --data_root data/raw/gopro_spike \
    --output_root data/processed/gopro_spike_unified \
    --mode scan

# (可选) 预计算voxel缓存
python scripts/prepare_data.py \
    --data_root data/raw/gopro_spike \
    --output_root data/processed/gopro_spike \
    --mode precompute_voxels \
    --num_bins 5
```

---

## 🔧 DataLoader配置

### 1. 核心配置参数

```yaml
DATA:
  # 数据路径
  ROOT: data/processed/gopro_spike_unified
  SPIKE_DIR: spike                     # spike子目录名称
  
  # Voxel配置
  USE_PRECOMPUTED_VOXELS: false        # false=实时生成, true=加载预计算
  NUM_VOXEL_BINS: 5                    # K=5 bins
  
  # 缓存配置
  USE_RAM_CACHE: true                  # 启用LRU缓存
  CACHE_SIZE_GB: 4.0                   # 每个worker 4GB缓存
  
  # 裁剪配置
  TRAIN_CROP_SIZE: 256                 # 训练时裁剪尺寸
  VAL_CROP_SIZE: null                  # 验证时不裁剪

DATALOADER:
  # Worker配置
  TOTAL_WORKERS: "auto"                # 或指定数字，如 32
  TRAIN_PREFETCH_FACTOR: 4             # 每worker预取4个batch
  VAL_PREFETCH_FACTOR: 2               # 验证时预取2个batch
  
  # 性能优化
  PIN_MEMORY: true                     # 使用锁页内存
  PERSISTENT_WORKERS: true             # 保持worker进程
  DROP_LAST: true                      # 丢弃不完整batch
```

### 2. Worker数量自动计算

```python
# TOTAL_WORKERS: "auto" 的计算逻辑
if TOTAL_WORKERS == "auto":
    # 使用80%的CPU核心
    total_workers = int(cpu_count() * 0.8)
    
    # 平均分配给每个GPU
    workers_per_gpu = total_workers // num_gpus
    
    # 确保至少4个workers
    workers_per_gpu = max(4, workers_per_gpu)
```

**示例**:
- 40核CPU, 3 GPUs: `32 workers` → 每GPU 10 workers
- 64核CPU, 4 GPUs: `51 workers` → 每GPU 12 workers

---

## 🔄 实时Spike Voxel加载

### 优势

1. **节省磁盘空间**
   - 不需要预计算和存储voxel
   - 原始.dat文件更紧凑

2. **简化流程**
   - 无需预处理步骤
   - 直接从原始数据训练

3. **灵活性**
   - 可以动态调整voxel bins数量
   - 便于实验不同的体素化参数

### 数据流

```
.dat file (T,H,W) 
    ↓
spike array (10,396,640)
    ↓
voxelize() 
    ↓
voxel (K,396,640)
    ↓
resize to (K,720,1280)
    ↓
random crop to (K,256,256)
    ↓
batch (B,T,K,256,256)
```

### 配置对比

| 模式 | USE_PRECOMPUTED_VOXELS | 磁盘占用 | 加载速度 | 灵活性 |
|------|------------------------|----------|----------|--------|
| **实时模式** | false | 低 (仅.dat) | 略慢 | 高 |
| **预计算模式** | true | 高 (.npy) | 快 | 低 |

**推荐**: 对于大多数场景，使用**实时模式**（false）

### 性能优化

通过以下措施，实时模式的性能损失很小：

1. **LRU缓存**: 缓存最近使用的voxel
2. **多Worker并行**: 30+ workers并行体素化
3. **Prefetch**: 提前加载下一批数据
4. **Persistent Workers**: 避免重复初始化开销

---

## ⚡ 性能优化

### 1. 内存优化

#### LRU缓存配置

```yaml
DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0        # 每worker 4GB
```

**原理**:
- 使用LRU (Least Recently Used) 淘汰策略
- 自动管理内存使用
- 保持高命中率 (通常>80%)

**内存估算**:
```
总内存使用 = workers_per_gpu × CACHE_SIZE_GB × num_gpus
           = 10 × 4GB × 3
           = 120GB
```

#### 内存监控

```python
# 查看缓存统计
# 训练日志会显示:
# Cache stats: hits=850/1000 (85.0%), size=3.2GB/4.0GB
```

### 2. CPU优化

#### Worker数量调优

```yaml
# 根据系统负载调整
DATALOADER:
  TOTAL_WORKERS: 32      # 或 "auto"
```

**调优建议**:
- **CPU密集型系统**: `workers = cpu_count * 0.8`
- **内存受限系统**: `workers = cpu_count * 0.5`
- **共享服务器**: 手动指定合理数量

#### Prefetch优化

```yaml
DATALOADER:
  TRAIN_PREFETCH_FACTOR: 4    # 增大可提高GPU利用率
  VAL_PREFETCH_FACTOR: 2      # 验证时较小即可
```

**权衡**:
- 增大 → GPU利用率↑，内存使用↑
- 减小 → GPU可能等待，内存使用↓

### 3. GPU优化

#### Pin Memory

```yaml
DATALOADER:
  PIN_MEMORY: true      # 使用锁页内存，加速CPU→GPU传输
```

**效果**:
- 加速数据传输 30-50%
- 增加 ~2GB 主机内存使用

#### Persistent Workers

```yaml
DATALOADER:
  PERSISTENT_WORKERS: true    # 保持worker进程，避免重复初始化
```

**效果**:
- 避免每个epoch重启workers
- 保持缓存和状态
- 减少初始化开销

### 4. 典型配置方案

#### 高性能配置 (推荐)

```yaml
# 40核CPU, 3×A100 GPUs, 256GB RAM
DATA:
  USE_PRECOMPUTED_VOXELS: false
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0

DATALOADER:
  TOTAL_WORKERS: 30              # 每GPU 10 workers
  TRAIN_PREFETCH_FACTOR: 4
  PIN_MEMORY: true
  PERSISTENT_WORKERS: true
```

**预期性能**: 
- 吞吐量: 2.0-2.5 samples/s
- GPU利用率: >95%
- CPU利用率: 70-80%

#### 内存受限配置

```yaml
# 限制内存使用
DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 2.0             # 减小缓存

DATALOADER:
  TOTAL_WORKERS: 12              # 减少workers
  TRAIN_PREFETCH_FACTOR: 2       # 减小prefetch
```

#### 调试配置

```yaml
# 快速迭代，减少开销
DATALOADER:
  TOTAL_WORKERS: 4
  TRAIN_PREFETCH_FACTOR: 2
  PERSISTENT_WORKERS: false      # 便于修改代码
```

---

## 🔍 故障排查

### 1. 常见问题

#### Q1: DataLoader速度慢，GPU利用率低

**症状**:
```
GPU Util: 30-50%
samples/s: <1.0
```

**可能原因**:
1. Worker数量不足
2. Prefetch factor太小
3. 磁盘I/O瓶颈

**解决方案**:
```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"          # 增加workers
  TRAIN_PREFETCH_FACTOR: 4       # 增大prefetch
  PIN_MEMORY: true               # 启用pin memory
```

#### Q2: 内存不足 (OOM)

**症状**:
```
RuntimeError: Out of memory
或
系统内存使用100%
```

**解决方案**:
```yaml
DATA:
  CACHE_SIZE_GB: 2.0             # 减小缓存
  
DATALOADER:
  TOTAL_WORKERS: 16              # 减少workers
  TRAIN_PREFETCH_FACTOR: 2       # 减小prefetch
```

#### Q3: Spike文件读取失败

**症状**:
```
FileNotFoundError: spike/000001.dat
```

**检查步骤**:
```bash
# 1. 检查数据结构
ls data/processed/gopro_spike_unified/train/GOPR*/spike/

# 2. 检查配置
# CONFIG.yaml:
DATA:
  SPIKE_DIR: spike               # 确保目录名正确
  
# 3. 检查文件权限
chmod -R u+r data/processed/gopro_spike_unified/
```

#### Q4: Voxel维度不匹配

**症状**:
```
RuntimeError: Expected spike_vox shape [..., 5, H, W], got [..., 32, H, W]
```

**解决方案**:
```yaml
DATA:
  NUM_VOXEL_BINS: 5              # 确保与模型配置一致
```

### 2. 性能诊断

#### 检查数据加载性能

```python
# 添加到train.py
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

**目标**: data_time < 10ms

#### 监控系统资源

```bash
# GPU监控
watch -n 1 nvidia-smi

# CPU监控
htop

# 内存监控
watch -n 1 free -h

# 磁盘I/O监控
iostat -x 1
```

### 3. 调试模式

启用详细日志：

```yaml
# 添加到配置文件
DEBUG:
  DATALOADER_VERBOSE: true       # 详细加载日志
  CACHE_STATS: true              # 缓存统计
  TIMING_ANALYSIS: true          # 计时分析
```

或使用诊断脚本：

```bash
# 测试DataLoader配置
python test_dataloader_config.py

# 诊断内存使用
python docs/diagnose_memory.py
```

---

## 📚 相关文档

- **[配置指南](CONFIG_GUIDE.md)** - 完整配置参数说明
- **[训练指南](QUICK_START.md)** - 训练流程和命令
- **[优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - 性能调优
- **[架构文档](ARCHITECTURE.md)** - 模型架构和数据流

---

## 📝 数据接入规范（供应方数据封装）

### Vendor脚本封装

对于第三方数据集（如GoPro），使用vendor封装模式：

```
src/data/vendors/gopro/
├── original/                           # 供应方原始脚本（只读）
│   ├── deblur_gopro_sequence.py
│   ├── deblur_gopro_dataset.py
│   └── deblur_read_gopro_test_data.py
└── io_gopro.py                         # 我们的薄封装
```

**原则**:
1. **不修改原始脚本** - 放在`original/`目录
2. **薄封装层** - `io_gopro.py`提供统一接口
3. **标准化输出** - 转换为项目标准格式

### 统一接口

```python
# io_gopro.py
from .original.deblur_gopro_sequence import GoProSequence

class GoProDataInterface:
    """统一的GoPro数据接口"""
    
    def read_sequence(self, path: str) -> Dict:
        """读取序列数据
        
        Returns:
            {
                'blur': np.ndarray,    # (T, H, W, 3)
                'sharp': np.ndarray,   # (T, H, W, 3)
                'spike': np.ndarray    # (T, H_s, W_s)
            }
        """
        pass
```

---

## ✅ 检查清单

### 数据准备完成检查

- [ ] 原始数据已下载到 `data/raw/gopro_spike/`
- [ ] 目录结构符合规范（train/test/blur/sharp/spike）
- [ ] Spike文件格式正确（.dat，uint8）
- [ ] 已运行数据扫描脚本
- [ ] 配置文件中路径正确

### DataLoader配置检查

- [ ] `DATA.ROOT` 路径正确
- [ ] `DATA.SPIKE_DIR` 与实际目录匹配
- [ ] `NUM_VOXEL_BINS` 与模型一致
- [ ] Worker数量适合系统配置
- [ ] 缓存大小合理（不超过可用内存）

### 性能检查

- [ ] samples/s > 2.0
- [ ] GPU利用率 > 90%
- [ ] data_time < 10ms
- [ ] 内存使用稳定（无泄漏）
- [ ] CPU利用率 < 85%

---

**最后更新**: 2025-10-21  
**适用版本**: VRT+Spike v1.0+


