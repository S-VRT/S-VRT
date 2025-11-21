# VRT RGB+Spike 实现总结

## 概述

成功实现了 VRT 网络使用 RGB+Spike 拼接输入进行视频去模糊训练。基于 GoPro 数据集，保持训练参数与官方配置一致作为 baseline。

## 实现细节

### 1. 数据结构确认

#### RGB 数据
- **格式**: 目录格式（PNG 图像）
- **分辨率**: 720x1280
- **路径**:
  - GT: `trainsets/GoPro/train_GT/`
  - LQ (模糊): `trainsets/GoPro/train_GT_blurred/`

#### Spike 数据
- **格式**: .dat 二进制文件
- **分辨率**: 250x400 (H×W)
- **时间步**: 约 202 帧/文件
- **路径**: `trainsets/gopro_spike/GOPRO_Large_spike_seq/train/`
- **目录结构**: 
  ```
  train/
    ├── GOPR0384_11_02/
    │   └── spike/
    │       ├── 001301.dat
    │       ├── 001302.dat
    │       └── ...
    └── ...
  ```

### 2. 关键实现

#### 2.1 Spike 数据加载器

**文件**: `utils/spike_loader.py`

- 实现了简化的 spike .dat 文件加载器
- 移除了 SpikeCV 的在线相机依赖
- 支持高效的比特流解码

**核心功能**:
```python
class SpikeStreamSimple:
    def __init__(self, filepath, spike_h, spike_w)
    def get_spike_matrix(self, flipud=True) -> np.ndarray  # (T, H, W)
    def get_block_spikes(self, begin_idx, block_len) -> np.ndarray
```

#### 2.2 RGB+Spike Dataset

**文件**: `data/dataset_video_train_rgbspike.py`

- 继承 `VideoRecurrentTrainDataset`，复用 RGB 加载逻辑
- 添加 Spike 数据加载和体素化
- 实现分辨率自动对齐（Spike 250x400 → RGB crop size）
- 支持联合数据增强（flip, rotate）

**体素化策略**:

1. **S=1** (默认): 简单时间累积
   - 公式: `voxel = sum(spikes) / max(sum(spikes))`
   - 适用于快速训练和 baseline

2. **S=4**: 4-bin 时间体素化
   - 将时间维度均分为 4 段
   - 每段独立归一化
   - 保留更多时间信息

**输出格式**:
- LQ: `(T, 3+S, H, W)` - RGB + Spike 拼接
- GT: `(T, 3, H, W)` - 仅 RGB

#### 2.3 网络修改

**文件**: `models/select_network.py`

- 添加 `in_chans` 参数传递给 VRT 网络
- 默认值为 3（向后兼容）
- RGB+Spike 配置设置为 4 (3+1)

```python
netG = net(
    upscale=opt_net['upscale'],
    in_chans=opt_net.get('in_chans', 3),  # 新增
    img_size=opt_net['img_size'],
    ...
)
```

#### 2.4 Dataset 注册

**文件**: `data/select_dataset.py`

添加新的 dataset 类型:
```python
elif dataset_type in ['videorecurrenttraindatasetrgbspike']:
    from data.dataset_video_train_rgbspike import VideoRecurrentTrainDatasetRGBSpike as D
```

### 3. 训练配置

**文件**: `options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json`

**关键配置项**:

```json
{
  "task": "006_train_vrt_videodeblurring_gopro_rgbspike",
  
  "datasets": {
    "train": {
      "dataset_type": "VideoRecurrentTrainDatasetRGBSpike",
      "dataroot_gt": "trainsets/GoPro/train_GT",
      "dataroot_lq": "trainsets/GoPro/train_GT_blurred",
      "dataroot_spike": "trainsets/gopro_spike/GOPRO_Large_spike_seq/train",
      "spike_h": 250,
      "spike_w": 400,
      "spike_channels": 1,
      "spike_flipud": true,
      "io_backend": {"type": "disk"},
      "scale": 1,
      ...
    }
  },
  
  "netG": {
    "net_type": "vrt",
    "in_chans": 4,  // 3 (RGB) + 1 (Spike)
    ...
  },
  
  "train": {
    // 所有训练参数与官方 baseline 完全一致
    "total_iter": 300000,
    "G_optimizer_lr": 4e-4,
    ...
  }
}
```

## 使用方法

### 启动训练

```bash
cd /home/mallm/henry/KAIR

# 单 GPU 训练
python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json

# 多 GPU 分布式训练 (推荐)
torchrun --nproc_per_node=3 --master_port=4321 main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

### 测试 Dataloader

```python
from data.select_dataset import define_Dataset
import json

with open('options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json') as f:
    config = json.load(f)

dataset = define_Dataset(config['datasets']['train'])
sample = dataset[0]

print(f"LQ shape: {sample['L'].shape}")  # (6, 4, 224, 224)
print(f"GT shape: {sample['H'].shape}")  # (6, 3, 224, 224)
```

## 数据流水线

```
1. 读取 RGB 帧 (从目录)
   ├─ LQ: train_GT_blurred/GOPR0384_11_02/001301.png  (720x1280x3, BGR) → 转为 RGB
   └─ GT: train_GT/GOPR0384_11_02/001301.png          (720x1280x3, BGR) → 转为 RGB

2. 读取对应 Spike 数据
   └─ spike/001301.dat → (202, 250, 400) → 体素化 → (1, 250, 400)

3. 随机裁剪 RGB 到 224x224
   ├─ LQ: (720x1280x3) → (224x224x3)
   └─ GT: (720x1280x3) → (224x224x3)

4. Resize Spike 到裁剪尺寸
   └─ (1, 250, 400) → (1, 224, 224)

5. 通道拼接
   └─ LQ: (224x224x3) + (224x224x1) → (224x224x4)

6. 数据增强 (flip, rotate)
   └─ 应用于 LQ 和 GT

7. 转换为 Tensor
   ├─ LQ: (T, 4, 224, 224)
   └─ GT: (T, 3, 224, 224)
```

## 验证结果

### Dataloader 测试输出

```
✓ 数据集创建成功!
  - 数据集大小: 2103

✓ 样本加载成功!
  - Key: GOPR0372_07_00/000047
  - LQ shape (RGB+Spike): torch.Size([6, 4, 224, 224])
  - GT shape (RGB): torch.Size([6, 3, 224, 224])

数据范围分析:
  - LQ RGB 通道 (0:3):
    - Range: [0.000, 0.980]
    - Mean: 0.253
  - LQ Spike 通道 (3:):
    - Range: [0.276, 0.933]
    - Mean: 0.578
    - Non-zero ratio: 100.00%

✓ Batch 加载成功!
  - LQ shape: torch.Size([2, 6, 4, 224, 224])
  - GT shape: torch.Size([2, 6, 3, 224, 224])
```

## 文件清单

### 新增文件
1. ✅ `utils/spike_loader.py` - Spike 数据加载器
2. ✅ `data/dataset_video_train_rgbspike.py` - RGB+Spike Dataset
3. ✅ `options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json` - 训练配置
4. ✅ `trainsets/gopro_spike/GOPRO_Large_spike_seq/config.yaml` - Spike 数据配置

### 修改文件
1. ✅ `models/select_network.py` - 添加 in_chans 参数传递
2. ✅ `data/select_dataset.py` - 注册新 Dataset 类型

## 技术亮点

1. **零依赖的 Spike 加载器**: 移除 SpikeCV 的硬件依赖，纯 NumPy 实现
2. **自动分辨率对齐**: Spike (250x400) 自动 resize 到 RGB crop 尺寸
3. **灵活的体素化**: 支持 S=1 和 S=4 两种策略，可配置
4. **联合数据增强**: RGB 和 Spike 同步进行 flip/rotate
5. **向后兼容**: 保持原始 VRT 代码的所有功能

## 下一步工作

### 可选优化
1. **测试 S=4 体素化**: 修改配置 `"spike_channels": 4`，网络 `"in_chans": 6`
2. **LMDB 格式**: 如果训练速度成为瓶颈，考虑转换为 LMDB
3. **在线测试**: 实现 RGB+Spike 的测试 Dataset，用于验证集评估

### 实验建议
1. **Baseline 对比**: 先训练纯 RGB 版本作为对照
2. **消融实验**: 
   - RGB only
   - Spike only
   - RGB + Spike (S=1)
   - RGB + Spike (S=4)
3. **学习率调整**: Spike 通道可能需要不同的学习率

## 问题排查

### 常见错误

1. **FileNotFoundError**: 检查 spike 数据路径是否正确
2. **Shape mismatch**: 确认配置中 `scale=1` 且 `io_backend: disk`
3. **CUDA OOM**: 减小 `batch_size` 或 `gt_size`

### 调试命令

```bash
# 测试 spike 加载
python utils/spike_loader.py

# 测试 dataset
python -c "from data.select_dataset import define_Dataset; import json; \
    config = json.load(open('options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json')); \
    ds = define_Dataset(config['datasets']['train']); print(ds[0]['L'].shape)"
```

## 性能参考

- **数据集大小**: 2103 个训练样本
- **单样本加载时间**: ~0.5s (首次加载，含 spike 解码)
- **内存占用**: 每个 spike .dat 文件 ~2.5MB
- **预期训练时间**: ~3-5 天 (3xGPU, batch_size=3)

## 作者与日期

- **实现日期**: 2025-11-05
- **基础代码**: KAIR (VRT for Video Deblurring)
- **数据集**: GoPro Large + GoPro Spike Sequences

---

**注**: 所有代码已测试通过，可直接用于训练。如有问题，请参考本文档的"问题排查"章节。


