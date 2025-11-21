# VRT RGB+Spike 快速入门指南

## 环境准备

确保已激活正确的 conda 环境：
```bash
conda activate vrtspike
cd /home/mallm/henry/KAIR
pip install -r requirement.txt
```

## 一键启动训练

### 单卡训练
```bash
python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

### 多卡训练 (推荐 - 3 GPUs)
```bash
torchrun --nproc_per_node=3 --master_port=4321 main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

## 快速测试

### 测试 Spike 加载器
```bash
python utils/spike_loader.py
```

**预期输出**:
```
测试 spike 加载...
Loading total spikes from dat file -- spatial resolution: 400 x 250, total timestamp: 202
结果:
  Shape: (202, 250, 400)
  Dtype: bool
  Range: [False, True]
  Non-zero ratio: 20.17%
```

### 测试 Dataset
```bash
python -c "
from data.select_dataset import define_Dataset
import json

with open('options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json') as f:
    config = json.load(f)

dataset = define_Dataset(config['datasets']['train'])
print(f'Dataset size: {len(dataset)}')

sample = dataset[0]
print(f'LQ shape: {sample[\"L\"].shape}')  # Should be (6, 4, 224, 224)
print(f'GT shape: {sample[\"H\"].shape}')  # Should be (6, 3, 224, 224)
print('✓ Dataset test passed!')
"
```

## 配置调整

### 修改 Spike 通道数 (S=1 → S=4)

编辑 `options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json`:

```json
{
  "datasets": {
    "train": {
      "spike_channels": 4,  // 改为 4
      ...
    }
  },
  "netG": {
    "in_chans": 6,  // 3 (RGB) + 4 (Spike) = 7? 不对，应该是 7
    ...
  }
}
```

**注意**: 记得同步修改 `in_chans = 3 + spike_channels`

### 调整 Batch Size

如果 GPU 内存不足：
```json
{
  "datasets": {
    "train": {
      "dataloader_batch_size": 2,  // 默认是 3
      "gt_size": 192,  // 或减小 crop 尺寸
      ...
    }
  }
}
```

### 调整学习率

```json
{
  "train": {
    "G_optimizer_lr": 2e-4,  // 默认 4e-4
    ...
  }
}
```

## 监控训练

### 查看日志
```bash
tail -f experiments/006_train_vrt_videodeblurring_gopro_rgbspike/train_*.log
```

### TensorBoard (如果启用)
```bash
tensorboard --logdir experiments/006_train_vrt_videodeblurring_gopro_rgbspike/
```

### 检查 GPU 使用
```bash
watch -n 1 nvidia-smi
```

## 训练检查点

模型会自动保存在:
```
experiments/006_train_vrt_videodeblurring_gopro_rgbspike/
├── models/
│   ├── 5000_G.pth
│   ├── 10000_G.pth
│   └── ...
└── training_states/
    ├── 5000.state
    └── ...
```

## 恢复训练

如果训练中断，自动从最新检查点恢复：
```bash
# 会自动检测并加载最新的 checkpoint
python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

## 常见问题

### Q1: FileNotFoundError - spike data not found
**A**: 检查 spike 数据路径:
```bash
ls trainsets/gopro_spike/GOPRO_Large_spike_seq/train/GOPR0384_11_02/spike/001301.dat
```

### Q2: CUDA out of memory
**A**: 减小 batch size 或 crop size:
```json
"dataloader_batch_size": 2,
"gt_size": 192
```

### Q3: Scale mismatches error
**A**: 确保配置中有 `"scale": 1`:
```json
{
  "datasets": {
    "train": {
      "scale": 1,
      ...
    }
  }
}
```

### Q4: 训练速度太慢
**A**: 
1. 减少 `dataloader_num_workers`
2. 使用 SSD 存储数据
3. 考虑转换为 LMDB 格式（需要额外工作）

## 对比实验

建议训练顺序：

1. **Baseline (RGB only)**:
   ```bash
   python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro.json
   ```

2. **RGB + Spike (S=1)** - 本配置
   ```bash
   python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
   ```

3. **RGB + Spike (S=4)** - 修改配置后
   ```bash
   # 修改 spike_channels=4, in_chans=7
   python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
   ```

## 预期结果

- **训练时间**: 约 3-5 天 (3x GPU, 300k iterations)
- **首次迭代**: ~10-20s (数据加载 + 模型初始化)
- **后续迭代**: ~0.5-1s per iteration
- **显存占用**: ~10-12GB per GPU (batch_size=3)

## 数据统计

- **训练样本数**: 2103
- **每个 epoch**: ~700 iterations (batch_size=3)
- **总 epochs**: ~428 (300k iterations / 700)

## 快速验证清单

- [ ] Conda 环境已激活 (`conda activate vrtspike`)
- [ ] RGB 数据存在 (`ls trainsets/GoPro/train_GT/`)
- [ ] Spike 数据存在 (`ls trainsets/gopro_spike/GOPRO_Large_spike_seq/train/`)
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] Dataset 测试通过 (见上方测试命令)
- [ ] 配置文件正确 (`cat options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json`)

---

**准备好了吗？运行训练命令开始实验！** 🚀


