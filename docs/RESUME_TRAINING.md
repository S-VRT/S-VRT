# 训练恢复功能使用说明

## 概述

项目现已支持从保存的检查点恢复训练。训练过程中会定期保存完整的训练状态，包括：

- 模型权重 (state_dict)
- 优化器状态 (optimizer)
- 学习率调度器状态 (scheduler)
- 混合精度训练状态 (AMP scaler)
- 当前训练步数 (step)
- 当前epoch (epoch)
- 最佳PSNR (best_psnr)

## 使用方法

### 1. 从最新检查点恢复训练

```bash
python src/train.py \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --resume outputs/ckpts/last.pth
```

### 2. 从最佳检查点恢复训练

```bash
python src/train.py \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --resume outputs/ckpts/best.pth
```

### 3. 多GPU分布式训练恢复

```bash
torchrun --nproc_per_node=4 --master_port=29500 \
    src/train.py \
    --config configs/deblur/vrt_spike_baseline.yaml \
    --resume outputs/ckpts/last.pth
```

或使用训练脚本：

```bash
# 编辑 scripts/train_multi_gpu.sh，添加 --resume 参数
./scripts/train_multi_gpu.sh
```

## 检查点保存位置

训练过程中，检查点会自动保存到：

```
outputs/
└── ckpts/
    ├── last.pth      # 每N步保存一次（由 VAL_EVERY_STEPS 控制）
    └── best.pth      # 验证PSNR最佳时保存
```

## 检查点内容

每个检查点文件包含：

```python
{
    'step': 当前训练步数,
    'epoch': 当前epoch,
    'state_dict': 模型权重,
    'optimizer': 优化器状态,
    'scheduler': 学习率调度器状态,
    'scaler': AMP缩放器状态 (如果使用混合精度),
    'best_psnr': 最佳PSNR值
}
```

## 恢复训练的效果

当使用 `--resume` 参数时，训练脚本会：

1. ✅ 加载模型权重
2. ✅ 恢复优化器状态（包括动量等）
3. ✅ 恢复学习率调度器状态
4. ✅ 恢复混合精度训练状态
5. ✅ 从正确的训练步数继续
6. ✅ 保留历史最佳PSNR记录

## 示例输出

```
[train] Loading checkpoint from outputs/ckpts/last.pth
[train] ✓ Model weights loaded
[train] ✓ Resuming from step 5000
[train] ✓ Best PSNR so far: 28.5432
[train] ✓ Optimizer state loaded
[train] ✓ Scheduler state loaded
[train] ✓ AMP scaler state loaded
[train] Resume complete. Continuing from step 5000/300000
```

## 注意事项

### 1. 配置文件兼容性

恢复训练时应使用**相同的配置文件**，或确保模型架构参数一致。如果需要修改训练超参数（如batch size、学习率等），可以修改配置文件，但：

- ✅ 可以修改：batch size、learning rate、数据增强参数
- ⚠️ 需要谨慎：optimizer类型、scheduler类型
- ❌ 不能修改：模型架构参数（channels、heads等）

### 2. Batch Size 变化

如果修改了batch size或gradient accumulation steps：
- 模型权重会正确加载
- 优化器状态会加载（但动量缓冲区大小可能不匹配）
- 学习率调度器会继续，但step计数方式可能不同

### 3. 学习率调整

如果需要修改学习率但保留模型权重，有两种方式：

**方式1：只加载模型权重（手动）**
```python
# 在代码中只加载 state_dict
checkpoint = torch.load('outputs/ckpts/last.pth')
model.load_state_dict(checkpoint['state_dict'])
# 不加载 optimizer 和 scheduler
```

**方式2：恢复后手动调整学习率**
```python
# 恢复训练后，optimizer 会使用检查点中的学习率
# 配置文件中的学习率不会覆盖已保存的状态
```

### 4. 中断训练场景

适用场景：
- 训练意外中断（断电、OOM、进程被杀等）
- 需要暂停训练腾出GPU资源
- 发现性能瓶颈，调整配置后继续训练
- 在不同机器上继续训练

## 高级用法

### 查看检查点信息

```python
import torch

ckpt = torch.load('outputs/ckpts/last.pth', map_location='cpu')
print(f"Step: {ckpt['step']}")
print(f"Epoch: {ckpt['epoch']}")
print(f"Best PSNR: {ckpt['best_psnr']:.4f}")
print(f"Contains optimizer: {'optimizer' in ckpt}")
print(f"Contains scheduler: {'scheduler' in ckpt}")
```

### 迁移学习：只加载模型权重

如果只想加载模型权重而不恢复训练状态，可以创建一个简化的检查点：

```python
import torch

# 加载完整检查点
full_ckpt = torch.load('outputs/ckpts/last.pth')

# 只保存模型权重
weights_only = {'state_dict': full_ckpt['state_dict']}
torch.save(weights_only, 'outputs/ckpts/weights_only.pth')
```

然后正常训练时就不会加载optimizer等状态。

## 故障排除

### 问题1: "Checkpoint not found"

**原因**: 检查点路径不存在

**解决方案**: 检查路径是否正确
```bash
ls -lh outputs/ckpts/
```

### 问题2: 加载后性能下降

**原因**: 可能加载了错误的检查点

**解决方案**: 
- 确认加载的是 `best.pth` 而非损坏的 `last.pth`
- 查看检查点保存时的验证指标

### 问题3: OOM错误

**原因**: Optimizer状态需要额外显存

**解决方案**: 
- optimizer的动量缓冲区需要约2倍模型参数的显存
- 如果显存不足，考虑只加载模型权重（不用--resume）

## 总结

恢复训练功能让长时间训练更加可靠和灵活：
- ✅ 自动保存完整训练状态
- ✅ 一行命令即可恢复训练
- ✅ 支持单GPU和多GPU分布式训练
- ✅ 保留所有训练进度信息

建议始终使用 `--resume` 参数恢复训练，以确保训练状态的连续性。

