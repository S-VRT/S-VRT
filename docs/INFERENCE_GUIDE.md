# 推理策略完整指南

本文档整合了验证推理策略和Tile Inference的完整说明。

---

## 目录

1. [训练vs验证的裁剪策略](#1-训练vs验证的裁剪策略)
2. [Tile Inference详解](#2-tile-inference详解)
3. [其他主流推理方法](#3-其他主流推理方法)
4. [方法选择指南](#4-方法选择指南)
5. [实际应用建议](#5-实际应用建议)
6. [命令行使用](#6-命令行使用)

---

## 1. 训练vs验证的裁剪策略

### 1.1 训练时裁剪 (split == "train")

```python
# 训练时使用随机裁剪
if split == "train":
    crop_size = 256
    # 随机裁剪增加数据多样性，提高泛化能力
```

**训练裁剪的特点：**
- **随机性**：每次训练时裁剪位置不同，增加数据多样性
- **小尺寸**：256×256便于GPU内存管理和训练效率
- **数据增强**：通过随机裁剪模拟不同场景，提高模型鲁棒性

### 1.2 验证时策略 (split == "val")

```python
# 验证时保持原始尺寸
if split == "val":
    crop_size = None  # 不裁剪，保持原始尺寸
```

**验证不裁剪的原因：**
1. **真实评估**：保持原始分辨率，评估模型在真实场景下的性能
2. **一致性**：避免随机性影响评估结果的可重复性
3. **完整性**：保留完整图像信息，不丢失边缘细节

### 1.3 验证时裁剪的注意事项

如果验证时也需要裁剪，应该注意：

#### 确定性裁剪
```python
# 使用中心裁剪而非随机裁剪
def center_crop(image, crop_size):
    h, w = image.shape[-2:]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return image[..., start_h:start_h+crop_size, start_w:start_w+crop_size]
```

#### 多尺度评估
```python
# 在不同尺度下评估
scales = [256, 512, 1024]
for scale in scales:
    cropped = center_crop(image, scale)
    result = model(cropped)
```

---

## 2. Tile Inference详解

### 2.1 概述

Tile Inference实现了对大尺寸图像的分块推理，避免内存溢出（OOM）问题。

**核心功能：**
- 自动分块决策
- 平滑边界融合
- 完整错误处理
- GPU加速支持

### 2.2 核心原理

#### 步骤1: 计算tile网格
```python
# 计算需要的tile数量
stride = tile_size - tile_overlap
h_tiles = (H - tile_overlap + stride - 1) // stride
w_tiles = (W - tile_overlap + stride - 1) // stride
```

#### 步骤2: 创建权重图
```python
# 创建cosine窗函数用于平滑融合
tile_weight = torch.ones(1, 1, 1, tile_size, tile_size)
if tile_overlap > 0:
    fade = torch.linspace(0, 1, tile_overlap)
    fade_window = torch.cos((1 - fade) * torch.pi / 2)
    # 应用到四个边缘，实现平滑过渡
    tile_weight[:, :, :, :tile_overlap, :] *= fade_window.view(1, 1, 1, -1, 1)
```

#### 步骤3: 逐tile处理
```python
for i in range(h_tiles):
    for j in range(w_tiles):
        # 计算tile坐标
        h_start = i * stride
        w_start = j * stride
        h_end = min(h_start + tile_size, H)
        w_end = min(w_start + tile_size, W)
        
        # 提取tile
        blur_tile = blur[:, :, :, h_start:h_end, w_start:w_end]
        spike_vox_tile = spike_vox[:, :, :, h_s_start:h_s_end, w_s_start:w_s_end]
        
        # 推理
        recon_tile = model(blur_tile, spike_vox_tile)
        
        # 加权累加
        output[:, :, :, h_start:h_end, w_start:w_end] += recon_tile * current_weight
        weight[:, :, :, h_start:h_end, w_start:w_end] += current_weight
```

#### 步骤4: 归一化融合
```python
# 按权重归一化
output = output / (weight + 1e-8)
```

### 2.3 自动分块决策

```python
# 自动判断是否需要分块
def should_tile(image_size, tile_size, force_enable, force_disable):
    if force_disable:
        return False
    if force_enable:
        return True
    # 自动决策：图像大于tile_size时启用
    return image_size[0] > tile_size or image_size[1] > tile_size
```

**决策逻辑：**
- 小图像直接处理，避免不必要开销
- 大图像自动启用分块
- 支持命令行强制控制

### 2.4 优势与挑战

**优势：**
✅ **内存友好**：避免大图像导致的内存溢出  
✅ **灵活性**：可以处理任意尺寸的图像  
✅ **并行性**：tile之间可以并行处理  
✅ **平滑融合**：余弦窗函数避免边界伪影

**挑战：**
⚠️ **边界效应**：tile边界可能出现伪影（已通过重叠解决）  
⚠️ **计算开销**：重叠区域导致重复计算  
⚠️ **融合质量**：权重融合可能影响最终质量

---

## 3. 其他主流推理方法

### 3.1 滑动窗口 (Sliding Window)

```python
def sliding_window_inference(model, image, window_size, stride):
    """滑动窗口推理"""
    results = []
    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            window = image[y:y+window_size, x:x+window_size]
            result = model(window)
            results.append((x, y, result))
    return merge_results(results)
```

**特点：**
- 固定步长滑动
- 适用于目标检测、分割任务
- 计算量大但结果稳定

### 3.2 多尺度推理 (Multi-scale)

```python
def multi_scale_inference(model, image, scales=[0.5, 1.0, 1.5]):
    """多尺度推理"""
    results = []
    for scale in scales:
        scaled_image = resize(image, scale)
        result = model(scaled_image)
        results.append(resize(result, 1/scale))
    return fuse_multi_scale(results)
```

**特点：**
- 提高对不同尺寸目标的检测能力
- 计算成本高
- 需要后处理融合

### 3.3 金字塔推理 (Pyramid)

```python
def pyramid_inference(model, image, levels=3):
    """金字塔推理"""
    pyramid = build_pyramid(image, levels)
    results = []
    for level in pyramid:
        result = model(level)
        results.append(result)
    return reconstruct_from_pyramid(results)
```

**特点：**
- 从粗到细的处理
- 适合多尺度特征提取
- 计算复杂度高

### 3.4 全图推理 (Full Image)

```python
def full_image_inference(model, image):
    """全图推理"""
    # 直接处理整个图像
    return model(image)
```

**特点：**
- 最简单直接
- 需要足够的内存
- 保持全局一致性

---

## 4. 方法选择指南

### 4.1 根据任务选择

| 任务类型 | 推荐方法 | 原因 |
|---------|---------|------|
| **图像复原** | Tile Inference + 重叠融合 | 保持全局一致性 |
| **目标检测** | 滑动窗口 + NMS | 适合多目标场景 |
| **语义分割** | Tile Inference + 重叠融合 | 边界平滑过渡 |
| **超分辨率** | Tile Inference + 边缘处理 | 避免边界伪影 |

### 4.2 根据资源选择

| GPU内存 | 推荐配置 | 说明 |
|---------|---------|------|
| **< 8GB** | `tile_size=128, overlap=16` | 小块处理 |
| **8-16GB** | `tile_size=256, overlap=32` | 平衡配置 |
| **> 16GB** | `tile_size=512, overlap=64` | 高质量模式 |

### 4.3 根据图像尺寸选择

| 图像尺寸 | 推荐策略 | 原因 |
|---------|---------|------|
| **< 512px** | 全图推理 | 无需分块 |
| **512-2048px** | Tile Inference | 平衡性能和质量 |
| **> 2048px** | 多尺度 + Tile | 处理超大图像 |

### 4.4 质量配置建议

| 质量要求 | 重叠配置 | 说明 |
|---------|---------|------|
| **高质量** | `overlap = tile_size // 4` | 25%重叠 |
| **平衡** | `overlap = tile_size // 8` | 12.5%重叠 |
| **快速** | `overlap = tile_size // 16` | 6.25%重叠 |

---

## 5. 实际应用建议

### 5.1 典型使用场景

#### 场景1: 常规验证评估
```bash
# 使用默认配置，适合大多数情况
python tools/test.py configs/deblur/vrt_spike_baseline.yaml
```
**适用于：**
- 标准分辨率图像 (< 1024px)
- 足够的GPU内存
- 需要快速验证

#### 场景2: 超大图像处理
```bash
# 启用tile inference处理大图像
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference \
  --tile_size 256 \
  --tile_overlap 32
```
**适用于：**
- 高分辨率图像 (> 2048px)
- 内存受限环境
- 需要处理完整图像

#### 场景3: 高质量推理
```bash
# 增大重叠区域以提高质量
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference \
  --tile_size 512 \
  --tile_overlap 128
```
**适用于：**
- 对质量要求极高
- GPU内存充足
- 可以接受较长处理时间

#### 场景4: 内存受限环境
```bash
# 使用小tile_size降低内存占用
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference \
  --tile_size 128 \
  --tile_overlap 16
```
**适用于：**
- GPU内存不足 (< 8GB)
- 需要在低配设备运行
- 优先保证程序运行

### 5.2 性能优化技巧

#### 技巧1: 合理设置批处理大小
```yaml
# 在配置文件中调整
dataloader:
  val:
    batch_size: 1  # tile inference时建议使用1
```

#### 技巧2: 预热GPU
```python
# 首次推理前进行预热
dummy_input = torch.randn(1, 3, 256, 256).cuda()
model(dummy_input)
torch.cuda.synchronize()
```

#### 技巧3: 使用混合精度
```python
# 启用自动混合精度
with torch.cuda.amp.autocast():
    output = tile_inference(model, input)
```

### 5.3 常见问题排查

#### 问题1: 内存溢出 (OOM)
**症状：** `RuntimeError: CUDA out of memory`

**解决方案：**
1. 启用tile inference
2. 减小tile_size
3. 减小batch_size
4. 减小tile_overlap

```bash
# 逐步降低tile_size
python tools/test.py config.yaml --tile_inference --tile_size 128
```

#### 问题2: 边界伪影
**症状：** tile边界出现明显接缝

**解决方案：**
1. 增加tile_overlap
2. 检查权重融合是否正确
3. 使用更平滑的窗函数

```bash
# 增加重叠区域
python tools/test.py config.yaml --tile_inference --tile_overlap 64
```

#### 问题3: 处理速度慢
**症状：** 推理时间过长

**解决方案：**
1. 减小tile_overlap
2. 增大tile_size（如果内存允许）
3. 禁用不必要的tile inference

```bash
# 对小图像禁用tile inference
python tools/test.py config.yaml --no_tile_inference
```

---

## 6. 命令行使用

### 6.1 基本参数

```bash
python tools/test.py <config_file> [OPTIONS]
```

**必需参数：**
- `config_file`: 配置文件路径

**可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--tile_inference` | flag | False | 启用tile inference |
| `--no_tile_inference` | flag | False | 强制禁用tile inference |
| `--tile_size` | int | 256 | tile的尺寸 |
| `--tile_overlap` | int | 32 | tile之间的重叠像素 |
| `--checkpoint` | str | None | 模型检查点路径 |
| `--save_results` | flag | False | 保存推理结果 |

### 6.2 使用示例

#### 示例1: 基本使用
```bash
python tools/test.py configs/deblur/vrt_spike_baseline.yaml
```

#### 示例2: 启用tile inference
```bash
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference
```

#### 示例3: 自定义tile参数
```bash
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference \
  --tile_size 512 \
  --tile_overlap 64
```

#### 示例4: 指定检查点
```bash
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --checkpoint experiments/vrt_spike/checkpoints/best.pth \
  --tile_inference
```

#### 示例5: 保存结果
```bash
python tools/test.py configs/deblur/vrt_spike_baseline.yaml \
  --tile_inference \
  --save_results
```

### 6.3 配置文件中的设置

除了命令行参数，也可以在配置文件中设置tile inference参数：

```yaml
# configs/deblur/vrt_spike_baseline.yaml

inference:
  tile_inference:
    enable: true          # 启用tile inference
    tile_size: 256       # tile尺寸
    tile_overlap: 32     # 重叠像素
    auto_decide: true    # 自动决策是否启用
```

**优先级：**
命令行参数 > 配置文件 > 默认值

### 6.4 高级用法

#### 批量处理
```bash
# 处理多个配置
for config in configs/deblur/*.yaml; do
  python tools/test.py $config --tile_inference
done
```

#### 使用GPU指定
```bash
# 指定GPU设备
CUDA_VISIBLE_DEVICES=0 python tools/test.py config.yaml --tile_inference
```

#### 组合多个选项
```bash
# 完整配置示例
CUDA_VISIBLE_DEVICES=0,1 python tools/test.py \
  configs/deblur/vrt_spike_baseline.yaml \
  --checkpoint experiments/best.pth \
  --tile_inference \
  --tile_size 512 \
  --tile_overlap 64 \
  --save_results
```

---

## 总结

### 核心要点

1. **训练 vs 验证**
   - 训练使用随机裁剪增强数据
   - 验证保持原始尺寸评估真实性能

2. **Tile Inference**
   - 解决大图像内存溢出问题
   - 通过重叠和加权融合保证质量
   - 自动决策何时启用

3. **方法选择**
   - 根据任务、资源、图像尺寸选择合适方法
   - 平衡质量、速度、内存占用

4. **实践建议**
   - 常规场景使用默认配置
   - 大图像启用tile inference
   - 遇到问题查看问题排查章节

### 推荐配置

| 场景 | tile_size | tile_overlap | 说明 |
|------|-----------|--------------|------|
| **快速测试** | 256 | 16 | 最快速度 |
| **标准使用** | 256 | 32 | 平衡推荐 |
| **高质量** | 512 | 64 | 最佳质量 |
| **内存受限** | 128 | 16 | 最低内存 |

### 参考资源

- 代码实现：`src/models/common/tile_inference.py`
- 配置示例：`configs/deblur/vrt_spike_baseline.yaml`
- 测试脚本：`tools/test.py`

---

**最后更新：** 2025年10月21日  
**版本：** v1.0

