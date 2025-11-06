# RGB+Spike 融合关键设计决策

## 完整数据流水线总结

本节提供 RGB+Spike 融合的端到端数据流水线概览，展示从数据加载到模型输出的完整过程。

### 1. 数据加载阶段（真实concat）

**文件**: `data/dataset_video_train_rgbspike.py`

```python
# RGB: (720, 1280, 3)
# Spike: (202, 250, 400) → voxelize → (1, 250, 400)

# 分辨率对齐
RGB:      crop to (224, 224, 3)
Spike:    resize to (224, 224, 1)

# 通道拼接 - 真实 concat！
for img_lq, spike_lq in zip(img_lqs, spike_lqs_processed):
    img_combined = np.concatenate([img_lq, spike_lq], axis=2)  # (224, 224, 4)
```

**输出**:
- `LQ`: `(T, 4, H, W)` - 4 通道 = `[R, G, B, Spike]`
- `GT`: `(T, 3, H, W)` - 3 通道 = `[R, G, B]`

### 2. 模型输入层适配

**文件**: `models/network_vrt.py` - `__init__` 方法

```python
# in_chans=4 时，第一层卷积自动适配
conv_first_in_chans = in_chans * (1 + 2*pa_frames)  # 4 * 9 = 36
self.conv_first = nn.Conv3d(conv_first_in_chans, embed_dims[0], 
                            kernel_size=(1,3,3), padding=(0,1,1))
```

**通道计算**:
- `4`: RGB+Spike 的 4 个通道
- `*(1+2*4)`: 当前帧 + 前后各 2 帧的时间对齐（共 9 帧）
- 总计: `36` 个输入通道

### 3. 光流计算（智能分离）

**文件**: `models/network_vrt.py` - `get_flow_2frames` 方法

```python
# 只用 RGB 通道计算光流
if c > 3:
    x_rgb = x[:, :, :3, :, :]  # 提取前 3 个通道
else:
    x_rgb = x

flows_backward = self.spynet(x_rgb)  # SpyNet 只接受 3 通道
```

**关键点**:
- ✅ SpyNet 在 RGB 上预训练，只能接受 3 通道
- ✅ 光流依赖 RGB 的纹理和边缘信息
- ✅ **完整的 4 通道数据仍然参与特征提取！**

### 4. 特征对齐和提取

```
输入 [B, N, 4, H, W]
      |
      ├─→ RGB [B,N,3,H,W] ─→ SpyNet ─→ 光流 [B,N-1,2,H,W]
      |                                    |
      └────────────────────────────────────┼─→ DCN 对齐
                                           |   (4 通道 + 光流)
                                           v
                                    Transformer
                                    (特征提取)
```

### 5. 输出和残差连接

**文件**: `models/network_vrt.py` - `forward` 方法

```python
# 输出为 3 通道（RGB）
x = self.conv_last(x)  # (B, N, 3, H, W)

# 残差连接：输入 4 通道，输出 3 通道
if x_lq.shape[2] != x.shape[2]:
    return x + x_lq[:, :, :self.out_chans, :, :]  # 只用前 3 通道
```

### 完整流水线图

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载                                                  │
├─────────────────────────────────────────────────────────────┤
│ RGB LQ:   (720, 1280, 3)                                    │
│ Spike:    (202, 250, 400) → voxelize → (1, 250, 400)       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. 分辨率对齐                                                │
├─────────────────────────────────────────────────────────────┤
│ RGB:      crop to (224, 224, 3)                             │
│ Spike:    resize to (224, 224, 1)                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. 通道拼接 ✅ [真实 concat]                                 │
├─────────────────────────────────────────────────────────────┤
│ Combined: (224, 224, 4) = [R, G, B, Spike]                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. 转为 Tensor                                               │
├─────────────────────────────────────────────────────────────┤
│ LQ:       (T, 4, 224, 224)  ← RGB+Spike                     │
│ GT:       (T, 3, 224, 224)  ← 纯 RGB                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. 模型处理                                                  │
├─────────────────────────────────────────────────────────────┤
│ • 输入 conv_first: 4 channels → 36 channels                 │
│ • 光流计算: 只用前 3 通道（RGB）                             │
│ • 特征提取: 使用完整的 4 通道                                │
│ • 输出: 3 channels（RGB）                                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键优点

1. ✅ **真实拼接**：使用 numpy/torch 的 concat 操作，不是硬塞
2. ✅ **参数匹配**：`in_chans=4` 正确配置了网络输入
3. ✅ **智能分离**：光流用 RGB，特征用全部
4. ✅ **向后兼容**：纯 RGB 模型（`in_chans=3`）仍然可用
5. ✅ **自动对齐**：Spike 分辨率自动 resize 到 RGB 大小

### 设计原则

这是一个**架构级别的优雅设计**，而非简单的数据硬塞：
- **模态特异性**：不同模态在不同阶段发挥作用
- **任务分工**：RGB 负责空间结构，Spike 补充时序信息
- **参数高效**：保持预训练模型权重有效性
- **可解释性**：每个阶段的输入输出都有明确的物理意义

---

## 光流计算只使用 RGB 通道

### 问题
当输入是 RGB+Spike (4通道) 时，SpyNet 光流网络如何处理？

### 解决方案
在 `get_flow_2frames` 方法中，**只提取前 3 个 RGB 通道用于光流计算**，Spike 通道不参与光流估计。

### 理由

#### 1. SpyNet 预训练限制
SpyNet 在 RGB 图像上预训练，使用 ImageNet 标准化参数：
```python
self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
```
- 只支持 **3 通道输入**
- 预训练权重对第 4 个通道无效
- 强行输入 4 通道会导致维度不匹配

#### 2. 光流的物理意义
- **光流**：描述像素级别的运动向量 (u, v)
- **RGB 图像**：包含清晰的边缘、纹理、颜色等空间特征，非常适合计算光流
- **Spike 数据**：是事件流的累积，统计特性与 RGB 完全不同

#### 3. Spike 的时间特性
- **光流**：描述的是 **帧间运动** (t → t+1)
- **Spike**：本身就是 **高时间分辨率的运动信息** (微秒级)
- 如果把 Spike 也输入光流网络，会产生 **信息冗余和混淆**

#### 4. 多模态融合分工原则

VRT 的多模态融合采用 **分阶段处理**：

| 阶段 | 模块 | 输入 | 作用 |
|------|------|------|------|
| **1. 光流估计** | SpyNet | **仅 RGB** (3 通道) | 估计帧间运动向量 |
| **2. 特征对齐** | DCN (可变形卷积) | **RGB+Spike** (4 通道) + 光流 | 使用光流引导对齐多模态特征 |
| **3. 特征提取** | Transformer | **对齐后的 RGB+Spike** | 深度特征学习和重建 |

这样的设计：
- ✅ **保持 SpyNet 预训练权重有效性**
- ✅ **让 RGB 和 Spike 各司其职**
  - RGB：提供空间结构和运动估计
  - Spike：提供高时间分辨率的运动细节
- ✅ **在对齐阶段才融合多模态信息**

### 代码实现

**修改位置**: `models/network_vrt.py` 第 1489-1514 行

```python
def get_flow_2frames(self, x):
    '''Get flow between frames t and t+1 from x.'''

    b, n, c, h, w = x.size()
    
    # For RGB+Spike input (c=4), only use RGB channels (first 3) for optical flow
    # SpyNet is pretrained on RGB images and expects 3 channels
    if c > 3:
        x_rgb = x[:, :, :3, :, :]  # Extract RGB channels only
    else:
        x_rgb = x
    
    x_1 = x_rgb[:, :-1, :, :, :].reshape(-1, 3, h, w)
    x_2 = x_rgb[:, 1:, :, :, :].reshape(-1, 3, h, w)

    # backward
    flows_backward = self.spynet(x_1, x_2)
    flows_backward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                      zip(flows_backward, range(4))]

    # forward
    flows_forward = self.spynet(x_2, x_1)
    flows_forward = [flow.view(b, n-1, 2, h // (2 ** i), w // (2 ** i)) for flow, i in
                     zip(flows_forward, range(4))]

    return flows_backward, flows_forward
```

### 数据流示意图

```
输入: RGB+Spike [B, N, 4, H, W]
                |
                ├─────────────────┐
                |                 |
                v                 v
        [B,N,3,H,W]         [B,N,4,H,W]
        (仅 RGB)            (完整)
                |                 |
                v                 |
            SpyNet                |
        (光流估计)                |
                |                 |
                v                 |
        flows [B,N-1,2,H,W]       |
                |                 |
                └────────┬─────────┘
                         v
                   DCN 对齐
                (使用光流引导)
                         |
                         v
                  Transformer
                  (特征提取)
                         |
                         v
                    输出重建
```

## 总结

这个设计体现了 **模态特异性** 和 **任务分工** 的原则：
- 光流网络专注于 RGB 的空间运动估计
- Spike 在后续阶段补充时序信息
- 避免了预训练模型的维度冲突
- 保持了模型的可解释性和可扩展性

如果未来需要让 Spike 也参与光流估计，可以：
1. 训练一个 4 通道的 SpyNet
2. 或者设计双分支光流网络（RGB 分支 + Spike 分支）

但当前方案是最简单、最稳定的实现方式。

